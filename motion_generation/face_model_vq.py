import torch
from torch import nn
import torch.nn.functional as F

def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(channel, channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class Quantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super(Quantizer, self).__init__()

        self.e_dim = e_dim
        self.n_e = n_e
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vectort that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        :param z (B, seq_len, channel):
        :return z_q:
        """
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)

        # B x V
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q - z.detach())**2) + self.beta * \
               torch.mean((z_q.detach() - z)**2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        min_encodings = F.one_hot(min_encoding_indices, self.n_e).type(z.dtype)
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean*torch.log(e_mean + 1e-10)))
        return loss, z_q, min_encoding_indices, perplexity

    def map2index(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vectort that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        :param z (B, seq_len, channel):
        :return z_q:
        """
        assert z.shape[-1] == self.e_dim
        #print(z.shape)
        z_flattened = z.contiguous().view(-1, self.e_dim)
        #print(z_flattened.shape)

        # B x V
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        return min_encoding_indices.reshape(z.shape[0], -1)

    def get_codebook_entry(self, indices):
        """

        :param indices(B, seq_len):
        :return z_q(B, seq_len, e_dim):
        """
        index_flattened = indices.view(-1)
        z_q = self.embedding(index_flattened)
        z_q = z_q.view(indices.shape + (self.e_dim, )).contiguous()
        return z_q

class VQEncoderV5_DS(nn.Module):
    def __init__(self, args):
        super(VQEncoderV5_DS, self).__init__()
        n_down = args.vae_layer
        channels = [args.vae_length]
        for i in range(n_down-1):
            channels.append(args.vae_length)
        
        input_size = args.vae_test_dim
        assert len(channels) == n_down
        layers = [
            nn.Conv1d(input_size, channels[0], 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(channels[0]),
        ]

        kernel_size = args.vae_stride * 2
        stride = args.vae_stride
        padding = args.vae_stride // 2
        for i in range(1, n_down):
            layers += [
                nn.Conv1d(channels[i-1], channels[i], kernel_size=kernel_size, stride=stride, padding=padding),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(channels[i]),
            ]
        self.main = nn.Sequential(*layers)
        # self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        # self.out_net.apply(init_weight)
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        # print(outputs.shape)
        return outputs

class VQDecoderV5DF_US(nn.Module):
    def __init__(self, args):
        super(VQDecoderV5DF_US, self).__init__()
        n_up = args.vae_layer
        channels = []
        for i in range(n_up-1):
            channels.append(args.vae_length)
        channels.append(args.vae_length)
        channels.append(args.pose_dims)
        input_size = args.vae_length
        n_resblk = 2
        assert len(channels) == n_up + 1
        # if input_size == channels[0]:
        #     layers = []
        # else:
        #     layers = [nn.Conv1d(input_size, channels[0], kernel_size=3, stride=1, padding=1)]
        layers = []
        self.dfb = DynamicFusionBlockV2(args, codebook_emf_input_size=input_size)

        for i in range(n_resblk):
            layers += [ResBlock(channels[0])]
        # channels = channels
        for i in range(n_up):
            up_factor = args.vae_stride if i < n_up - 1 else 1
            layers += [
                nn.Upsample(scale_factor=up_factor, mode='nearest'),
                nn.Conv1d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        layers += [nn.Conv1d(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1)]
        self.main = nn.Sequential(*layers)
        self.main.apply(init_weight)

    def forward(self, codebook_emf_inputs, af_emf_inputs=None):
        codebook_emf_inputs = codebook_emf_inputs.permute(0, 2, 1)
        df_output = self.dfb(codebook_emf=codebook_emf_inputs, af_emf=af_emf_inputs)
        outputs = self.main(df_output).permute(0, 2, 1)
        return outputs

class AudioFeatProjV3(nn.Module):
    def __init__(self, args):
        super(AudioFeatProjV3, self).__init__()
        channels = [args.vae_length] *2
        input_size = args.audio_feat_dims

        layers = []
        for i in range(len(channels)):
            layers += [
                # nn.Conv1d(channels[i], channels[i], kernel_size=6, stride=2, padding=2, dilation=3),
                nn.Conv1d(input_size, channels[i], kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                # ResBlock(channels[i]),
            ]
            input_size = channels[i]
        self.main = nn.Sequential(*layers)
        # self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        # self.out_net.apply(init_weight)
    def forward(self, inputs):
        outputs = self.main(inputs)
        # print(outputs.shape)
        return outputs

class AudioFeatProjV4(nn.Module):
    def __init__(self, args):
        super(AudioFeatProjV4, self).__init__()
        channels = [args.vae_length] * 3
        input_size = args.audio_feat_dims

        layers = []
        for i in range(len(channels)):
            layers += [
                # nn.Conv1d(channels[i], channels[i], kernel_size=6, stride=2, padding=2, dilation=3),
                nn.Conv1d(input_size, channels[i], kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                # ResBlock(channels[i]),
            ]
            input_size = channels[i]
        self.main = nn.Sequential(*layers)
        # self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        # self.out_net.apply(init_weight)
    def forward(self, inputs):
        outputs = self.main(inputs)
        # print(outputs.shape)
        return outputs

class DynamicFusionBlock(nn.Module):

    def __init__(self, args, codebook_emf_input_size):
        super(DynamicFusionBlock, self).__init__()
        self.codebook_emf_proj = nn.Conv1d(codebook_emf_input_size, args.vae_length, kernel_size=3, stride=1, padding=1)
        self.af_emf_proj = AudioFeatProjV3(args)
        
        # 动态权重因子 0.5初始化，开局五五开
        self.df_factor = nn.Parameter(torch.ones([1, args.vae_length, 1], requires_grad=True) / 2)
        self.register_parameter('df_factor', self.df_factor)
    
    def forward(self, codebook_emf, af_emf):
        codebook_emf_proj_out = self.codebook_emf_proj(codebook_emf)
        af_emf = af_emf.permute(0, 2, 1)
        af_emf = torch.nn.functional.interpolate(af_emf, size=(codebook_emf_proj_out.shape[-1]*4), mode='linear')
        af_emf_proj_out = self.af_emf_proj(af_emf)
        df_out = codebook_emf_proj_out * self.df_factor + (af_emf_proj_out * (1 - self.df_factor))
        return df_out

class DynamicFusionBlockV2(nn.Module):

    def __init__(self, args, codebook_emf_input_size):
        super(DynamicFusionBlockV2, self).__init__()
        self.codebook_emf_proj = nn.Conv1d(codebook_emf_input_size, args.vae_length, kernel_size=3, stride=1, padding=1)
        self.af_emf_proj = AudioFeatProjV4(args)
        
        # 动态权重因子 0.5初始化，开局五五开
        self.df_factor = nn.Parameter(torch.ones([1, args.vae_length, 1], requires_grad=True) / 2)
        self.register_parameter('df_factor', self.df_factor)
    
    def forward(self, codebook_emf, af_emf):
        codebook_emf_proj_out = self.codebook_emf_proj(codebook_emf)
        af_emf = af_emf.permute(0, 2, 1)
        af_emf = torch.nn.functional.interpolate(af_emf, size=(codebook_emf_proj_out.shape[-1]*8), mode='linear')
        af_emf_proj_out = self.af_emf_proj(af_emf)
        df_out = codebook_emf_proj_out * self.df_factor + (af_emf_proj_out * (1 - self.df_factor))
        return df_out

class Af2FaceVQVAEConvZeroStrideV3(nn.Module):
    def __init__(self, args):
        super(Af2FaceVQVAEConvZeroStrideV3, self).__init__()
        self.encoder = VQEncoderV5_DS(args)
        self.quantizer = Quantizer(args.vae_codebook_size, args.vae_length, args.vae_quantizer_lambda)
        self.decoder = VQDecoderV5DF_US(args)
        
    def forward(self, x, af_inputs):
        
        pre_latent = self.encoder(x)
        embedding_loss, vq_latent, index, perplexity = self.quantizer(pre_latent)
        rec_pose = self.decoder(vq_latent, af_inputs)
        return {
            "poses_feat":vq_latent,
            "embedding_loss":embedding_loss,
            "perplexity":perplexity,
            "rec_pose": rec_pose,
            "index": index
            }
    
    def map2index(self, inputs):
        pre_latent = self.encoder(inputs)
        index = self.quantizer.map2index(pre_latent)
        return index
    
    def map2latent(self, inputs):
        pre_latent = self.encoder(inputs)
        index = self.quantizer.map2index(pre_latent)
        z_q = self.quantizer.get_codebook_entry(index)
        return z_q
    
    def decode(self, index=None, af_inputs=None):
        if index == None:
            z_q = torch.zeros([af_inputs.shape[0], int(af_inputs.shape[1]/2.5), self.args])
        z_q = self.quantizer.get_codebook_entry(index)
        rec_pose = self.decoder(codebook_emf_inputs=z_q, af_emf_inputs=af_inputs)
        return rec_pose