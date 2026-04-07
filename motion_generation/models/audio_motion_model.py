"""
Audio-Motion Mask Transformer Model (Hubert Features Version)

输入: motion_1, motion_5 (各4个motion token) + 5帧的hubert audio特征
输出: 预测被mask的motion token (motion_2, motion_3, motion_4)

Audio特征(hubert layer9, 768dim)通过线性层投影后，以加法方式融合到对应帧的motion token embedding上。
Motion token: codebook_size=512, 4 quantizers, offset=512
vocab_size = 512 * 4 + 1 = 2049 (最后一个是mask token)

@File    :   audio_motion_model.py
@Time    :   2025/07/16
"""

import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.integrations import use_kernel_forward_from_hub
from transformers.activations import ACT2FN
from transformers import PretrainedConfig


class AudioMotionConfig(PretrainedConfig):
    model_type = "audio_motion_transformer"

    def __init__(
        self,
        hidden_size=512,
        num_layers=8,
        num_heads=16,
        intermediate_size=1536,
        max_position_embeddings=512,
        vocab_size=2049,        # 512 * 4 + 1 (mask token)
        codebook_size=512,      # 每个quantizer的codebook大小
        audio_feat_dim=768,     # hubert layer9 特征维度
        num_tokens_per_frame=4, # 每帧motion token数量 (4 quantizers)
        num_frames=5,           # 窗口帧数
        dropout=0.2,
        cond_drop_prob=0.2,
        rms_norm_eps=1e-6,
        hidden_act="gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size
        self.codebook_size = codebook_size
        self.audio_feat_dim = audio_feat_dim
        self.num_tokens_per_frame = num_tokens_per_frame
        self.num_frames = num_frames
        self.dropout = dropout
        self.cond_drop_prob = cond_drop_prob
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act


@use_kernel_forward_from_hub("RMSNorm")
class AudioMotionRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class AudioEncoder(nn.Module):
    """将hubert音频原始特征投影到hidden_size空间"""

    def __init__(self, audio_feat_dim, hidden_size, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(audio_feat_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, audio_features):
        """
        Args:
            audio_features: (batch_size, num_frames, audio_feat_dim)
        Returns:
            (batch_size, num_frames, hidden_size)
        """
        return self.proj(audio_features)


class PositionEmbedding(nn.Module):
    """可学习的位置嵌入，使用正弦初始化"""

    def __init__(self, max_position_embeddings, hidden_size):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self._init_embeddings(max_position_embeddings, hidden_size)

    def _init_embeddings(self, max_pos, hidden_size):
        position = torch.arange(max_pos).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size)
        )
        embeddings = torch.zeros(max_pos, hidden_size)
        embeddings[:, 0::2] = torch.sin(position * div_term)
        embeddings[:, 1::2] = torch.cos(position * div_term)
        self.position_embeddings.weight.data.copy_(embeddings)

    def forward(self, seq_len, device):
        """
        Args:
            seq_len: int
            device: torch.device
        Returns:
            (1, seq_len, hidden_size)
        """
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        return self.position_embeddings(position_ids)


class AudioMotionTransformer(PreTrainedModel):
    config_class = AudioMotionConfig

    def __init__(self, config: AudioMotionConfig):
        super().__init__(config)
        self.config = config

        # Motion token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Audio encoder: 768 -> hidden_size
        self.audio_encoder = AudioEncoder(
            config.audio_feat_dim, config.hidden_size, config.dropout
        )

        # Position embedding
        self.position_emb = PositionEmbedding(
            config.max_position_embeddings, config.hidden_size
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation=ACT2FN[config.hidden_act],
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers, enable_nested_tensor=False
        )

        # Output head
        self.out_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Normalization
        self.norm = AudioMotionRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights
        self.post_init()

    def _fuse_audio(self, token_embeddings, audio_features):
        """
        将音频特征融合到motion token embedding上。
        每帧的audio特征投影后加到该帧的所有4个motion token embedding上。

        Args:
            token_embeddings: (batch_size, seq_len, hidden_size)
                seq_len = num_frames * num_tokens_per_frame
            audio_features: (batch_size, num_frames, audio_feat_dim)
        Returns:
            (batch_size, seq_len, hidden_size) - 融合后的embedding
        """
        num_tokens_per_frame = self.config.num_tokens_per_frame

        # 投影音频特征: (B, num_frames, hidden_size)
        audio_emb = self.audio_encoder(audio_features)

        # 扩展到每帧的每个token: (B, num_frames, 1, hidden_size) -> (B, num_frames, 4, hidden_size)
        audio_emb = audio_emb.unsqueeze(2).expand(
            -1, -1, num_tokens_per_frame, -1
        )
        # reshape: (B, num_frames * 4, hidden_size) = (B, seq_len, hidden_size)
        audio_emb = audio_emb.reshape(
            audio_emb.size(0), -1, audio_emb.size(-1)
        )

        return token_embeddings + audio_emb

    def forward(self, input_ids, labels=None, audio_features=None):
        """
        Args:
            input_ids: (batch_size, seq_len) - Motion token IDs (with offset)
                seq_len = num_frames * num_tokens_per_frame = 5 * 4 = 20
            labels: (batch_size, seq_len) - Target token IDs, -100 for ignore
            audio_features: (batch_size, num_frames, audio_feat_dim)
                num_frames = 5, audio_feat_dim = 768
        Returns:
            If labels provided: (loss, predictions, accuracy)
            Else: logits
        """
        # Embed motion tokens: (B, seq_len, hidden_size)
        x = self.embed_tokens(input_ids)

        # Add position embeddings
        x = x + self.position_emb(x.size(1), x.device)

        # Fuse audio features
        if audio_features is not None:
            x = self._fuse_audio(x, audio_features)

        # Normalize and encode
        x = self.norm(x)
        output = self.encoder(x)  # (B, seq_len, hidden_size)

        # Get logits
        logits = self.out_head(output)  # (B, seq_len, vocab_size)

        if labels is None:
            return logits

        # Calculate loss (only on masked positions, labels=-100 for non-masked)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            # Only compute accuracy on non-ignored positions
            valid_mask = labels != -100
            if valid_mask.sum() > 0:
                acc = (preds[valid_mask] == labels[valid_mask]).float().mean()
            else:
                acc = torch.tensor(0.0)

        return loss, preds, acc

    def generate_sbs(self, input_ids, audio_features, generate_steps=1):
        """
        Step-by-step generation: 逐步填充mask token。
        按照预测概率从高到低依次填充。

        Args:
            input_ids: (batch_size, seq_len) - 包含mask token的输入
            audio_features: (batch_size, num_frames, audio_feat_dim)
            generate_steps: int - 生成步数
        Returns:
            output_ids: (batch_size, seq_len) - 填充后的token序列
        """
        mask_token_id = self.config.vocab_size - 1  # 2048
        num_tokens_per_frame = self.config.num_tokens_per_frame

        device = next(self.parameters()).device
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, device=device)
        if not isinstance(audio_features, torch.Tensor):
            audio_features = torch.tensor(audio_features, dtype=torch.float32, device=device)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(0)

        step_out = input_ids.clone()
        batch_size = step_out.size(0)

        # 找到所有mask位置
        mask_positions = [
            (step_out[b] == mask_token_id).nonzero(as_tuple=True)[0].tolist()
            for b in range(batch_size)
        ]

        total_masks = max(len(pos) for pos in mask_positions) if mask_positions else 0
        if total_masks == 0:
            return step_out

        masks_per_step = max(1, total_masks // generate_steps)

        for step in range(generate_steps):
            logits = self.forward(step_out, audio_features=audio_features)
            # logits: (B, seq_len, vocab_size)
            preds_probs, preds_index = logits.max(dim=-1)

            for b in range(batch_size):
                remaining = mask_positions[b]
                if not remaining:
                    continue

                # 按概率排序，选择概率最高的位置填充
                probs_at_mask = [(pos, preds_probs[b, pos].item()) for pos in remaining]
                probs_at_mask.sort(key=lambda x: x[1], reverse=True)

                num_to_fill = min(masks_per_step, len(probs_at_mask))
                if step == generate_steps - 1:
                    num_to_fill = len(probs_at_mask)  # 最后一步填充所有剩余

                for i in range(num_to_fill):
                    pos = probs_at_mask[i][0]
                    step_out[b, pos] = preds_index[b, pos]
                    mask_positions[b].remove(pos)

        return step_out

    def interpolate(self, motion_frame1, motion_frame5, audio_features_5frames):
        """
        插帧推理接口：给定第1帧和第5帧的motion token，以及5帧的audio特征，
        预测中间3帧的motion token。

        Args:
            motion_frame1: list of 4 ints - 第1帧的4个motion token (raw, 0~511)
            motion_frame5: list of 4 ints - 第5帧的4个motion token (raw, 0~511)
            audio_features_5frames: (5, 768) numpy array or tensor - 5帧的hubert特征
        Returns:
            list of lists - 5帧的motion tokens [[frame1], [frame2], [frame3], [frame4], [frame5]]
                每个frame是4个raw token (0~511)
        """
        mask_token_id = self.config.vocab_size - 1  # 2048
        ntpf = self.config.num_tokens_per_frame
        codebook_size = self.config.codebook_size

        # offset: [0, 512, 1024, 1536]
        offsets = [codebook_size * i for i in range(ntpf)]

        input_tokens = []
        # Frame 1 (known)
        for j in range(ntpf):
            input_tokens.append(motion_frame1[j] + offsets[j])
        # Frames 2, 3, 4 (masked)
        for _ in range(3 * ntpf):
            input_tokens.append(mask_token_id)
        # Frame 5 (known)
        for j in range(ntpf):
            input_tokens.append(motion_frame5[j] + offsets[j])

        device = next(self.parameters()).device
        input_ids = torch.tensor([input_tokens], device=device)

        if not isinstance(audio_features_5frames, torch.Tensor):
            audio_features_5frames = torch.tensor(
                audio_features_5frames, dtype=torch.float32
            )
        audio_feat = audio_features_5frames.unsqueeze(0).to(device)

        # Generate
        output = self.generate_sbs(input_ids, audio_feat, generate_steps=1)
        output = output[0].cpu().tolist()

        # Parse output into frames
        frames = []
        for f in range(self.config.num_frames):
            frame_tokens = []
            for j in range(ntpf):
                token_id = output[f * ntpf + j]
                frame_tokens.append(token_id - offsets[j])
            frames.append(frame_tokens)

        return frames
