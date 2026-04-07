"""
Copyright 2024 LINE Corporation
LINE Corporation licenses this file to you under the Apache License,
version 2.0 (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at:
    https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.
"""

from typing import List, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import transformers
import clip
import numpy as np

from .actor import ACTORStyleEncoder, ACTORStyleDecoder


class KLLoss:
    def __call__(self, q, p):
        mu_q, logvar_q = q
        mu_p, logvar_p = p

        log_var_ratio = logvar_q - logvar_p
        t1 = (mu_p - mu_q).pow(2) / logvar_p.exp()
        div = 0.5 * (log_var_ratio.exp() + t1 - 1 - log_var_ratio)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"
    
class InfoNCE_with_filtering:
    def __init__(self, temperature=0.7, threshold_selfsim=0.8):
        self.temperature = temperature
        self.threshold_selfsim = threshold_selfsim

    def get_sim_matrix(self, x, y):
        x_logits = torch.nn.functional.normalize(x, dim=-1)
        y_logits = torch.nn.functional.normalize(y, dim=-1)
        sim_matrix = x_logits @ y_logits.T
        return sim_matrix

    def __call__(self, x, y, sent_emb=None):
        bs, device = len(y), y.device
        sim_matrix = self.get_sim_matrix(x, y) / self.temperature

        if sent_emb is not None and self.threshold_selfsim:
            # put the threshold value between -1 and 1
            real_threshold_selfsim = 2 * self.threshold_selfsim - 1
            # Filtering too close values
            # mask them by putting -inf in the sim_matrix
            selfsim = sent_emb @ sent_emb.T
            selfsim_nodiag = selfsim - selfsim.diag().diag()
            idx = torch.where(selfsim_nodiag > real_threshold_selfsim)
            sim_matrix[idx] = -torch.inf

        labels = torch.arange(bs, device=device)

        #modified to not include augmented texts
        total_loss = (
            F.cross_entropy(sim_matrix[:bs], labels) + F.cross_entropy(sim_matrix.T, labels)
        ) / 2

        return total_loss
    
class TextEncoder(nn.Module):
    def __init__(self, model_name: str, token_num: int, vae: bool = True, trainable: bool = True) -> None:
        super().__init__()
        self.clip = False
        self.t5 = False
        if model_name == "ViT-B/32":
            self.clip = True
            clip_model, _ = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
            #clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
            self.text_model = clip_model.float()
        elif model_name == "t5-base" or model_name == "t5-large":
            self.t5 = True
            self.text_model = transformers.AutoModel.from_pretrained(model_name)
        else:
            self.text_model = transformers.AutoModel.from_pretrained("/data/home/jinch/tech_report/susu_avatar_training_gen_demo/ChronAccRet-master/bert-base-chinese")

        for param in self.text_model.parameters():
            param.requires_grad = trainable

        self.target_token_idx = 0
        self.token_num = token_num
        self.vae = vae

    def forward(self, input_ids, length):
        if self.clip:
            x = self.text_model.token_embedding(input_ids).type(self.text_model.dtype)
            x = x + self.text_model.positional_embedding.type(self.text_model.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.text_model.transformer(x)
            x = x.permute(1, 0, 2)
            last_hidden_state = self.text_model.ln_final(x).type(self.text_model.dtype)
            if self.vae == False:
                last_hidden_state = self.text_model.encode_text(input_ids).type(self.text_model.dtype)
                #last_hidden_state = last_hidden_state[torch.arange(last_hidden_state.shape[0]), input_ids.argmax(dim=-1)]# @ self.text_model.text_projection
            return last_hidden_state
        elif self.t5:
            mask = length_to_mask(length, self.token_num, length.device)
            decoder_input_ids = self.text_model._shift_right(input_ids)
            output = self.text_model(input_ids=input_ids,decoder_input_ids=decoder_input_ids, attention_mask=mask)
            last_hidden_state = output.last_hidden_state
            if self.vae == False:
                last_hidden_state = last_hidden_state[:, self.target_token_idx, :]
            return last_hidden_state
        else:
            mask = length_to_mask(length, self.token_num, length.device)
            output = self.text_model(input_ids=input_ids, attention_mask=mask)
            last_hidden_state = output.last_hidden_state
            if self.vae == False:
                last_hidden_state = last_hidden_state[:, self.target_token_idx, :]
            return last_hidden_state


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()

        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x += projected
        x = self.layer_norm(x)
        
        return x
        

def length_to_mask(length: List[int], max_length, device: torch.device) -> Tensor:
    if device is None:
        device = "cpu"
    if isinstance(length, list):
        length = torch.tensor(length, device=device)
    max_len = max(length)
    if max_len < max_length:
        max_len = max_length
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    return mask


class ChronTMR(nn.Module):
    def __init__(
        self,
        cfg,
        vae: bool,
        fact: Optional[float] = None,
        sample_mean: Optional[bool] = False
    ) -> None:
        super().__init__()

        self.text_model = TextEncoder(
            model_name=cfg.model.text_model_name, token_num=cfg.model.token_num, vae=cfg.text_encoder.vae, trainable=cfg.train.train_text_encoder
        )

        self.motion_encoder = ACTORStyleEncoder(
            cfg.motion_encoder.nfeats,
            cfg.motion_encoder.vae,
            cfg.motion_encoder.latent_dim,
            cfg.motion_encoder.ff_size,
            cfg.motion_encoder.num_layers,
            cfg.motion_encoder.num_heads,
            cfg.motion_encoder.dropout,
            cfg.motion_encoder.activation
        )
        if cfg.text_encoder.vae == True:
            self.text_encoder = ACTORStyleEncoder(
                cfg.text_encoder.nfeats,
                cfg.text_encoder.vae,
                cfg.text_encoder.latent_dim,
                cfg.text_encoder.ff_size,
                cfg.text_encoder.num_layers,
                cfg.text_encoder.num_heads,
                cfg.text_encoder.dropout,
                cfg.text_encoder.activation
            )
        else:
            self.text_encoder = ProjectionHead(
                cfg.text_encoder.nfeats,
                cfg.text_encoder.latent_dim,
                cfg.text_encoder.dropout
            )
        
        self.motion_decoder = ACTORStyleDecoder(
            cfg.motion_decoder.nfeats,
            cfg.motion_decoder.latent_dim,
            cfg.motion_decoder.ff_size,
            cfg.motion_decoder.num_layers,
            cfg.motion_decoder.num_heads,
            cfg.motion_decoder.dropout,
            cfg.motion_decoder.activation
        )
        self.cfg = cfg
        # sampling parameters
        self.textvae = cfg.text_encoder.vae
        self.motvae = cfg.motion_encoder.vae
        self.token_num = cfg.model.token_num
        self.fact = fact if fact is not None else 1.0
        self.sample_mean = sample_mean
        self.noneg = cfg.model.noneg

        # losses
        self.reconstruction_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")
        self.latent_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")
        self.kl_loss_fn = KLLoss()
        self.contrastive_loss_fn = InfoNCE_with_filtering(
            temperature=cfg.model.temperature, threshold_selfsim=cfg.model.threshold_selfsim
        )
        self.threshold_selfsim_metrics = cfg.model.threshold_selfsim_metrics

        # lambda weighting for the losses
        self.lmd = cfg.lmd
        print(self.lmd)

    

    def encode(
        self,
        input,
        lengths,
        modality: str = "txt",
        sample_mean: Optional[bool] = None,
        fact: Optional[float] = None,
        return_distribution: bool = False,
    ):
        sample_mean = self.sample_mean if sample_mean is None else sample_mean
        fact = self.fact if fact is None else fact
        if modality == "txt":
            mask = length_to_mask(lengths,self.token_num,lengths.device)
        else:
            max_motion_length = self.cfg.dataset.max_motion_length
            mask = length_to_mask(lengths,max_motion_length,lengths.device)
        inputs = {"x": input, "mask": mask}
        # Encode the inputs
        if modality == "txt":
            encoder = self.text_encoder
            
        else:
            encoder = self.motion_encoder

           
        
        # Sampling
        if return_distribution:
            encoded = encoder(inputs)
            dists = encoded.unbind(1)
            mu, logvar = dists
            if sample_mean:
                latent_vectors = mu
            else:
                # Reparameterization 
                std = logvar.exp().pow(0.5)
                eps = std.data.new(std.size()).normal_()
                latent_vectors = mu + fact * eps * std
        else:
            if modality == "txt":
                if self.textvae == False:
                    latent_vectors = encoder(input)
                else:
                    encoded = encoder(inputs)
                    (latent_vectors,) = encoded.unbind(1)
            else:
                encoded = encoder(inputs)
                (latent_vectors,) = encoded.unbind(1)
                
            dists = None


        return latent_vectors, dists

    def decode(
        self,
        latent_vectors: Tensor,
        lengths: Optional[List[int]] = None
    ):
        max_motion_length = self.cfg.dataset.max_motion_length
        mask = length_to_mask(lengths,max_motion_length,lengths.device)
        z_dict = {"z": latent_vectors, "mask": mask}
        motions = self.motion_decoder(z_dict)
        return motions


    def forward(self, texts, motions, sent_emb, shuffled_event, event, lengths, t_lengths, e_lengths, return_loss=False):

        ref_motions = motions
        bs = motions.shape[0]
        
        # embed texts into language encoder
        texts_emb = self.text_model(texts, t_lengths).float()
    
        # text -> motion
        t_latents, t_dists = self.encode(
            texts_emb, t_lengths, modality="txt", return_distribution=self.textvae 
        )
        
        t_motions = self.decode(t_latents[:bs], lengths)
        #motion -> motion
        m_latents, m_dists = self.encode(
            motions, lengths, modality="motion",return_distribution=self.motvae
        )
       
        m_motions = self.decode(m_latents, lengths)
        # Store all losses
        losses = {}

        # Reconstructions losses
        if self.cfg.lmd.recon == True:
            losses["recons"] = (
                + self.reconstruction_loss_fn(t_motions, ref_motions) # text -> motion
                + self.reconstruction_loss_fn(m_motions, ref_motions) # motion -> motion
            )
        else:
            losses["recons"] = 0.0
       
        # VAE losses
        if self.textvae:
            # Create a centred normal distribution to compare with
            # logvar = 0 -> std = 1
            ref_mus = torch.zeros_like(m_dists[0])
            ref_logvar = torch.zeros_like(m_dists[1])
            ref_dists = (ref_mus, ref_logvar)
            t_mu = t_dists[0][:bs]
            t_var = t_dists[1][:bs]
            t_dists_bs = (t_mu, t_var)
            losses["kl"] = (
                self.kl_loss_fn(t_dists_bs, m_dists)  # text_to_motion
                + self.kl_loss_fn(m_dists, t_dists_bs)  # motion_to_text
                + self.kl_loss_fn(m_dists, ref_dists)  # motion
                + self.kl_loss_fn(t_dists_bs, ref_dists)  # text
            )
        else:
            losses["kl"] = 0.0
        losses["contrast"] = self.contrastive_loss_fn(t_latents, m_latents, sent_emb)
        
        # Latent manifold loss
        losses["latent"] = self.latent_loss_fn(t_latents[:bs],m_latents)

        # Weighted average of the losses
        loss_all = sum(
            self.lmd[x] * val for x, val in losses.items() if x in self.lmd
        )
        if return_loss:
            return loss_all
        else:
            return t_latents, m_latents

