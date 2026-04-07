#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   rvqvae.py
@Time    :   2026/01/18 16:00:00
@Author  :   Chuhao Jin 
@Contact :   jinchuhao@ruc.edu.cn

@Description:
    RVQ-VAE 模型，用于动作数据的编码和解码
'''

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .components.encoder import Encoder
from .components.decoder import Decoder
from .components.residual_vq import ResidualVQ


class RVQVAE(nn.Module):
    """
    残差矢量量化变分自编码器 (RVQ-VAE)
    
    支持多部位编码：whole, body, left, right
    """
    
    def __init__(
        self,
        config,
        input_dim: int = 393,
        nb_code: int = 1024,
        code_dim: int = 512,
        output_dim: int = 512,
        down_t: int = 1,
        stride_t: int = 2,
        width: int = 512,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        activation: str = 'relu',
        norm: Optional[str] = None,
    ):
        """
        初始化 RVQ-VAE 模型
        
        Args:
            config: 配置对象
            input_dim: 输入特征维度 (whole = 393)
            nb_code: 码本大小
            code_dim: 码本维度
            output_dim: 输出维度
            down_t: 时间维度下采样率
            stride_t: 时间维度步长
            width: 网络宽度
            depth: 网络深度
            dilation_growth_rate: 扩张增长率
            activation: 激活函数类型
            norm: 归一化方式
        """
        super().__init__()
        
        self.code_dim = code_dim
        self.num_code = nb_code
        self.unit_length = down_t * 2
        
        # 获取 CNN 深度
        vq_cnn_depth = getattr(config.model, 'vq_cnn_depth', 3)
        
        # 特征维度配置
        self.body_dim = config.data.body_dim  # 153
        # ==================== 编码器 ====================
        
        # Body 编码器
        self.body_encoder = Encoder(
            input_dim=self.body_dim,
            output_dim=output_dim,
            down_t=down_t,
            stride_t=stride_t,
            width=width,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            norm=norm,
            vq_cnn_depth=vq_cnn_depth
        )
        
        decoder_out_dim = output_dim * 1
        self.decoder = Decoder(
            input_dim=self.body_dim,
            output_dim=decoder_out_dim,  # 3/4 个编码器的输出拼接
            down_t=down_t,
            stride_t=stride_t,
            width=width * 2,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            norm=norm,
            vq_cnn_depth=vq_cnn_depth
        )
        
        # 编码器字典
        self.encoder_dict = {
            "body": self.body_encoder,
        }
        
        # ==================== 量化器 ====================
        rvqvae_config = {
            'num_quantizers': config.model.num_quantizers,
            'shared_codebook': config.model.shared_codebook,
            'quantize_dropout_prob': config.model.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': config.model.quantize_dropout_cutoff_index,
            'nb_code': nb_code,
            'code_dim': code_dim,
        }
        
        self.quantizer_body = ResidualVQ(**rvqvae_config)
        
        self.quantizer_dict = {
            "body": self.quantizer_body,
        }
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """预处理: (bs, T, D) -> (bs, D, T)"""
        return x.permute(0, 2, 1).float()
    
    def postprocess(self, x: torch.Tensor) -> torch.Tensor:
        """后处理: (bs, D, T) -> (bs, T, D)"""
        return x.permute(0, 2, 1)
    
    def forward(
        self,
        x_body: torch.Tensor,
        return_idx: bool = False
    ) -> Tuple:
        """
        前向传播
        
        Args:
            x_body: body 数据 (bs, T, body_dim)
            x_left: left 数据 (bs, T, left_dim)
            x_right: right 数据 (bs, T, right_dim)
            return_idx: 是否返回码本索引
        
        Returns:
            x_out: 重建输出
            commit_loss: commitment loss
            perplexity: 困惑度
            idx_code_idx: 码本索引（可选）
        """
        
        # Body 编码和量化
        x_in = self.preprocess(x_body)
        x_encoder = self.encoder_dict["body"](x_in)
        x_body_quantized, code_idx_body, commit_loss, perplexity = \
            self.quantizer_dict["body"](x_encoder, sample_codebook_temp=0.5)
            
        x_quantized = x_body_quantized
        # print("code_idx_body:", code_idx_body.shape)
        # print("x_quantized:", x_quantized.shape)
        x_out = self.decoder(x_quantized)
        # print("x_out:", x_out.shape)
        if return_idx:
            idx_code_idx = {
                "body": code_idx_body
            }
            return x_out, commit_loss, perplexity, idx_code_idx
        
        return x_out, commit_loss, perplexity
    
    def encode(
        self,
        x_body: torch.Tensor,
    ) -> Dict:
        """
        编码
        
        Args:
            x_body: body 数据
            x_left: left 数据
            x_right: right 数据
        
        Returns:
            包含 code_idx 和 codes 的字典
        """
        
        # Body
        x_in = self.preprocess(x_body)
        x_encoder = self.encoder_dict["body"](x_in)
        body_code_idx, body_all_codes = self.quantizer_dict["body"].quantize(x_encoder, return_latent=True)
        body_all_codes = body_all_codes.squeeze(0)
        
        
        outputs = {
            "code_idx": {
                "body": body_code_idx,
            },
            "codes": {
                "body": body_all_codes,
            }
        }        
        return outputs
    
    def decode(self, code_idx: Dict) -> torch.Tensor:
        """
        解码
        
        Args:
            code_idx: 包含各部位码本索引的字典
        
        Returns:
            解码输出
        """
        body_code_idx = code_idx["body"]
    
        body_all_codes = self.quantizer_dict["body"].get_codes_from_indices(body_code_idx)
        body_all_codes = body_all_codes.squeeze(1)
        body_all_codes = body_all_codes.sum(0)
        body_all_codes = body_all_codes.unsqueeze(0)
        # print("body_all_codes:", body_all_codes.shape) 
        x_quantized = body_all_codes
        x_quantized = x_quantized.permute(0, 2, 1)
        x_out = self.decoder(x_quantized)
        
        return x_out
    
    def forward_decoder(self, code_idx: Dict) -> torch.Tensor:
        """解码（别名）"""
        return self.decode(code_idx)
