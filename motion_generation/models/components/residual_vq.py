#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   residual_vq.py
@Time    :   2026/01/18 16:00:00
@Author  :   Chuhao Jin 
@Contact :   jinchuhao@ruc.edu.cn

@Description:
    残差向量量化模块
    参考: https://arxiv.org/pdf/2107.03312.pdf
'''

import random
from math import ceil
from random import randrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from .quantizer import Quantizer


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class ResidualVQ(nn.Module):
    """
    残差向量量化器 (RVQ)
    
    实现 Algorithm 1 from https://arxiv.org/pdf/2107.03312.pdf
    """
    
    def __init__(
        self,
        num_quantizers: int,
        shared_codebook: bool = False,
        quantize_dropout_prob: float = 0.5,
        quantize_dropout_cutoff_index: int = 0,
        nb_code: int = 1024,
        code_dim: int = 512,
        mu: float = 0.99,
        **kwargs
    ):
        """
        初始化 RVQ
        
        Args:
            num_quantizers: 量化器数量
            shared_codebook: 是否共享码本
            quantize_dropout_prob: 量化 dropout 概率
            quantize_dropout_cutoff_index: dropout 截止索引
            nb_code: 码本大小
            code_dim: 码本维度
            mu: EMA 系数
        """
        super().__init__()
        
        self.num_quantizers = num_quantizers
        
        # 创建量化器层
        if shared_codebook:
            layer = Quantizer(nb_code=nb_code, code_dim=code_dim, mu=mu)
            self.layers = nn.ModuleList([layer for _ in range(num_quantizers)])
        else:
            self.layers = nn.ModuleList([
                Quantizer(nb_code=nb_code, code_dim=code_dim, mu=mu)
                for _ in range(num_quantizers)
            ])
        
        assert quantize_dropout_cutoff_index >= 0 and quantize_dropout_prob >= 0
        
        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_prob = quantize_dropout_prob
    
    @property
    def codebooks(self):
        """获取所有码本"""
        codebooks = [layer.codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim=0)
        return codebooks  # (q, c, d)
    
    def get_codes_from_indices(self, indices):
        """
        从索引获取码本向量
        
        Args:
            indices: 索引张量 (b, n, q)
        
        Returns:
            所有码本向量 (q, b, n, d)
        """
        batch, quantize_dim = indices.shape[0], indices.shape[-1]
        
        # 处理 dropout 导致的维度不足
        if quantize_dim < self.num_quantizers:
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value=-1)
        
        # 准备 gather
        codebooks = repeat(self.codebooks, 'q c d -> q b c d', b=batch)
        gather_indices = repeat(indices, 'b n q -> q b n d', d=codebooks.shape[-1])
        
        # 处理 dropout 的 mask
        mask = gather_indices == -1.
        gather_indices = gather_indices.masked_fill(mask, 0)
        
        # gather 所有码本向量
        all_codes = codebooks.gather(2, gather_indices)
        
        # mask 掉 dropout 的码本
        all_codes = all_codes.masked_fill(mask, 0.)
        
        return all_codes  # (q, b, n, d)
    
    def get_codebook_entry(self, indices):
        """
        获取码本条目（求和）
        
        Args:
            indices: 索引张量 (b, n, q)
        
        Returns:
            潜在向量 (b, d, n)
        """
        all_codes = self.get_codes_from_indices(indices)  # (q, b, n, d)
        latent = torch.sum(all_codes, dim=0)  # (b, n, d)
        latent = latent.permute(0, 2, 1)  # (b, d, n)
        return latent
    
    def forward(
        self,
        x,
        return_all_codes: bool = False,
        sample_codebook_temp: float = None,
        force_dropout_index: int = -1
    ):
        """
        前向传播
        
        Args:
            x: 输入张量 (b, c, t)
            return_all_codes: 是否返回所有码本向量
            sample_codebook_temp: 采样温度
            force_dropout_index: 强制 dropout 索引
        
        Returns:
            quantized_out: 量化输出
            all_indices: 所有索引
            all_losses: 平均损失
            all_perplexity: 平均困惑度
        """
        num_quant = self.num_quantizers
        quant_dropout_prob = self.quantize_dropout_prob
        device = x.device
        
        quantized_out = 0.
        residual = x
        
        all_losses = []
        all_indices = []
        all_perplexity = []
        
        # 判断是否进行 dropout
        should_quantize_dropout = self.training and random.random() < quant_dropout_prob
        
        start_drop_quantize_index = num_quant
        if num_quant == 1:
            should_quantize_dropout = False
        
        if should_quantize_dropout:
            start_drop_quantize_index = randrange(self.quantize_dropout_cutoff_index, num_quant)
            null_indices_shape = [x.shape[0], x.shape[-1]]
            null_indices = torch.full(null_indices_shape, -1., device=device, dtype=torch.long)
        
        if force_dropout_index >= 0:
            should_quantize_dropout = True
            start_drop_quantize_index = force_dropout_index
            null_indices_shape = [x.shape[0], x.shape[-1]]
            null_indices = torch.full(null_indices_shape, -1., device=device, dtype=torch.long)
        
        # 遍历所有量化器
        for quantizer_index, layer in enumerate(self.layers):
            if should_quantize_dropout and quantizer_index > start_drop_quantize_index:
                all_indices.append(null_indices)
                continue
            
            quantized, *rest = layer(residual, return_idx=True, temperature=sample_codebook_temp)
            
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized
            
            embed_indices, loss, perplexity = rest
            all_indices.append(embed_indices)
            all_losses.append(loss)
            all_perplexity.append(perplexity)
        
        # 汇总结果
        all_indices = torch.stack(all_indices, dim=-1)
        all_losses = sum(all_losses) / len(all_losses)
        all_perplexity = sum(all_perplexity) / len(all_perplexity)
        
        ret = (quantized_out, all_indices, all_losses, all_perplexity)
        
        if return_all_codes:
            all_codes = self.get_codes_from_indices(all_indices)
            ret = (*ret, all_codes)
        
        return ret
    
    def quantize(self, x, return_latent: bool = False):
        """
        量化（推理用）
        
        Args:
            x: 输入张量
            return_latent: 是否返回潜在向量
        
        Returns:
            code_idx: 码本索引
            all_codes: 所有码本向量（可选）
        """
        all_indices = []
        quantized_out = 0.
        residual = x
        all_codes = []
        
        for quantizer_index, layer in enumerate(self.layers):
            quantized, *rest = layer(residual, return_idx=True)
            
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized
            
            embed_indices, loss, perplexity = rest
            all_indices.append(embed_indices)
            all_codes.append(quantized)
        
        code_idx = torch.stack(all_indices, dim=-1)
        all_codes = torch.stack(all_codes, dim=0)
        
        if return_latent:
            return code_idx, all_codes
        return code_idx
