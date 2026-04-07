#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   quantizer.py
@Time    :   2026/01/18 16:00:00
@Author  :   Chuhao Jin 
@Contact :   jinchuhao@ruc.edu.cn

@Description:
    向量量化器模块
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def log(t, eps=1e-20):
    """安全的对数函数"""
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    """生成 Gumbel 噪声"""
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(logits, temperature=1., stochastic=False, dim=-1, training=True):
    """
    Gumbel 采样
    
    Args:
        logits: 输入 logits
        temperature: 温度参数
        stochastic: 是否使用随机采样
        dim: 采样维度
        training: 是否训练模式
    
    Returns:
        采样索引
    """
    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits
    
    ind = sampling_logits.argmax(dim=dim)
    return ind


class Quantizer(nn.Module):
    """
    EMA 向量量化器（带重置）
    
    使用指数移动平均更新码本
    """
    
    def __init__(
        self,
        nb_code: int,
        code_dim: int,
        mu: float = 0.99
    ):
        """
        初始化量化器
        
        Args:
            nb_code: 码本大小
            code_dim: 码本维度
            mu: EMA 系数
        """
        super().__init__()
        
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        
        self.reset_codebook()
    
    def reset_codebook(self):
        """重置码本"""
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim, requires_grad=False))
    
    def _tile(self, x):
        """扩展输入以匹配码本大小"""
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out
    
    def init_codebook(self, x):
        """初始化码本"""
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True
    
    def quantize(self, x, sample_codebook_temp=0.):
        """
        量化
        
        Args:
            x: 输入张量 (N*T, C)
            sample_codebook_temp: 采样温度
        
        Returns:
            码本索引
        """
        # N X C -> C X N
        k_w = self.codebook.t()
        
        # 计算距离
        distance = (
            torch.sum(x ** 2, dim=-1, keepdim=True) -
            2 * torch.matmul(x, k_w) +
            torch.sum(k_w ** 2, dim=0, keepdim=True)
        )
        
        code_idx = gumbel_sample(
            -distance,
            dim=-1,
            temperature=sample_codebook_temp,
            stochastic=True,
            training=self.training
        )
        
        return code_idx
    
    def dequantize(self, code_idx):
        """反量化"""
        x = F.embedding(code_idx, self.codebook)
        return x
    
    def get_codebook_entry(self, indices):
        """获取码本条目"""
        return self.dequantize(indices).permute(0, 2, 1)
    
    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        """计算困惑度"""
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)
        
        code_count = code_onehot.sum(dim=-1)
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
    
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        """更新码本（EMA + 重置）"""
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)
        
        code_sum = torch.matmul(code_onehot, x)
        code_count = code_onehot.sum(dim=-1)
        
        out = self._tile(x)
        code_rand = out[:self.nb_code]
        
        # EMA 更新
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count
        
        # 重置未使用的码本
        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)
        self.codebook = usage * code_update + (1 - usage) * code_rand
        
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        
        return perplexity
    
    def preprocess(self, x):
        """预处理: (N, C, T) -> (N*T, C)"""
        x = rearrange(x, 'n c t -> (n t) c')
        return x
    
    def forward(self, x, return_idx=False, temperature=0.):
        """
        前向传播
        
        Args:
            x: 输入张量 (N, C, T)
            return_idx: 是否返回索引
            temperature: 采样温度
        
        Returns:
            x_d: 量化后的张量
            code_idx: 码本索引（可选）
            commit_loss: commitment loss
            perplexity: 困惑度
        """
        N, width, T = x.shape
        
        x = self.preprocess(x)
        
        if self.training and not self.init:
            self.init_codebook(x)
        
        code_idx = self.quantize(x, temperature)
        x_d = self.dequantize(code_idx)
        
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)
        
        commit_loss = F.mse_loss(x, x_d.detach())
        
        # Straight-through estimator
        x_d = x + (x_d - x).detach()
        
        # 后处理
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        code_idx = code_idx.view(N, T).contiguous()
        
        if return_idx:
            return x_d, code_idx, commit_loss, perplexity
        return x_d, commit_loss, perplexity


class QuantizerEMA(Quantizer):
    """
    EMA 向量量化器（不带重置）
    """
    
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        """更新码本（仅 EMA，不重置）"""
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)
        
        code_sum = torch.matmul(code_onehot, x)
        code_count = code_onehot.sum(dim=-1)
        
        # EMA 更新
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count
        
        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)
        self.codebook = usage * code_update + (1 - usage) * self.codebook
        
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        
        return perplexity
