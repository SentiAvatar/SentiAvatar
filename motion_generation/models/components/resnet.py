#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   resnet.py
@Time    :   2026/01/18 16:00:00
@Author  :   Chuhao Jin 
@Contact :   jinchuhao@ruc.edu.cn

@Description:
    ResNet 1D 模块
'''

import torch
import torch.nn as nn


class Nonlinearity(nn.Module):
    """Swish/SiLU 激活函数"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResBlock(nn.Module):
    """
    1D 残差卷积块
    """
    
    def __init__(
        self,
        n_in: int,
        n_state: int,
        dilation: int = 1,
        activation: str = 'relu',
        norm: str = None,
        dropout: float = 0.2
    ):
        """
        初始化残差块
        
        Args:
            n_in: 输入通道数
            n_state: 中间状态通道数
            dilation: 扩张率
            activation: 激活函数类型
            norm: 归一化类型
            dropout: dropout 概率
        """
        super().__init__()
        
        padding = dilation
        self.norm_type = norm
        
        # 归一化层
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        
        # 激活函数
        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
        elif activation == "silu":
            self.activation1 = Nonlinearity()
            self.activation2 = Nonlinearity()
        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()
        else:
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
        
        # 卷积层
        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """前向传播"""
        x_orig = x
        
        # 第一个归一化和激活
        if self.norm_type == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)
        
        x = self.conv1(x)
        
        # 第二个归一化和激活
        if self.norm_type == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)
        
        x = self.conv2(x)
        x = self.dropout(x)
        
        # 残差连接
        x = x + x_orig
        return x


class Resnet1D(nn.Module):
    """
    1D ResNet 模块
    
    由多个残差块组成
    """
    
    def __init__(
        self,
        n_in: int,
        n_depth: int,
        dilation_growth_rate: int = 1,
        reverse_dilation: bool = True,
        activation: str = 'relu',
        norm: str = None
    ):
        """
        初始化 1D ResNet
        
        Args:
            n_in: 输入通道数
            n_depth: 残差块数量
            dilation_growth_rate: 扩张增长率
            reverse_dilation: 是否反转扩张顺序
            activation: 激活函数类型
            norm: 归一化类型
        """
        super().__init__()
        
        blocks = [
            ResBlock(
                n_in,
                n_in,
                dilation=dilation_growth_rate ** depth,
                activation=activation,
                norm=norm
            )
            for depth in range(n_depth)
        ]
        
        if reverse_dilation:
            blocks = blocks[::-1]
        
        self.model = nn.Sequential(*blocks)
    
    def forward(self, x):
        """前向传播"""
        return self.model(x)
