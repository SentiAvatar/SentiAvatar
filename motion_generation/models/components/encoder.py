#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   encoder.py
@Time    :   2026/01/18 16:00:00
@Author  :   Chuhao Jin 
@Contact :   jinchuhao@ruc.edu.cn

@Description:
    编码器模块
'''

import torch.nn as nn
from .resnet import Resnet1D


class Encoder(nn.Module):
    """
    1D 卷积编码器
    
    将输入特征编码为潜在表示
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 512,
        down_t: int = 2,
        stride_t: int = 2,
        width: int = 512,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        activation: str = 'relu',
        norm: str = None,
        vq_cnn_depth: int = 2
    ):
        """
        初始化编码器
        
        Args:
            input_dim: 输入特征维度
            output_dim: 输出特征维度
            down_t: 下采样次数
            stride_t: 步长
            width: 网络宽度
            depth: ResNet 深度
            dilation_growth_rate: 扩张增长率
            activation: 激活函数
            norm: 归一化方式
            vq_cnn_depth: VQ-CNN 深度
        """
        super().__init__()
        
        self.vq_cnn_depth = vq_cnn_depth
        
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        
        # 输入卷积
        blocks.append(nn.Conv1d(input_dim, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        # 额外的 CNN 层
        cnn_depth = vq_cnn_depth - down_t
        assert cnn_depth >= 0, f"vq_cnn_depth ({vq_cnn_depth}) must be >= down_t ({down_t})"
        
        for _ in range(cnn_depth):
            block = nn.Sequential(
                nn.Conv1d(width, width, 3, 1, 1),
                Resnet1D(
                    width,
                    depth,
                    dilation_growth_rate,
                    activation=activation,
                    norm=norm
                ),
            )
            blocks.append(block)
        
        # 下采样层
        for i in range(down_t):
            block = nn.Sequential(
                nn.Conv1d(width, width, filter_t, stride_t, pad_t),
                Resnet1D(
                    width,
                    depth,
                    dilation_growth_rate,
                    activation=activation,
                    norm=norm
                ),
            )
            blocks.append(block)
        
        # 输出卷积
        blocks.append(nn.Conv1d(width, output_dim, 3, 1, 1))
        
        self.model = nn.Sequential(*blocks)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch, channels, time)
        
        Returns:
            编码后的张量
        """
        return self.model(x)
