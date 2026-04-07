#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   rotation_utils.py
@Time    :   2026/01/18 16:00:00
@Author  :   Chuhao Jin 
@Contact :   jinchuhao@ruc.edu.cn

@Description:
    旋转表示转换工具
'''

import torch
import torch.nn.functional as F


def sixd_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    将 6D 旋转表示转换为 3x3 旋转矩阵。
    基于 Gram-Schmidt 正交化过程。
    
    Args:
        d6: (..., 6) 6D 表示
        
    Returns:
        rotation_matrix: (..., 3, 3) 旋转矩阵
    """
    # 分割出两个 3D 向量 a1, a2
    a1 = d6[..., :3]
    a2 = d6[..., 3:]

    # 1. 归一化第一个向量得到 x 轴
    b1 = F.normalize(a1, dim=-1, eps=1e-6)

    # 2. 对 a2 进行 Gram-Schmidt 正交化处理，得到 y 轴
    dot_prod = torch.sum(b1 * a2, dim=-1, keepdim=True)
    b2 = F.normalize(a2 - dot_prod * b1, dim=-1, eps=1e-6)

    # 3. 通过叉乘得到 z 轴
    b3 = torch.cross(b1, b2, dim=-1)

    # 4. 堆叠成矩阵 (按列堆叠)
    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_sixd(matrix: torch.Tensor) -> torch.Tensor:
    """
    将 3x3 旋转矩阵转换为 6D 表示。
    """
    # 取出前两列并展平
    return matrix[..., :2].transpose(-2, -1).reshape(*matrix.shape[:-2], 6)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    将四元数转换为旋转矩阵。
    假设四元数顺序为 [w, x, y, z]。
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    将旋转矩阵转换为四元数 [w, x, y, z]。
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    
    trace = m00 + m11 + m22

    def _safe_sqrt(x):
        return torch.sqrt(torch.clamp(x, min=1e-6))

    eps = 1e-6
    
    # Case 1: Trace > 0
    s1 = _safe_sqrt(trace + 1.0) * 0.5
    w = s1
    x = (matrix[..., 2, 1] - matrix[..., 1, 2]) / (4 * s1 + eps)
    y = (matrix[..., 0, 2] - matrix[..., 2, 0]) / (4 * s1 + eps)
    z = (matrix[..., 1, 0] - matrix[..., 0, 1]) / (4 * s1 + eps)
    q1 = torch.stack([w, x, y, z], dim=-1)

    # Case 2: m00 is largest
    s2 = _safe_sqrt(1.0 + m00 - m11 - m22) * 0.5
    w2 = (matrix[..., 2, 1] - matrix[..., 1, 2]) / (4 * s2 + eps)
    x2 = s2
    y2 = (matrix[..., 0, 1] + matrix[..., 1, 0]) / (4 * s2 + eps)
    z2 = (matrix[..., 0, 2] + matrix[..., 2, 0]) / (4 * s2 + eps)
    q2 = torch.stack([w2, x2, y2, z2], dim=-1)

    # Case 3: m11 is largest
    s3 = _safe_sqrt(1.0 + m11 - m00 - m22) * 0.5
    w3 = (matrix[..., 0, 2] - matrix[..., 2, 0]) / (4 * s3 + eps)
    x3 = (matrix[..., 0, 1] + matrix[..., 1, 0]) / (4 * s3 + eps)
    y3 = s3
    z3 = (matrix[..., 1, 2] + matrix[..., 2, 1]) / (4 * s3 + eps)
    q3 = torch.stack([w3, x3, y3, z3], dim=-1)

    # Case 4: m22 is largest
    s4 = _safe_sqrt(1.0 + m22 - m00 - m11) * 0.5
    w4 = (matrix[..., 1, 0] - matrix[..., 0, 1]) / (4 * s4 + eps)
    x4 = (matrix[..., 0, 2] + matrix[..., 2, 0]) / (4 * s4 + eps)
    y4 = (matrix[..., 1, 2] + matrix[..., 2, 1]) / (4 * s4 + eps)
    z4 = s4
    q4 = torch.stack([w4, x4, y4, z4], dim=-1)

    # 选择最佳路径
    q = torch.where(trace.unsqueeze(-1) > 0, q1, 
            torch.where((m00 > m11).unsqueeze(-1) & (m00 > m22).unsqueeze(-1), q2,
                torch.where((m11 > m22).unsqueeze(-1), q3, q4)))
    
    return F.normalize(q, dim=-1)


def quaternion_to_sixd(quaternions: torch.Tensor) -> torch.Tensor:
    """四元数 -> 旋转矩阵 -> 6D"""
    mat = quaternion_to_matrix(quaternions)
    return matrix_to_sixd(mat)


def sixd_to_quaternion(d6: torch.Tensor) -> torch.Tensor:
    """6D -> 旋转矩阵 -> 四元数"""
    mat = sixd_to_matrix(d6)
    return matrix_to_quaternion(mat)
