#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   fk_model.py
@Time    :   2026/01/18 16:00:00
@Author  :   Chuhao Jin 
@Contact :   jinchuhao@ruc.edu.cn

@Description:
    前向运动学模型 (Forward Kinematics)
    从四元数和根节点偏移计算世界坐标位置
'''

import numpy as np
import json, os 
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import RotationSpline
import torch
import torch.nn as nn
import utils.visualization_torch.BVH_mod as BVH
import torch.nn.functional as F


class WorldPosFromQuat(nn.Module):
    """
    forward:
        input_quat:  (B, F, J_src, 4)
        input_offset:(B, F, 3) or None
    return:
        world_pos:   (B, F, J_out, 3)  (默认 J_out=J_src，顺序与输入一致；未映射到模板的关节输出为0)
    """
    def __init__(
        self,
        template_bvh_path: str="./meta/template_susu_retarget_63nodes.bvh",
        input_order: str = "wxyz",     # "wxyz" or "xyzw"
        output_in_src_order: bool = True,
        scale: float = 1.0 / 100.0,    # 对齐你原来 new_position/100
        apply_pelvis_fix: bool = True,
        pelvis_name: str = "pelvis",
        apply_swizzle: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        
        src_joint_dict = {
            'pelvis': 0, 
            
            # 右腿
            'thigh_r': 1, 
            'calf_r': 2,
            'foot_r': 3, 
            'ball_r': 4, 
                
            # 左腿
            'thigh_l': 5, 
            'calf_l': 6,
            'foot_l': 7, 
            'ball_l': 8, 
            
            # 骨盆区域
            'spine_01': 9, 
            'spine_02': 10,
            'spine_03': 11, 
            'spine_04': 12, 
            'spine_05': 13, 
            'neck_01': 14, 
            'neck_02': 15, 
            'head': 16, 
            
            # 左臂
            'clavicle_l': 17,
            'upperarm_l': 18, 
            'lowerarm_l': 19, 
            'hand_l': 23, 
            
            # 右臂
            'clavicle_r': 20, 
            'upperarm_r': 21, 
            'lowerarm_r': 22, 
            'hand_r': 24,
            
            # 左手 [23, 25, 26, 27, 28]
            'index_metacarpal_l': 25, 
            'index_01_l': 26, 
            'index_02_l': 27, 
            'index_03_l': 28, 
            
            # [23, 29, 90, 31, 32]
            'middle_metacarpal_l': 29,
            'middle_01_l': 30, 
            'middle_02_l': 31, 
            'middle_03_l': 32, 
            
            # [23, 33, 34, 35, 36]
            'ring_metacarpal_l': 33,
            'ring_01_l': 34,
            'ring_02_l': 35, 
            'ring_03_l': 36, 
            
            # [23, 37, 38, 39, 40]
            'pinky_metacarpal_l': 37, 
            'pinky_01_l': 38, 
            'pinky_02_l': 39, 
            'pinky_03_l': 40, 
            
            # [23, 41, 42, 43]
            'thumb_01_l': 41, 
            'thumb_02_l': 42, 
            'thumb_03_l': 43, 
            
            # 右手 [24, 44, 45, 46, 47]
            'index_metacarpal_r': 44,
            'index_01_r': 45, 
            'index_02_r': 46, 
            'index_03_r': 47, 
            
            # [24, 48, 49, 50, 51]
            'middle_metacarpal_r': 48,
            'middle_01_r': 49, 
            'middle_02_r': 50, 
            'middle_03_r': 51, 
            
            # [24, 52, 53, 54, 55]
            'ring_metacarpal_r': 52,
            'ring_01_r': 53, 
            'ring_02_r': 54,
            'ring_03_r': 55, 
            
            # [24, 56, 57, 58, 59]
            'pinky_metacarpal_r': 56, 
            'pinky_01_r': 57, 
            'pinky_02_r': 58,
            'pinky_03_r': 59, 
            
            # [24, 60, 61, 62]
            'thumb_01_r': 60, 
            'thumb_02_r': 61, 
            'thumb_03_r': 62,
        }
        self.input_order = input_order
        self.output_in_src_order = output_in_src_order
        self.scale = float(scale)
        self.apply_pelvis_fix = apply_pelvis_fix
        self.pelvis_name = pelvis_name
        self.apply_swizzle = apply_swizzle
        self.eps = eps

        # 只在 init 里 load 模板一次（forward 不做IO）
        anim = BVH.load(template_bvh_path, need_quater=True)

        # 模板信息（作为 buffer，跟随 .to(device)）
        parents = anim.parents
        if not torch.is_tensor(parents):
            parents = torch.tensor(parents, dtype=torch.long)
        else:
            parents = parents.to(dtype=torch.long)

        base_local_pos = anim.positions[0]  # (J_full, 3)
        if not torch.is_tensor(base_local_pos):
            base_local_pos = torch.tensor(base_local_pos, dtype=torch.float32)

        self.register_buffer("parents", parents)                 # (J_full,)
        self.register_buffer("base_local_pos", base_local_pos)   # (J_full,3)

        self.template_names = list(anim.names)
        self.num_full_joints = int(base_local_pos.shape[0])

        # 建立 src_indices / dst_indices 映射（一次性算好）
        src_indices = []
        dst_indices = []
        for name, src_idx in src_joint_dict.items():
            if name in self.template_names:
                dst_idx = self.template_names.index(name)
                src_indices.append(int(src_idx))
                dst_indices.append(int(dst_idx))

        if len(src_indices) == 0:
            raise ValueError("src_joint_dict 与模板 anim.names 没有任何重叠关节，无法做映射。")

        self.register_buffer("src_indices", torch.tensor(src_indices, dtype=torch.long))
        self.register_buffer("dst_indices", torch.tensor(dst_indices, dtype=torch.long))

        # pelvis 在映射后的相对位置（可选）
        pelvis_rel = -1
        pelvis_src = src_joint_dict.get(pelvis_name, None)
        if pelvis_src is not None:
            pelvis_src = int(pelvis_src)
            # 找 pelvis_src 在 src_indices 里的位置
            matches = (self.src_indices == pelvis_src).nonzero(as_tuple=True)[0]
            if matches.numel() > 0:
                pelvis_rel = int(matches.item())
        self.pelvis_rel_idx = pelvis_rel  # python int

        # 常量：单位四元数 & pelvis 修正四元数（wxyz）
        self.register_buffer("q_identity", torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32))
        self.register_buffer("diff_inv", torch.tensor([0.7071, 0.0, 0.0, -0.7071], dtype=torch.float32))

    # --------- quaternion utils (wxyz) ----------
    def _q_normalize(self, q: torch.Tensor) -> torch.Tensor:
        
        return F.normalize(q, p=2, dim=-1, eps=self.eps)
        return q / (q.norm(dim=-1, keepdim=True) + self.eps)

    def _q_mul_wxyz(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        # q = q1 * q2, both (...,4) in (w,x,y,z)
        #这是一个数学的美丽巧合。虽然xyzw当成了wxyz。无敌hhh
        
        w1, x1, y1, z1 = q1.unbind(dim=-1)
        w2, x2, y2, z2 = q2.unbind(dim=-1)
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return torch.stack([w, x, y, z], dim=-1)

    def _quat_to_mat_wxyz(self, q: torch.Tensor) -> torch.Tensor:
        # q: (...,4) in (w,x,y,z) -> (...,3,3)
        # import pdb; pdb.set_trace()
        q = self._q_normalize(q)
        w, x, y, z = q.unbind(dim=-1)

        ww = w*w; xx = x*x; yy = y*y; zz = z*z
        wx = w*x; wy = w*y; wz = w*z
        xy = x*y; xz = x*z; yz = y*z

        m00 = 1 - 2*(yy + zz)
        m01 = 2*(xy - wz)
        m02 = 2*(xz + wy)

        m10 = 2*(xy + wz)
        m11 = 1 - 2*(xx + zz)
        m12 = 2*(yz - wx)

        m20 = 2*(xz - wy)
        m21 = 2*(yz + wx)
        m22 = 1 - 2*(xx + yy)

        return torch.stack([
            torch.stack([m00, m01, m02], dim=-1),
            torch.stack([m10, m11, m12], dim=-1),
            torch.stack([m20, m21, m22], dim=-1),
        ], dim=-2)

    # --------- FK ----------
    def _fk_global_positions(self, q_local_wxyz: torch.Tensor, t_local: torch.Tensor) -> torch.Tensor:
        """
        q_local_wxyz: (N, J, 4)
        t_local:      (N, J, 3)
        return:       (N, J, 3)
        """
        N, J, _ = t_local.shape
        R_local = self._quat_to_mat_wxyz(q_local_wxyz)  # (N,J,3,3)

        Rg = torch.empty((N, J, 3, 3), device=t_local.device, dtype=t_local.dtype)
        tg = torch.empty((N, J, 3), device=t_local.device, dtype=t_local.dtype)
        Rg_list = [None] * J
        tg_list = [None] * J
        # root
        Rg_list[0] = R_local[:, 0]
        tg_list[0] = t_local[:, 0]

        # chain
        parents = self.parents
        # for i in range(1, J):
        #     p = int(parents[i].item())
        #     Rp = Rg[:, p]
        #     Rg[:, i] = torch.matmul(Rp, R_local[:, i])
        #     tg[:, i] = tg[:, p] + torch.matmul(Rp, t_local[:, i].unsqueeze(-1)).squeeze(-1)
        for i in range(1, J):
            p = int(parents[i].item())
            
            # 从列表中读取父节点，这是引用，不会导致版本冲突
            Rp = Rg_list[p]
            tp = tg_list[p]
            
            # 计算当前节点，存入列表（这是创建新 Tensor，不是修改旧 Tensor）
            Rg_list[i] = torch.matmul(Rp, R_local[:, i])
            tg_list[i] = tp + torch.matmul(Rp, t_local[:, i].unsqueeze(-1)).squeeze(-1)

        # 最后将列表堆叠成 Tensor
        # Rg = torch.stack(Rg_list, dim=1) # 如果后续不需要 Rg 可以不 stack，省显存
        tg = torch.stack(tg_list, dim=1)
        return tg

    def forward(self, input_quat: torch.Tensor, input_offset: torch.Tensor | None):
        """
        input_quat:   (B,F,J_src,4)
        input_offset: (B,F,3) or None
        """
        B, F, J_src, _ = input_quat.shape
        device = input_quat.device
        dtype = input_quat.dtype

        # ---- A) 输入四元数 order 处理：对齐你原代码的逻辑 ----
        if self.input_order == "wxyz":
            # 你原来是 wxyz -> xyzw
            w, x, y, z = input_quat.unbind(dim=-1)
            q_work = torch.stack([x, y, z, w], dim=-1).to(device=device, dtype=dtype)  # (B,F,J,4) now "xyzw"
        elif self.input_order == "xyzw":
            q_work = input_quat
        else:
            raise ValueError(f"input_order must be 'wxyz' or 'xyzw', got {self.input_order}")
        # import pdb; pdb.set_trace()
        # ---- B) 取映射关节 ----
        q_sel = q_work[:, :, self.src_indices, :].clone()  # (B,F,M,4)

        # ---- C) pelvis 特殊修正（按你原代码的写法做）----
        if self.apply_pelvis_fix and self.pelvis_rel_idx >= 0:
            pelvis_q = q_sel[:, :, self.pelvis_rel_idx, :]  # (B,F,4)
            diff = self.diff_inv.to(device=device, dtype=dtype).view(1, 1, 4).expand(B, F, 4)
            pelvis_fixed = self._q_mul_wxyz(pelvis_q, diff)  # (B,F,4)
            q_sel[:, :, self.pelvis_rel_idx, :] = pelvis_fixed

        # ---- D) 全局 swizzle（按你原代码）----
        if self.apply_swizzle:
            x, y, z, w = q_sel.unbind(dim=-1)              # 因为此时是 "xyzw"
            q_mapped = torch.stack([w, -x, y, -z], dim=-1) # 输出 "wxyz"
        else:
            raise('error')

        # ---- E) 填到 Full_J（未提供的关节=单位四元数）----
        J_full = self.num_full_joints
        q_full = self.q_identity.to(device=device, dtype=dtype).view(1, 1, 1, 4).repeat(B, F, J_full, 1)
        q_full[:, :, self.dst_indices, :] = q_mapped

        # ---- F) root offset： (x,y,z)->(x,z,y) ----
        local_pos = self.base_local_pos.to(device=device, dtype=dtype).view(1, 1, J_full, 3).expand(B, F, J_full, 3).clone()
        if input_offset is not None:
            rx, ry, rz = input_offset.unbind(dim=-1)
            root_pos = torch.stack([rx, rz, ry], dim=-1)  # (B,F,3)
            local_pos[:, :, 0, :] = root_pos

        # ---- G) FK：展平 B*F 当作 N 帧 ----
        N = B * F
        qN = q_full.reshape(N, J_full, 4)
        tN = local_pos.reshape(N, J_full, 3)
        # import pdb; pdb.set_trace()
        worldN = self._fk_global_positions(qN, tN) * self.scale   # (N,J_full,3)
        world = worldN.reshape(B, F, J_full, 3)

        # ---- H) 输出顺序：默认回到输入 J_src 的关节顺序（便于直接算 loss）----
        if self.output_in_src_order:
            out = torch.zeros((B, F, J_src, 3), device=device, dtype=dtype)
            # template 的 dst_indices 对应输入的 src_indices
            out[:, :, self.src_indices, :] = world[:, :, self.dst_indices, :]
            return out

        # 否则返回模板顺序的全关节世界坐标
        return world


def resample_quaternions(data, original_fps, target_fps=20, interpolation_kind='slerp'):
    """
    data: 输入四元数序列，shape=(frames, joints, 4) 
          注意：Scipy 默认格式为 (x, y, z, w)，但也支持 (w, x, y, z) 只要保持一致即可。
    original_fps: 原始帧率
    target_fps: 目标帧率（默认20）
    interpolation_kind: 'slerp' (线性球面插值，推荐) 或 'spline' (连续性更好，但可能过冲)
    """
    frames, joints, _ = data.shape
    t_original = np.arange(frames) / original_fps  # 原始时间轴
    max_time = t_original[-1]
    target_frames = int(max_time * target_fps)
    t_target = np.linspace(0, max_time, target_frames)  # 目标时间轴

    # 初始化输出容器
    resampled_data = np.zeros((target_frames, joints, 4))

    # 对每个关节（Joint）独立处理
    for k in range(joints):
        # 提取该关节的所有帧四元数
        quats = data[:, k, :]
        
        # 1. 创建旋转对象 (会自动处理归一化)
        # 注意：如果数据可能存在断层导致 q 和 -q 跳变，Scipy 内部通常会自动处理
        rotations = R.from_quat(quats)
        
        if interpolation_kind == 'slerp':
            # 2a. 使用 SLERP (球面线性插值)
            # 相当于位置插值中的 'linear'，但在球面上是最短路径
            slerp = Slerp(t_original, rotations)
            interp_rotations = slerp(t_target)
            
        elif interpolation_kind == 'spline':
            # 2b. 使用 RotationSpline (球面样条插值)
            # 相当于位置插值中的 'cubic'，平滑度更高，但计算量稍大
            
            spline = RotationSpline(t_original, rotations)
            interp_rotations = spline(t_target)
            
        # 3. 转换回四元数数组并存入结果
        resampled_data[:, k, :] = interp_rotations.as_quat()

    return resampled_data