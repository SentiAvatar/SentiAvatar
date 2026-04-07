#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   postprocess.py
@Time    :   2025/12/19 11:24:47
@Author  :   Chuhao Jin 
@Contact :   jinchuhao@ruc.edu.cn
'''
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

from typing import Dict, Any, Optional
from utils.skeleton import Skeleton
from utils.visualization_torch.joints2bvh_mta import Joint2BVHConvertor
import utils.visualization_torch.BVH_mod as BVH
from params.mta63joints_params import SkeletonSpec
from params.mta63joints_constants import (
    template_bvh,
    mta_joint_sorted_file,
    skeleton_file,
    kinematic_chain_file,
    t2m_raw_offsets,
    src_joint_dict_file,
    joint_nodes_file,
    static_face_file,
    demo_quats_file
)

def process_batch_data(batch_quat, batch_offset, template_anim, src_joint_dict,shape='wxyz'):
    """
    将 (B, F, J, 4) 的原始数据转换为符合 BVH 结构的 (B, F, Full_J, 4) 数据
    全过程在 Tensor 上并行完成，不涉及 IO。
    """
    if shape=='xyzw':
        # batch_quat = quaternion_to_matrix(batch_quat)
        pass
    elif shape=='wxyz':
        w,x,y,z = batch_quat.unbind(dim=-1)
        batch_quat = torch.stack([x,y,z,w], dim=-1)
    B, F, J, _ = batch_quat.shape
    device = batch_quat.device
    num_bvh_joints = template_anim.rotations.qs.shape[1] # 模板的全量骨骼数
    
    # --- A. 建立索引映射 ---
    src_indices = []
    dst_indices = []
    for name, src_idx in src_joint_dict.items():
        if name in template_anim.names:
            try:
                dst_idx = list(template_anim.names).index(name)
                src_indices.append(src_idx)
                dst_indices.append(dst_idx)
            except ValueError:
                pass
    
    src_indices = torch.tensor(src_indices, dtype=torch.long, device=device)
    dst_indices = torch.tensor(dst_indices, dtype=torch.long, device=device)
    # import pdb; pdb.set_trace()
    # --- B. 提取有效骨骼 ---
    # shape: (B, F, Mapped_J, 4)
    selected_quats = batch_quat[:, :, src_indices, :].clone()
    
    # --- C. Pelvis 特殊旋转修正 ---
    # 找到 pelvis 在 selected_quats 里的相对索引
    pelvis_src_idx = src_joint_dict.get('pelvis')
    if pelvis_src_idx is not None and pelvis_src_idx in src_indices:
        # 在 src_indices 中找到 pelvis 的位置
        pelvis_rel_idx = (src_indices == pelvis_src_idx).nonzero(as_tuple=True)[0].item()
        
        # 构造逆变换四元数
        # diff: [0.7071, 0, 0, 0.7071] (w,x,y,z) -> inv: [0.7071, -0, -0, -0.7071]
        diff_inv = torch.tensor([0.7071, 0.0, 0.0, -0.7071], device=device)
        
        # 广播到 (B, F, 4)
        diff_inv_expanded = diff_inv.view(1, 1, 4).expand(B, F, -1)
        
        # 提取 pelvis 数据: (B, F, 4)
        pelvis_q = selected_quats[:, :, pelvis_rel_idx, :]
        
        # 四元数乘法: pelvis * diff_inv
        # 手写 q_mul 以支持 broadcasting
        w1, x1, y1, z1 = pelvis_q.unbind(-1)
        w2, x2, y2, z2 = diff_inv_expanded.unbind(-1)
        
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        # 写回
        selected_quats[:, :, pelvis_rel_idx, :] = torch.stack((w, x, y, z), dim=-1)

    # --- D. 全局 Swizzle (坐标轴重排) ---
    # Input (w, x, y, z) -> Target (z, -w, x, -y)
    x, y, z ,w= selected_quats.unbind(dim=-1)
    # swizzled_quats = torch.stack((z, -w, x, -y), dim=-1)
    swizzled_quats = torch.stack((w, -x, y, -z), dim=-1)
    
    # --- E. 填充到完整的 BVH 骨骼结构中 ---
    # 初始化全量 Tensor (B, F, Full_J, 4) 为 Identity [1, 0, 0, 0]
    final_quats = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).view(1, 1, 1, 4).repeat(B, F, num_bvh_joints, 1)
    
    # 批量赋值：利用广播机制，一次性把所有帧、所有batch的对应骨骼填进去
    # final_quats[:, :, dst_indices, :] shape matches swizzled_quats
    final_quats[:, :, dst_indices, :] = swizzled_quats
    
    # --- F. Root 位置处理 ---
    # batch_offset: (B, F, 3) -> [x, y, z]
    # Target: [x, z, y]
    final_positions = None
    if batch_offset is not None:
        rx = batch_offset[:, :, 0]
        ry = batch_offset[:, :, 1]
        rz = batch_offset[:, :, 2]
        # 交换 y 和 z
        final_positions = torch.stack((rx, rz, ry), dim=-1) # (B, F, 3)
    # import pdb; pdb.set_trace()    
    return final_quats, final_positions


def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)

class MotionPostprocesser:
    """
    最终执行层：对接 UE/引擎，将 MotionSequence 转成实际骨骼帧或协议
    """
    def __init__(self):
        skel = SkeletonSpec.from_meta_files(
            template_bvh=template_bvh,
            mta_joint_sorted_file=mta_joint_sorted_file,
            skeleton_file=skeleton_file,
            kinematic_chain_file=kinematic_chain_file,
            t2m_raw_offsets=t2m_raw_offsets,
            src_joint_dict_file=src_joint_dict_file,
            joint_nodes_file=joint_nodes_file,
            static_face_file=static_face_file,
            demo_quats_file=demo_quats_file
        )
        self.skel = skel
        self.quat_converter = self.load_convertor()
    
    def load_convertor(self):
        joint_id_dict = {joint: idx for idx, joint in enumerate(self.skel.joint_list)}
        end_points_id = [joint_id_dict[joint] for joint in self.skel.end_points]
        joint_num = self.skel.joints_num

        re_order = [i for i in range(joint_num)]
        re_order_inv = [i for i in range(joint_num)]

        parents = [joint_id_dict.get(self.skel.skeleton_json[joint], -1) - 1 for joint in self.skel.joint_list]
        fid_r = [joint_id_dict["foot_r"], joint_id_dict["ball_r"]]
        fid_l = [joint_id_dict["foot_l"], joint_id_dict["ball_l"]]
        demo_quats = np.load(self.skel.demo_quats_file)
        self.anim = BVH.load(self.skel.template_bvh, need_quater=True)
        converter = Joint2BVHConvertor(
            template_bvh, re_order, re_order_inv, end_points_id, parents, fid_r, fid_l, self.skel.skeleton_json, self.skel.skel_dict, self.skel.static_face, demo_quats
        )
        return converter
    
    def convert_quat_motion_to_ue_from_bvh(
        self, 
        motion: Dict[str, np.ndarray],
        shape='wxyz'
        ) -> None:
        
        input_quat = motion["quat"]
        input_offset = motion["offset"]
        quat_anim = {
            "timestamp": 1754016912754,
            "fps": 30,
            "frames": []
        }
        
        input_quat = motion["quat"]
        input_offset = motion["offset"]
        if not isinstance(input_quat, torch.Tensor):
            input_quat = torch.as_tensor(input_quat)  # 保留 numpy 时更高效
            if len(input_quat.shape) == 3:
                input_quat = input_quat[None]
        # offset: 同理
        if not isinstance(input_offset, torch.Tensor):
            input_offset = torch.as_tensor(input_offset)
            if len(input_offset.shape) == 2:
                input_offset = input_offset[None]
        final_quats, final_root_pos = process_batch_data(input_quat, input_offset, self.anim, self.skel.src_joint_dict, shape=shape)
        
        # 转为 CPU 准备写入
        final_quats = final_quats.cpu()
        if final_root_pos is not None:
            final_root_pos = final_root_pos.cpu()
        # 准备模板的基础位置 (F, Full_J, 3)
        num_frames = final_quats.shape[1]
        base_pos = self.anim.positions[0].clone().unsqueeze(0).repeat(num_frames, 1, 1) # (F, J, 3)

        current_quats = final_quats[0]
        
        # pos: (F, Full_J, 3)
        current_pos = base_pos.clone()
        if final_root_pos is not None:
            # 更新 Root (Index 0) 位置
            current_pos[:, 0, :] = final_root_pos[0]
            
        # 赋值给 Anim 对象
        self.anim.rotations.qs = current_quats
        self.anim.positions = current_pos
        
        quats_t = self.anim.rotations.qs.detach().cpu()  # float32 by default

        # 假设原始顺序是 [w, x, y, z]
        w = quats_t[..., 0]
        x = quats_t[..., 1]
        y = quats_t[..., 2]
        z = quats_t[..., 3]

        # 对所有关节统一做：q = [-q[1], q[2], -q[3], q[0]]
        quats_reordered = torch.stack([-x, y, -z, w], dim=-1)  # (F, J, 4)

        # 索引准备（建议你在 __init__ 里就算好，这里简单写）
        names = self.anim.names
        pelvis_idx = names.index("pelvis")
        head_idx   = names.index("head")

        # head 直接设为单位四元数 [0, 0, 0, 1]
        # quats_reordered[:, head_idx] = torch.tensor([0, 0, 0, 1], dtype=quats_reordered.dtype)

        # pelvis 需要再和 diff 相乘：q = qmul(q, diff)
        diff = torch.tensor([0.7071, 0.0, 0.0, 0.7071], dtype=quats_reordered.dtype)
        pelvis_q = quats_reordered[:, pelvis_idx]                  # (F, 4)
        diff_batch = diff.view(1, 4).expand(pelvis_q.size(0), 4)   # (F, 4)
        quats_reordered[:, pelvis_idx] = qmul(pelvis_q, diff_batch)

        # 一次性 round + 转 numpy，后面循环只做取值
        quats_np = quats_reordered.numpy().astype(np.float32)  # (F, J, 4)

        # ---------------------------
        # 2. 位置同样一次性取出来
        # ---------------------------
        poss_root = self.anim.positions[:, 0].detach().cpu().numpy()  # (F, 3)

        # ---------------------------
        # 3. 构造 quat_anim：只做轻量级 Python 循环
        # ---------------------------

        quat_anim = {
            "timestamp": 1754016912754,
            "fps": 30,
            "frames": []
        }

        body_len = len(self.skel.skeleton_json)
        # 映射动画关节到 skeleton 索引，避免在内层循环反复查
        joint_indices = [self.skel.skel_dict[joint_name] for joint_name in names]
        # print("joint_indices:", joint_indices)
        for frame_idx in range(quats_np.shape[0]):
            pos = poss_root[frame_idx]  # (3,)
            # 注意：不要用 [[...]] * body_len，会共用同一子列表
            body = [[0, 0, 0, 1] for _ in range(body_len)]

            frame = {
                "offset": [round(pos[0], 4), round(pos[2], 4), round(pos[1], 4)],
                "body": body,
                "face": self.skel.static_face
            }

            q_frame = quats_np[frame_idx]  # (J, 4)

            # 这里不再做任何数值运算，只是赋值
            for j, joint_name in enumerate(names):
                joint_index = joint_indices[j]
                q = q_frame[j].tolist()
                q = [round(v, 4) for v in q]
                frame["body"][joint_index] = q
            # ball_l / ball_r 固定写死
            # frame["body"][self.skel.skel_dict["ball_l"]] = [0, 0, 0.8509, 0.5253]
            # frame["body"][self.skel.skel_dict["ball_r"]] = [0, 0, 0.8509, 0.5253]

            quat_anim["frames"].append(frame)
            
        
        return quat_anim
    
    
    def convert_quat_motion_to_ue(
        self, 
        motion: Dict[str, np.ndarray],
        shape='wxyz'
        ) -> None:
        
        input_quat = motion["quat"]
        input_offset = motion["offset"]
        quat_anim = {
            "timestamp": 1754016912754,
            "fps": 30,
            "frames": []
        }
        
        input_quat = motion["quat"]
        input_offset = motion["offset"]
        if not isinstance(input_quat, torch.Tensor):
            input_quat = torch.as_tensor(input_quat)  # 保留 numpy 时更高效
            if len(input_quat.shape) == 3:
                input_quat = input_quat[None]
        # offset: 同理
        if not isinstance(input_offset, torch.Tensor):
            input_offset = torch.as_tensor(input_offset)
            if len(input_offset.shape) == 2:
                input_offset = input_offset[None]
        final_quats, final_root_pos = process_batch_data(input_quat, input_offset, self.anim, self.skel.src_joint_dict, shape=shape)
        
        # 转为 CPU 准备写入
        final_quats = final_quats.cpu()
        if final_root_pos is not None:
            final_root_pos = final_root_pos.cpu()
        # 准备模板的基础位置 (F, Full_J, 3)
        num_frames = final_quats.shape[1]
        base_pos = self.anim.positions[0].clone().unsqueeze(0).repeat(num_frames, 1, 1) # (F, J, 3)

        current_quats = final_quats[0]
        
        # pos: (F, Full_J, 3)
        current_pos = base_pos.clone()
        if final_root_pos is not None:
            # 更新 Root (Index 0) 位置
            current_pos[:, 0, :] = final_root_pos[0]
            
        # 赋值给 Anim 对象
        self.anim.rotations.qs = current_quats
        self.anim.positions = current_pos
        
        quats_t = self.anim.rotations.qs.detach().cpu()  # float32 by default

        # 假设原始顺序是 [w, x, y, z]
        w = quats_t[..., 0]
        x = quats_t[..., 1]
        y = quats_t[..., 2]
        z = quats_t[..., 3]

        # 对所有关节统一做：q = [-q[1], q[2], -q[3], q[0]]
        quats_reordered = torch.stack([-x, y, -z, w], dim=-1)  # (F, J, 4)

        # 索引准备（建议你在 __init__ 里就算好，这里简单写）
        names = self.anim.names
        pelvis_idx = names.index("pelvis")
        head_idx   = names.index("head")

        # head 直接设为单位四元数 [0, 0, 0, 1]
        # quats_reordered[:, head_idx] = torch.tensor([0, 0, 0, 1], dtype=quats_reordered.dtype)

        # pelvis 需要再和 diff 相乘：q = qmul(q, diff)
        diff = torch.tensor([0.7071, 0.0, 0.0, 0.7071], dtype=quats_reordered.dtype)
        pelvis_q = quats_reordered[:, pelvis_idx]                  # (F, 4)
        diff_batch = diff.view(1, 4).expand(pelvis_q.size(0), 4)   # (F, 4)
        quats_reordered[:, pelvis_idx] = qmul(pelvis_q, diff_batch)

        # 一次性 round + 转 numpy，后面循环只做取值
        quats_np = quats_reordered.numpy().astype(np.float32)  # (F, J, 4)

        # ---------------------------
        # 2. 位置同样一次性取出来
        # ---------------------------
        poss_root = self.anim.positions[:, 0].detach().cpu().numpy()  # (F, 3)

        # ---------------------------
        # 3. 构造 quat_anim：只做轻量级 Python 循环
        # ---------------------------

        quat_anim = {
            "timestamp": 1754016912754,
            "fps": 30,
            "frames": []
        }

        body_len = len(self.skel.skeleton_json)
        # 映射动画关节到 skeleton 索引，避免在内层循环反复查
        joint_indices = [self.skel.skel_dict[joint_name] for joint_name in names]
        # print("joint_indices:", joint_indices)
        for frame_idx in range(quats_np.shape[0]):
            pos = poss_root[frame_idx]  # (3,)
            # 注意：不要用 [[...]] * body_len，会共用同一子列表
            body = [[0, 0, 0, 1] for _ in range(body_len)]

            frame = {
                "offset": [round(pos[0], 4), round(pos[2], 4), round(pos[1], 4)],
                "body": body,
                "face": self.skel.static_face
            }

            q_frame = quats_np[frame_idx]  # (J, 4)

            # 这里不再做任何数值运算，只是赋值
            for j, joint_name in enumerate(names):
                joint_index = joint_indices[j]
                q = q_frame[j].tolist()
                q = [round(v, 4) for v in q]
                frame["body"][joint_index] = q
            # ball_l / ball_r 固定写死
            # frame["body"][self.skel.skel_dict["ball_l"]] = [0, 0, 0.8509, 0.5253]
            # frame["body"][self.skel.skel_dict["ball_r"]] = [0, 0, 0.8509, 0.5253]

            quat_anim["frames"].append(frame)
            
        
        return quat_anim
    
    def save_quat_motion_to_bvh(
        self, 
        motion: Dict[str, np.ndarray],
        save_path: str = "sample.bvh",
        shape='wxyz'
        ) -> None:
        
        input_quat = motion["quat"]
        input_offset = motion["offset"]
        if not isinstance(input_quat, torch.Tensor):
            input_quat = torch.as_tensor(input_quat)  # 保留 numpy 时更高效
            if len(input_quat.shape) == 3:
                input_quat = input_quat[None]
        # offset: 同理
        if not isinstance(input_offset, torch.Tensor):
            input_offset = torch.as_tensor(input_offset)
            if len(input_offset.shape) == 2:
                input_offset = input_offset[None]
        final_quats, final_root_pos = process_batch_data(input_quat, input_offset, self.anim, self.skel.src_joint_dict, shape=shape)
        
        # 转为 CPU 准备写入
        final_quats = final_quats.cpu()
        if final_root_pos is not None:
            final_root_pos = final_root_pos.cpu()
        # 准备模板的基础位置 (F, Full_J, 3)
        num_frames = final_quats.shape[1]
        base_pos = self.anim.positions[0].clone().unsqueeze(0).repeat(num_frames, 1, 1) # (F, J, 3)

        current_quats = final_quats[0]
        
        # pos: (F, Full_J, 3)
        current_pos = base_pos.clone()
        if final_root_pos is not None:
            # 更新 Root (Index 0) 位置
            current_pos[:, 0, :] = final_root_pos[0]
            
        # 赋值给 Anim 对象
        self.anim.rotations.qs = current_quats
        self.anim.positions = current_pos
        
        # 写入文件
        BVH.save(save_path, self.anim, names=self.anim.names, frametime=1/20, order='zyx', quater=True)
        return self.anim
    
    def convert_quat_motion_to_ue(
        self, 
        motion: Dict[str, np.ndarray],
        shape='wxyz'
        ) -> None:
        
        input_quat = motion["quat"]
        input_offset = motion["offset"]
        quat_anim = {
            "timestamp": 1754016912754,
            "fps": 30,
            "frames": []
        }
        
        for frame_idx in range(input_quat.shape[0]):
            body_len = len(self.skel.skeleton_json)
            pos = input_offset[frame_idx]  # (3,)
            # 注意：不要用 [[...]] * body_len，会共用同一子列表
            body = [[0, 0, 0, 1] for _ in range(body_len)]

            frame = {
                "offset": [pos[0], pos[2], pos[1]],
                "body": body,
                "face": self.skel.static_face
            }
            q_frame = input_quat[frame_idx]  # (J, 4)
            print("\n\n===============================\n\n")
            # 这里不再做任何数值运算，只是赋值
            for j, joint_name in enumerate(self.skel.joint_list): # joint list 是按照ue驱动的骨骼点排序
                src_idx = self.skel.skel_dict.get(joint_name, -1) # mta输入和输出的映射
                if src_idx != -1:
                    idx = self.skel.mta_joint_idx_dict.get(src_idx) # 我们的排列格式与mta输入的映射
                    q = q_frame[idx].tolist()
                    q = [round(v, 4) for v in q]
                    frame["body"][j] = q
                    print(j, src_idx, idx, joint_name)
            # ball_l / ball_r 固定写死
            # frame["body"][self.skel.skel_dict["ball_l"]] = [0, 0, 0.8509, 0.5253]
            # frame["body"][self.skel.skel_dict["ball_r"]] = [0, 0, 0.8509, 0.5253]

            quat_anim["frames"].append(frame)
            
        return quat_anim
    
        
    def smooth_animation(self, motion:np.ndarray, window_length=11, polyorder=2, sigma=1.5, mode='savgol'):
        """
        平滑动画数据，处理角度周期性特性
        
        参数:
        motion: numpy数组，形状为(frames, joint_num, 3)，包含世界坐标数据
        window_length: Savitzky-Golay滤波器窗口长度（必须是奇数）
        polyorder: Savitzky-Golay滤波器多项式阶数
        sigma: 高斯滤波的标准差（当mode='gaussian'时使用）
        mode: 平滑模式，'savgol'或'gaussian'
        
        返回:
        平滑后的动画数据，形状与输入相同
        """
        # 参数校验
        if window_length % 2 == 0:
            window_length += 1  # 确保窗口长度为奇数
            print(f"警告: 窗口长度调整为奇数: {window_length}")
        
        if polyorder >= window_length:
            polyorder = window_length - 1
            print(f"警告: 多项式阶数调整为: {polyorder}")
        
        frames, joint_num, dims = motion.shape
        smoothed_data = np.zeros_like(motion)
        
        for joint in range(joint_num):
            for dim in range(dims):  # 对每个维度单独处理
                angle_series = motion[:, joint, dim]
                
                if mode == 'savgol':
                    # 使用Savitzky-Golay滤波器进行平滑
                    smoothed_series = savgol_filter(angle_series, 
                                                window_length, 
                                                polyorder,
                                                mode='mirror')  # 镜像边界处理
                elif mode == 'gaussian':
                    # 使用高斯滤波
                    smoothed_series = gaussian_filter1d(angle_series, 
                                                    sigma=sigma, 
                                                    mode='mirror')
                else:
                    raise ValueError("模式必须是'savgol'或'gaussian'")
                smoothed_data[:, joint, dim] = smoothed_series
        return smoothed_data
    
    def resample_motion(self, motion:np.ndarray, original_fps:int = 20, target_fps:int = 30):
        """
        data: 输入序列，shape=(frames, keypoints, 3)
        original_fps: 原始帧率（如24, 30）
        target_fps: 目标帧率（默认20）
        """
        frames, keypoints, dim = motion.shape
        t_original = np.arange(frames) / original_fps  # 原始时间轴
        max_time = t_original[-1]  # 总时长
        target_frames = int(max_time * target_fps)  # 目标帧数
        t_target = np.linspace(0, max_time, target_frames)  # 目标时间轴
        resampled_data = np.zeros((target_frames, keypoints, dim), dtype = motion.dtype)
        
        # 对每个关键点和坐标维度独立插值
        for k in range(keypoints):
            for d in range(dim):  # 处理X/Y/Z三个维度
                # 提取原始数据（一维时间序列）
                y_original = motion[:, k, d]
                # 创建插值函数（线性）
                f_interp = interp1d(
                    t_original, y_original, 
                    kind='cubic', # linear or cubic
                    bounds_error=False, 
                    fill_value="extrapolate"
                )
                # 生成新数据
                resampled_data[:, k, d] = f_interp(t_target)
        
        return resampled_data
    
    def uniform_skeleton(self, motion:np.ndarray):
        """
        Uniformly scales a skeleton to match a target offset.

        Args:
            positions (numpy.ndarray): Input skeleton joint positions.
            target_offset (torch.Tensor): Target offset for the skeleton.

        Returns:
            numpy.ndarray: New joint positions after scaling and inverse/forward kinematics.
        """
            
        # Creating a skeleton with a predefined kinematic chain
        src_skel = Skeleton(self.skel.n_raw_offsets, self.skel.kinematic_chain, 'cpu')

        # Calculate the global offset of the source skeleton
        src_offset = src_skel.get_offsets_joints(torch.from_numpy(motion[0]))
        src_offset = src_offset.numpy()
        tgt_offset = self.skel.tgt_offsets.numpy()

        # Calculate Scale Ratio as the ratio of legs
        src_leg_len = np.abs(src_offset[self.skel.l_idx1]).max() + np.abs(src_offset[self.skel.l_idx2]).max()
        tgt_leg_len = np.abs(tgt_offset[self.skel.l_idx1]).max() + np.abs(tgt_offset[self.skel.l_idx2]).max()
        
        # Scale ratio for uniform scaling
        scale_rt = tgt_leg_len / src_leg_len
        
        # Extract the root position of the source skeleton
        src_root_pos = motion[:, 0]
        
        # Scale the root position based on the calculated ratio
        tgt_root_pos = src_root_pos * scale_rt
        
        # Inverse Kinematics to get quaternion parameters
        quat_params = src_skel.inverse_kinematics_np(motion, self.skel.face_joint_indx)
        
        # Forward Kinematics with the new root position and target offset
        src_skel.set_offset(self.skel.tgt_offsets)
        new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
        error = np.mean(
            np.sum((motion - new_joints) ** 2.0, axis=-1) ** 0.5
        )
        return new_joints
    
    def convert_to_quat(self, motion:np.ndarray, ik_iter:int = 10):
        uni_position = [motion[:, self.skel.src_joint_dict[joint]] for joint in self.skel.joint_list]
        joints = np.stack(uni_position, axis=1).astype(np.float32)
        
        # ========= 转为四元数 =========
        bvh_anim, quat_anim, glb = self.quat_converter.convert(
            joints, iterations=ik_iter, device="cpu", silent = False
        )
        # BVH.save(bvh_save_path, new_anim, names=new_anim.names, frametime=1 / 30, order='zyx', quater=True)
        return quat_anim, bvh_anim
    
    def postprocess_pipline(self, motion:np.ndarray, ik_iter:int = 10, return_bvh_anim = False):
        motion = self.smooth_animation(motion)
        motion = self.resample_motion(motion)
        motion = self.uniform_skeleton(motion)
        quat_anim, bvh_anim = self.convert_to_quat(motion, ik_iter)
        if return_bvh_anim:
            return quat_anim, bvh_anim
        else:
            return quat_anim
    
    def postprocess_pipline_quat(self, motion: Dict[str, np.ndarray]):
        
        offset, quat = motion["offset"], motion["quat"]
        # offset = offset[:, None]
        # # quat = self.resample_motion(quat, 20, 30)
        # offset = self.resample_motion(offset, 20, 30)
        
        # # quat = self.smooth_animation(quat)
        # offset = self.smooth_animation(offset)
        
        # # offset = offset[:, 0] 
        smooth_motion = {"quat": quat, "offset": offset} # offset: (frames, 3) quat: (frames, joint_num, 4) 
        print("offset:", offset.shape, "quat:", quat.shape) 
        quat_anim = self.convert_quat_motion_to_ue_from_bvh(motion=smooth_motion)
        # self.save_quat_motion_to_bvh(motion=smooth_motion, save_path = "./debug/quat_anim.bvh")
        return quat_anim