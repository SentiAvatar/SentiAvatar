#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Motion Data 可视化工具

将 motion_data 目录中的 .npy 动作数据转换为 BVH 文件，便于在 Blender 等软件中查看。

数据格式:
    每个 .npy 文件是一个 dict:
    {
        "body": (T, 153) ndarray,     # offset(3) + body_6d(25*6=150)
        "left": (T, 120) ndarray,     # left_hand_6d(20*6)
        "right": (T, 120) ndarray,    # right_hand_6d(20*6)
    }

用法:
    # 单文件转换
    python tools/visualize_motion.py --input data/motion_data/xxx/xxx.npy --output output.bvh

    # 批量转换
    python tools/visualize_motion.py --input_dir data/motion_data --output_dir output_bvh --max_files 10

@Author  :   Chuhao Jin
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from glob import glob

# 添加 motion_generation 到 path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "motion_generation"))

from utils.rotation_utils import sixd_to_quaternion
from utils.constants import BODY_JOINTS_ID, LEFT_HAND_JOINTS_ID, RIGHT_HAND_JOINTS_ID
from actions.postprocess import MotionPostprocesser


def _gaussian_kernel1d(kernel_size, sigma, device, dtype):
    half = kernel_size // 2
    x = torch.arange(-half, half + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / (k.sum() + 1e-12)


def smooth_motion_gaussian(x, kernel_size=7, sigma=2.0):
    """对动作序列进行高斯平滑 (B,T,J,D)"""
    B, T, J, D = x.shape
    device, dtype = x.device, x.dtype
    y = x.permute(0, 2, 3, 1).contiguous().view(B, J * D, T)
    k = _gaussian_kernel1d(kernel_size, sigma, device, dtype)
    weight = k.view(1, 1, kernel_size).repeat(J * D, 1, 1)
    pad = kernel_size // 2
    y_pad = F.pad(y, (pad, pad), mode="reflect")
    y_smooth = F.conv1d(y_pad, weight=weight, bias=None, stride=1, padding=0, groups=J * D)
    return y_smooth.view(B, J, D, T).permute(0, 3, 1, 2).contiguous()


def resample_fps(x, src_fps=20.0, tgt_fps=30.0):
    """帧率重采样 (B,T,J,D)"""
    B, T, J, D = x.shape
    new_T = max(1, int(round(T * (tgt_fps / src_fps))))
    y = x.permute(0, 2, 3, 1).contiguous().view(B, J * D, T)
    y2 = F.interpolate(y, size=new_T, mode="linear", align_corners=False)
    return y2.view(B, J, D, new_T).permute(0, 3, 1, 2).contiguous()


def npy_to_motion(motion_dict, src_fps=20.0, tgt_fps=30.0, device="cpu"):
    """
    将 npy 格式的动作数据转换为 {"offset": (T,3), "quat": (T,63,4)} 格式
    
    Args:
        motion_dict: dict with "body", "left", "right" keys
        src_fps: 源帧率
        tgt_fps: 目标帧率
        device: 计算设备
    
    Returns:
        dict: {"offset": (T, 3) ndarray, "quat": (T, 63, 4) ndarray}
    """
    device = torch.device(device)
    
    body_motion = torch.tensor(motion_dict["body"], dtype=torch.float32).to(device)
    left_motion = torch.tensor(motion_dict["left"], dtype=torch.float32).to(device)
    right_motion = torch.tensor(motion_dict["right"], dtype=torch.float32).to(device)
    
    if left_motion.dim() == 3:
        left_motion = left_motion.squeeze(0)
    if right_motion.dim() == 3:
        right_motion = right_motion.squeeze(0)
    
    frames = body_motion.shape[0]
    
    # 分离 offset 和 body 6D 旋转
    body_offset = body_motion[:, :3].clone()
    body_6d = body_motion[:, 3:]
    
    # offset: 累加速度得到位移
    offset_frame0 = torch.tensor([0.0, 0.0, 102.0]).to(device)
    body_offset[:, 2] = body_offset[:, 2] - body_offset[0, 2]
    for i in range(1, body_offset.shape[0]):
        body_offset[i] = body_offset[i] - body_offset[i - 1]
    # actually the raw data already has velocity form, let's just accumulate
    pred_offset_vel = body_motion[:, :3].clone()
    pred_offset_vel[:, 2] = pred_offset_vel[:, 2] - pred_offset_vel[0, 2]
    pred_offset_vel[1:, :3] = pred_offset_vel[1:, :3] - pred_offset_vel[:-1, :3]
    for i in range(1, pred_offset_vel.shape[0]):
        pred_offset_vel[i] = pred_offset_vel[i] + pred_offset_vel[i - 1]
    pred_offset = (pred_offset_vel + offset_frame0).reshape(1, frames, 1, 3)
    
    # 重塑为 (1, T, J, 6)
    pred_body_6d = body_6d.reshape(1, frames, 25, 6)
    pred_left = left_motion[:frames].reshape(1, frames, 20, 6)
    pred_right = right_motion[:frames].reshape(1, frames, 20, 6)
    
    # 平滑 + 重采样
    pred_body_6d = resample_fps(smooth_motion_gaussian(pred_body_6d), src_fps, tgt_fps)
    pred_left = resample_fps(smooth_motion_gaussian(pred_left), src_fps, tgt_fps)
    pred_right = resample_fps(smooth_motion_gaussian(pred_right), src_fps, tgt_fps)
    pred_offset = resample_fps(smooth_motion_gaussian(pred_offset), src_fps, tgt_fps)
    
    new_frames = pred_body_6d.shape[1]
    
    # 6D → 四元数
    pred_body_quat = sixd_to_quaternion(pred_body_6d.reshape(-1, 6)).reshape(1, new_frames, 25, 4)
    pred_left_quat = sixd_to_quaternion(pred_left.reshape(-1, 6)).reshape(1, new_frames, 20, 4)
    pred_right_quat = sixd_to_quaternion(pred_right.reshape(-1, 6)).reshape(1, new_frames, 20, 4)
    
    offset_np = pred_offset.reshape(new_frames, 3).detach().cpu().numpy()
    
    # 合并 63 关节四元数
    merge_quat = torch.zeros(new_frames, 63, 4).to(device)
    merge_quat[:, BODY_JOINTS_ID] = pred_body_quat[0]
    merge_quat[:, LEFT_HAND_JOINTS_ID[1:]] = pred_left_quat[0, :, 1:]
    merge_quat[:, RIGHT_HAND_JOINTS_ID[1:]] = pred_right_quat[0, :, 1:]
    
    quat_np = merge_quat.detach().cpu().numpy()
    
    return {"offset": offset_np, "quat": quat_np}


def convert_single(input_path, output_path, src_fps=20.0, tgt_fps=30.0, save_json=False):
    """转换单个文件"""
    print(f"  Loading: {input_path}")
    motion_dict = np.load(input_path, allow_pickle=True)
    if isinstance(motion_dict, np.ndarray) and motion_dict.dtype == object:
        motion_dict = motion_dict.item()
    
    motion = npy_to_motion(motion_dict, src_fps=src_fps, tgt_fps=tgt_fps)
    print(f"  Motion: offset={motion['offset'].shape}, quat={motion['quat'].shape}")
    
    postprocesser = MotionPostprocesser()
    
    # 保存 BVH
    postprocesser.save_quat_motion_to_bvh(motion=motion, save_path=output_path)
    print(f"  Saved BVH: {output_path}")
    
    # 可选保存 JSON
    if save_json:
        import json
        json_path = output_path.replace(".bvh", ".json")
        anim = postprocesser.convert_quat_motion_to_ue_from_bvh(motion=motion)
        with open(json_path, "w") as f:
            json.dump(anim, f, indent=2, ensure_ascii=False)
        print(f"  Saved JSON: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="将 motion_data 中的 .npy 动作数据转换为 BVH 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单文件转换
  python tools/visualize_motion.py \\
      --input data/motion_data/xxx/xxx.npy \\
      --output output.bvh

  # 批量转换
  python tools/visualize_motion.py \\
      --input_dir data/motion_data \\
      --output_dir output_bvh \\
      --max_files 10

  # 同时输出 JSON
  python tools/visualize_motion.py \\
      --input data/motion_data/xxx/xxx.npy \\
      --output output.bvh \\
      --save_json
        """,
    )
    
    parser.add_argument("--input", type=str, default=None, help="单个 .npy 文件路径")
    parser.add_argument("--output", type=str, default=None, help="输出 .bvh 文件路径")
    parser.add_argument("--input_dir", type=str, default=None, help="批量模式：输入目录")
    parser.add_argument("--output_dir", type=str, default="./output_bvh", help="批量模式：输出目录")
    parser.add_argument("--max_files", type=int, default=None, help="最大转换文件数")
    parser.add_argument("--src_fps", type=float, default=20.0, help="源帧率 (默认: 20)")
    parser.add_argument("--tgt_fps", type=float, default=30.0, help="目标帧率 (默认: 30)")
    parser.add_argument("--save_json", action="store_true", help="同时保存 anim.json")
    
    args = parser.parse_args()
    
    if args.input:
        # 单文件模式
        output = args.output or args.input.replace(".npy", ".bvh")
        os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
        convert_single(args.input, output, args.src_fps, args.tgt_fps, args.save_json)
    
    elif args.input_dir:
        # 批量模式
        npy_files = sorted(glob(os.path.join(args.input_dir, "**/*.npy"), recursive=True))
        if args.max_files:
            npy_files = npy_files[:args.max_files]
        
        print(f"找到 {len(npy_files)} 个 .npy 文件")
        os.makedirs(args.output_dir, exist_ok=True)
        
        for i, npy_path in enumerate(npy_files):
            rel_path = os.path.relpath(npy_path, args.input_dir)
            bvh_path = os.path.join(args.output_dir, rel_path.replace(".npy", ".bvh"))
            os.makedirs(os.path.dirname(bvh_path), exist_ok=True)
            
            print(f"\n[{i+1}/{len(npy_files)}]")
            try:
                convert_single(npy_path, bvh_path, args.src_fps, args.tgt_fps, args.save_json)
            except Exception as e:
                print(f"  ❌ 转换失败: {e}")
        
        print(f"\n转换完成！输出目录: {args.output_dir}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
