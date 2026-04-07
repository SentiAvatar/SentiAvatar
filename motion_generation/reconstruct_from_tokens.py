#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Motion Token → Motion Sequence 重建脚本

从 pipeline 推理结果（JSON）中读取 dense_tokens 和 gt_tokens，
使用 RVQVAE 解码器重建 motion sequence，保存为 BVH/JSON 以便对比。

输入：pipeline_results.json（包含 dense_tokens 和 gt_tokens）
输出：每个样本生成 pred.bvh / pred.json 和 gt.bvh / gt.json

用法:
    python reconstruct_from_tokens.py \
        --input_json ../Motion_mask_transformer_audio/pipeline_demo_result.json \
        --checkpoint_path ./checkpoints/quat63nodes_v2_0120/gqzV4/model/epoch_30.pth \
        --output_dir ./output_gen/pipeline_reconstruct

@File    :   reconstruct_from_tokens.py
@Time    :   2025/07/16
"""

import os
import sys
import json
import argparse
import shutil
import numpy as np
import wave
import torch
import torch.nn.functional as F
import io 
from typing import List, Optional, Dict, Any

# 确保能导入本地模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from infer import (
    load_config_from_checkpoint,
    load_model,
    fixseed,
    smooth_then_resample,
)
from models.rvqvae import RVQVAE
from configs.default_config import Config
from actions.postprocess import MotionPostprocesser
from actions.schema import MotionTokens
from utils.rotation_utils import sixd_to_quaternion
from utils.constants import BODY_JOINTS_ID, LEFT_HAND_JOINTS_ID, RIGHT_HAND_JOINTS_ID
try:
    from susu_face_speech_align import infer_face_vqvae, load_face_vqvae, load_chinese_hubert
    FACE_MODEL_AVAILABLE = True
except ImportError:
    FACE_MODEL_AVAILABLE = False
    print("Warning: Face animation models not available. Will skip face animation.")
    
# ======================================================================
#  解码函数
# ======================================================================

def decode_body_tokens(
    model: RVQVAE,
    body_tokens: list,
    placeholder_motion_dict: dict,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    src_fps: float = 20.0,
    tgt_fps: float = 30.0,
) -> dict:
    """
    将 body motion tokens 解码为 motion sequence（offset + quaternion）。

    Args:
        model: RVQVAE 模型
        body_tokens: list of [res1, res2, res3, res4] per frame
        placeholder_motion_dict: 占位符动作数据（提供手部数据）
        mean, std: 归一化参数
        device: 推理设备
        src_fps: 模型输出帧率
        tgt_fps: 目标输出帧率

    Returns:
        dict: {"offset": (T, 3) ndarray, "quat": (T, 63, 4) ndarray}
    """
    with torch.no_grad():
        # 编码 tokens → 解码
        tokens_tensor = torch.tensor(
            np.array(body_tokens), dtype=torch.long
        ).unsqueeze(0).to(device)  # (1, N, 4)

        code_idx_dict = {"body": tokens_tensor}
        x_out = model.forward_decoder(code_idx_dict)
        pred_whole_motion = x_out.sum(0)  # (N, D)

        # 反归一化
        pred_whole_motion = pred_whole_motion * std + mean

        frames = pred_whole_motion.shape[0]

        # 分离 offset（前3维是根节点速度）和 body 6D 旋转
        offset_frame0 = torch.tensor([0.0, 0.0, 102.0], device=device)
        pred_offset_vel = pred_whole_motion[:, :3].clone()
        for i in range(1, pred_offset_vel.shape[0]):
            pred_offset_vel[i] = pred_offset_vel[i] + pred_offset_vel[i - 1]
        pred_offset = (pred_offset_vel + offset_frame0).reshape(1, frames, 1, 3)

        pred_body_6d = pred_whole_motion[:, 3:].reshape(1, frames, 25, 6)

        # 加载占位符手部数据
        left_motion = torch.tensor(
            placeholder_motion_dict["left"], dtype=torch.float32
        ).unsqueeze(0).to(device)
        right_motion = torch.tensor(
            placeholder_motion_dict["right"], dtype=torch.float32
        ).unsqueeze(0).to(device)

        # 对齐手部帧数
        num_ph_frames = left_motion.shape[1]
        if frames > num_ph_frames:
            pad = frames - num_ph_frames
            left_motion = F.pad(
                left_motion.permute(0, 2, 1), (0, pad), mode="replicate"
            ).permute(0, 2, 1)
            right_motion = F.pad(
                right_motion.permute(0, 2, 1), (0, pad), mode="replicate"
            ).permute(0, 2, 1)

        pred_left = left_motion[:, :frames].reshape(1, frames, 20, 6)
        pred_right = right_motion[:, :frames].reshape(1, frames, 20, 6)

        # 平滑 + 重采样
        pred_body_6d = smooth_then_resample(pred_body_6d, src_fps=src_fps, tgt_fps=tgt_fps)
        pred_left = smooth_then_resample(pred_left, src_fps=src_fps, tgt_fps=tgt_fps)
        pred_right = smooth_then_resample(pred_right, src_fps=src_fps, tgt_fps=tgt_fps)
        pred_offset = smooth_then_resample(pred_offset, src_fps=src_fps, tgt_fps=tgt_fps)

        new_frames = pred_right.shape[1]

        # 6D → 四元数
        pred_body_quat = sixd_to_quaternion(
            pred_body_6d.reshape(-1, 6)
        ).reshape(1, new_frames, 25, 4)
        pred_left_quat = sixd_to_quaternion(
            pred_left.reshape(-1, 6)
        ).reshape(1, new_frames, 20, 4)
        pred_right_quat = sixd_to_quaternion(
            pred_right.reshape(-1, 6)
        ).reshape(1, new_frames, 20, 4)

        offset_np = pred_offset.reshape(new_frames, 3).detach().cpu().numpy()

        # 合并 63 关节四元数
        merge_quat = torch.zeros(new_frames, 63, 4, device=device)
        merge_quat[:, BODY_JOINTS_ID] = pred_body_quat[0]
        merge_quat[:, LEFT_HAND_JOINTS_ID[1:]] = pred_left_quat[0, :, 1:]
        merge_quat[:, RIGHT_HAND_JOINTS_ID[1:]] = pred_right_quat[0, :, 1:]

        quat_np = merge_quat.detach().cpu().numpy()

    return {"offset": offset_np, "quat": quat_np}


def add_face_animation_to_json(
    anim_data: Dict,
    audio_bytes: bytes,
    sample_rate: int,
    face_data: np.ndarray,
    audio_feature_extractor,
    audio_encoder,
    face_vq_model,
    demo_id: str = "unknown"
) -> Dict:
    """
    为动画JSON添加面部动画数据
    
    Args:
        anim_data: 动画JSON数据，包含 "frames" 字段
        audio_bytes: 音频文件的字节数据
        sample_rate: 音频采样率
        face_data: 面部数据模板 (用于非唇部表情)
        audio_feature_extractor: Hubert特征提取器
        audio_encoder: Hubert音频编码器
        face_vq_model: 面部VQ模型
        demo_id: 样本ID（用于日志）
    
    Returns:
        添加了面部数据的动画JSON
    """
    expect_frames = len(anim_data["frames"])
    
    # 准备面部数据模板
    only_face = face_data.tolist()
    only_face = [{"face": item} for item in only_face]
    face_frame = len(only_face)
    
    # 获取音频帧数
    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
        n_frames = wf.getnframes()
        tts_hz = wf.getframerate()
    
    # 构建面部推理的输入数据
    face_payload = {
        "name": demo_id,
        "id": demo_id,
        "frame": face_frame,
        "expect_frame": expect_frames,
        "fps": 30,
        "motion": only_face,
        "tts": audio_bytes,
        "tts_frames": n_frames,
        "tts_hz": tts_hz,
    }
    
    # 执行面部动画推理
    print(f"Generating face animation for {demo_id}...")
    face_payload = infer_face_vqvae(face_payload, audio_feature_extractor, audio_encoder, face_vq_model)
    aligned_face = face_payload["motion"]
    
    # 验证帧数匹配
    if len(aligned_face) != expect_frames:
        print(f"Warning: Face frames ({len(aligned_face)}) != expected frames ({expect_frames})")
        # 调整面部帧数以匹配
        if len(aligned_face) < expect_frames:
            # 填充最后一帧
            last_face = aligned_face[-1] if aligned_face else {"face": [0.0] * 52}
            while len(aligned_face) < expect_frames:
                aligned_face.append(last_face)
        else:
            # 截断
            aligned_face = aligned_face[:expect_frames]
    
    # 将面部数据添加到每一帧
    for i in range(expect_frames):
        anim_data["frames"][i]["face"] = [round(v, 4) for v in aligned_face[i]["face"]]
    
    print(f"Added face animation to {expect_frames} frames")
    return anim_data

# ======================================================================
#  主流程
# ======================================================================

def process_single_result(
    result: dict,
    model: RVQVAE,
    placeholder_motion_dict: dict,
    mean: torch.Tensor,
    std: torch.Tensor,
    postprocesser: MotionPostprocesser,
    device: torch.device,
    output_dir: str,
    src_fps: float,
    tgt_fps: float,
    wave_folder: str = None,
    face_data: np.array = None,
):
    """处理单个样本：解码 dense_tokens 和 gt_tokens，保存 BVH/JSON"""
    
    name = result.get("name", "unknown")
    safe_name = name.replace("/", "_").replace("\\", "_")
    dense_tokens = result.get("dense_tokens", [])
    gt_tokens = result.get("gt_tokens", [])

    print(f"\n{'=' * 60}")
    print(f"  重建样本: {name}")
    print(f"  dense_tokens: {len(dense_tokens)} 帧, gt_tokens: {len(gt_tokens)} 帧")
    print(f"{'=' * 60}")

    src_audio_file = f"{wave_folder}/{name}.wav"
    # 读取音频文件
    with open(src_audio_file, "rb") as f:
        audio_bytes = f.read()
    
    # 获取音频采样率
    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
        sample_rate = wf.getframerate()
            
    # ---- 解码 dense_tokens (预测) ----
    if dense_tokens:
        print("  [Pred] 解码 dense_tokens ...")
        pred_motion = decode_body_tokens(
            model, dense_tokens, placeholder_motion_dict,
            mean, std, device, src_fps, tgt_fps,
        )
        print(f"  [Pred] offset: {pred_motion['offset'].shape}, quat: {pred_motion['quat'].shape}")

        pred_name = f"{safe_name}_pred"
        # JSON
        pred_anim = postprocesser.convert_quat_motion_to_ue_from_bvh(motion=pred_motion)
        pred_anim = add_face_animation_to_json(
            anim_data=pred_anim,
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
            face_data=face_data,
            audio_feature_extractor=audio_feature_extractor,
            audio_encoder=audio_encoder,
            face_vq_model=face_vq_model,
            demo_id=safe_name
        )
        
        json_path = os.path.join(output_dir, f"{pred_name}.json")
        with open(json_path, "w") as f:
            json.dump(pred_anim, f, indent=2, ensure_ascii=False)
        print(f"  [Pred] 保存 JSON → {json_path}")

        # BVH
        bvh_path = os.path.join(output_dir, f"{pred_name}.bvh")
        postprocesser.save_quat_motion_to_bvh(motion=pred_motion, save_path=bvh_path)
        print(f"  [Pred] 保存 BVH  → {bvh_path}")
    else:
        print("  [Pred] 无 dense_tokens，跳过")

    # ---- 解码 gt_tokens (Ground Truth) ----
    if gt_tokens:
        print("  [GT]   解码 gt_tokens ...")
        gt_motion = decode_body_tokens(
            model, gt_tokens, placeholder_motion_dict,
            mean, std, device, src_fps, tgt_fps,
        )
        print(f"  [GT]   offset: {gt_motion['offset'].shape}, quat: {gt_motion['quat'].shape}")

        gt_name = f"{safe_name}_gt"
        # JSON
        gt_anim = postprocesser.convert_quat_motion_to_ue_from_bvh(motion=gt_motion)
        gt_anim = add_face_animation_to_json(
            anim_data=gt_anim,
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
            face_data=face_data,
            audio_feature_extractor=audio_feature_extractor,
            audio_encoder=audio_encoder,
            face_vq_model=face_vq_model,
            demo_id=safe_name
        )
        
        json_path = os.path.join(output_dir, f"{gt_name}.json")
        with open(json_path, "w") as f:
            json.dump(gt_anim, f, indent=2, ensure_ascii=False)
        print(f"  [GT]   保存 JSON → {json_path}")

        # BVH
        bvh_path = os.path.join(output_dir, f"{gt_name}.bvh")
        postprocesser.save_quat_motion_to_bvh(motion=gt_motion, save_path=bvh_path)
        print(f"  [GT]   保存 BVH  → {bvh_path}")
    else:
        print("  [GT]   无 gt_tokens，跳过")

    # ---- 复制对应的音频文件（如果有） ----
    if wave_folder and name != "unknown":
        src_wav = os.path.join(wave_folder, f"{name}.wav")
        if os.path.exists(src_wav):
            dst_wav = os.path.join(output_dir, f"{safe_name}.wav")
            shutil.copy(src_wav, dst_wav)
            print(f"  [Audio] 复制 → {dst_wav}")


def main():
    parser = argparse.ArgumentParser(
        description="从 pipeline 推理结果重建 motion sequence（pred vs GT 对比）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单样本 (demo result)
  python reconstruct_from_tokens.py \\
      --input_json ../Motion_mask_transformer_audio/pipeline_demo_result.json \\
      --output_dir ./output_gen/pipeline_reconstruct

  # 批量 (batch results)
  python reconstruct_from_tokens.py \\
      --input_json ../Motion_mask_transformer_audio/pipeline_batch_results.json \\
      --output_dir ./output_gen/pipeline_batch_reconstruct
        """,
    )

    parser.add_argument("--input_json", type=str, required=True,
                        help="pipeline 推理结果 JSON 文件路径")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="RVQVAE 模型 checkpoint 路径")
    parser.add_argument("--placeholder_npy", type=str, default=None,
                        help="占位符动作数据 npy 路径（提供手部数据）")
    parser.add_argument("--output_dir", type=str, default="./output_gen/pipeline_reconstruct",
                        help="输出目录")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="推理设备")
    parser.add_argument("--src_fps", type=float, default=None,
                        help="模型输出帧率（None 则从 config 读取）")
    parser.add_argument("--tgt_fps", type=float, default=30.0,
                        help="目标输出帧率")
    parser.add_argument("--wave_folder", type=str, default=None,
                        help="音频文件目录（可选，用于复制对应音频）")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大处理样本数（None 表示全部）")
    parser.add_argument("--face_npy_path", type=str,
                        default=None,
                        help="面部动画模板数据路径")
    
    args = parser.parse_args()

    # ---- 默认路径 ----
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    module_dir = os.path.dirname(os.path.abspath(__file__))
    if args.checkpoint_path is None:
        args.checkpoint_path = os.path.join(
            project_dir, "checkpoints/rvqvae/model/epoch_30.pth"
        )
    if args.placeholder_npy is None:
        args.placeholder_npy = os.path.join(
            module_dir, "meta/xiu_joint_quat_vecs/Daiji_A_001_V001.npy"
        )
    if args.wave_folder is None:
        args.wave_folder = os.path.join(
            project_dir, "data/wav_data"
        )

    # ---- 加载 pipeline 结果 ----
    print(f"[数据] 加载推理结果: {args.input_json}")
    with open(args.input_json, "r") as f:
        data = json.load(f)

    # 支持单样本（dict）和批量（list）两种格式
    if isinstance(data, dict):
        results = [data]
    elif isinstance(data, list):
        results = data
    else:
        print(f"❌ 不支持的 JSON 格式: {type(data)}")
        return

    print(f"[数据] 共 {len(results)} 个样本")

    if args.max_samples is not None:
        results = results[:args.max_samples]
        print(f"[数据] 限制处理前 {args.max_samples} 个样本")

    # ---- 加载 RVQVAE 模型 ----
    print(f"\n[模型] 加载 RVQVAE: {args.checkpoint_path}")
    config = load_config_from_checkpoint(args.checkpoint_path)

    if args.src_fps is None:
        args.src_fps = float(config.data.fps)
        
    if args.face_npy_path is None:
        args.face_npy_path = os.path.join(module_dir, "meta/face_anim/20260120_1632_Exp_Basic_Neutral_M_A_3_iphone_cal.npy")
    print("Loading face animation models...")
    face_data = np.load(args.face_npy_path)
    print(f"Loaded face template data: shape={face_data.shape}")
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint_path, config, device)

    # ---- 加载 Face 模型 ----
    if FACE_MODEL_AVAILABLE:
        face_vq_model, weight_matrix, weight_matrix_R_I = load_face_vqvae()
        audio_encoder, audio_feature_extractor = load_chinese_hubert()
        print("Face animation models loaded successfully")
    else:
        face_vq_model = audio_encoder = audio_feature_extractor = None

    # ---- 加载归一化参数 ----
    meta_dir = os.path.join(module_dir, "meta/mta_gen_demo")
    mean = torch.tensor(np.load(os.path.join(meta_dir, "mean.npy"))).to(device)
    std = torch.tensor(np.load(os.path.join(meta_dir, "std.npy"))).to(device)
    print(f"[模型] mean/std 加载完成, shape: {mean.shape}")

    # ---- 加载占位符动作数据 ----
    print(f"[数据] 加载占位符: {args.placeholder_npy}")
    placeholder_motion_dict = np.load(args.placeholder_npy, allow_pickle=True).item()

    # ---- 后处理器 ----
    postprocesser = MotionPostprocesser()

    # ---- 创建输出目录 ----
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 逐样本处理 ----
    for idx, result in enumerate(results):
        print(f"\n[{idx + 1}/{len(results)}]", end="")
        process_single_result(
            result=result,
            model=model,
            placeholder_motion_dict=placeholder_motion_dict,
            mean=mean,
            std=std,
            postprocesser=postprocesser,
            device=device,
            output_dir=args.output_dir,
            src_fps=args.src_fps,
            tgt_fps=args.tgt_fps,
            wave_folder=args.wave_folder,
            face_data=face_data,
        )

    print(f"\n{'=' * 60}")
    print(f"  重建完成！共处理 {len(results)} 个样本")
    print(f"  输出目录: {args.output_dir}")
    print(f"  每个样本生成:")
    print(f"    - *_pred.json / *_pred.bvh  (预测结果)")
    print(f"    - *_gt.json   / *_gt.bvh    (Ground Truth)")
    print(f"    - *.wav                      (对应音频)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()