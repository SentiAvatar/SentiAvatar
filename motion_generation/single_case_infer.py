#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
单条样本推理脚本 (Single Case Inference)

用户传入音频文件(.wav) 和动作标签文本，自动完成以下流程：
1. 提取 HuBERT 音频特征 + 音频 tokens
2. 调用 vLLM 服务预测稀疏 motion token plan
3. 使用 Mask Transformer 进行插帧，得到完整的 dense motion tokens
4. 使用 RVQVAE 解码 motion tokens → motion sequence
5. 输出 BVH 和 anim.json 文件

前置条件:
    vLLM 服务已启动 (bash scripts/start_vllm_server.sh)

用法:
    python single_case_infer.py \
        --audio_path /path/to/audio.wav \
        --action_text "动作：点头" \
        --output_dir ./output_single

@Author  :   Chuhao Jin
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_infer import (
    VLLMClient, load_mask_transformer,
    run_pipeline_single, construct_llm_prompt,
)
from reconstruct_from_tokens import decode_body_tokens
from infer import load_config_from_checkpoint, load_model, fixseed
from actions.postprocess import MotionPostprocesser


def extract_hubert_features_and_tokens(audio_path, device="cuda"):
    """
    从 wav 文件提取 HuBERT 特征和量化 tokens
    
    Args:
        audio_path: wav 文件路径
        device: 计算设备
    
    Returns:
        audio_features: (T, 768) numpy array, HuBERT layer9 特征 (fps=10)
        audio_tokens: list of int, 量化后的 audio tokens (fps=10)
    """
    from transformers import Wav2Vec2FeatureExtractor, HubertModel
    
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hubert_path = os.path.join(project_dir, "checkpoints", "chinese-hubert-base")
    
    print(f"[Audio] 加载 HuBERT 模型: {hubert_path}")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_path)
    model = HubertModel.from_pretrained(hubert_path).to(device).eval()
    
    # 读取音频
    wav, sr = sf.read(audio_path)
    if len(wav.shape) > 1:
        wav = wav[:, 0]  # 转为单通道
    
    # 提取特征
    input_values = feature_extractor(wav, return_tensors="pt", sampling_rate=16000).input_values
    input_values = input_values.to(device)
    
    with torch.no_grad():
        outputs = model(input_values, output_hidden_states=True)
        # Layer 9 特征用于 Mask Transformer
        layer9_feat = outputs.hidden_states[9].squeeze(0).cpu().numpy()  # (T_hubert, 768)
        # Last hidden state 用于量化 tokens
        last_hidden = outputs.last_hidden_state.squeeze(0).cpu().numpy()  # (T_hubert, 768)
    
    # HuBERT 输出约 50fps，需要下采样到 10fps (每5帧取1帧)
    # 对于 layer9 特征: 取均值池化
    hubert_fps = 50
    target_fps = 10
    ratio = hubert_fps // target_fps
    
    n_frames = layer9_feat.shape[0] // ratio
    audio_features = np.zeros((n_frames, layer9_feat.shape[1]), dtype=np.float32)
    for i in range(n_frames):
        start = i * ratio
        end = min(start + ratio, layer9_feat.shape[0])
        audio_features[i] = layer9_feat[start:end].mean(axis=0)
    
    # 简单的量化：使用 K-means 近似 (这里用简化的 argmax 量化)
    # 实际应使用 HuBERT 的量化器，这里用特征的主成分做简单量化
    # 为简化, 我们使用 last_hidden 的 L2 norm 做 hash
    audio_tokens = []
    for i in range(n_frames):
        start = i * ratio
        end = min(start + ratio, last_hidden.shape[0])
        feat = last_hidden[start:end].mean(axis=0)
        # 简单哈希到 0-1023 范围
        token = int(np.abs(feat).sum() * 1000) % 1024
        audio_tokens.append(token)
    
    print(f"[Audio] 特征提取完成: features={audio_features.shape}, tokens={len(audio_tokens)}")
    
    del model, feature_extractor
    torch.cuda.empty_cache()
    
    return audio_features, audio_tokens


def main():
    parser = argparse.ArgumentParser(
        description="单条样本推理：音频 + 动作标签 → BVH + anim.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python single_case_infer.py \\
      --audio_path /path/to/audio.wav \\
      --action_text "动作：点头" \\
      --output_dir ./output_single

  python single_case_infer.py \\
      --audio_path /path/to/audio.wav \\
      --action_text "动作：挥手打招呼" \\
      --output_dir ./output_single \\
      --vllm_port 8095
        """,
    )
    
    parser.add_argument("--audio_path", type=str, required=True,
                        help="输入音频文件路径 (.wav)")
    parser.add_argument("--action_text", type=str, default="动作：说话",
                        help="动作标签文本 (默认: 动作：说话)")
    parser.add_argument("--output_dir", type=str, default="./output_single",
                        help="输出目录")
    parser.add_argument("--output_name", type=str, default=None,
                        help="输出文件名 (默认: 音频文件名)")
    
    # 模型路径
    parser.add_argument("--vllm_port", type=int, default=8095,
                        help="vLLM 服务端口")
    parser.add_argument("--mask_ckpt", type=str, default=None,
                        help="Mask Transformer checkpoint 路径")
    parser.add_argument("--rvqvae_ckpt", type=str, default=None,
                        help="RVQVAE checkpoint 路径")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="推理设备")
    
    # 生成参数
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="LLM 采样温度")
    parser.add_argument("--top_p", type=float, default=0.4,
                        help="LLM top_p")
    parser.add_argument("--generate_steps", type=int, default=6,
                        help="Mask Transformer 生成步数")
    
    args = parser.parse_args()
    
    # ---- 默认路径 ----
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    module_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.mask_ckpt is None:
        args.mask_ckpt = os.path.join(project_dir, "checkpoints/mask_transformer")
    if args.rvqvae_ckpt is None:
        args.rvqvae_ckpt = os.path.join(project_dir, "checkpoints/rvqvae/model/epoch_30.pth")
    if args.output_name is None:
        args.output_name = os.path.splitext(os.path.basename(args.audio_path))[0]
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'=' * 60}")
    print(f"  SentiAvatar 单条样本推理")
    print(f"  音频: {args.audio_path}")
    print(f"  动作: {args.action_text}")
    print(f"  输出: {args.output_dir}/{args.output_name}")
    print(f"{'=' * 60}\n")
    
    # ---- Step 1: 提取音频特征 ----
    print(">>> Step 1: 提取音频特征...")
    audio_features, audio_tokens = extract_hubert_features_and_tokens(
        args.audio_path, device=args.device
    )
    
    # ---- Step 2: 检查 vLLM 服务 ----
    vllm_url = f"http://localhost:{args.vllm_port}"
    print(f"\n>>> Step 2: 检查 vLLM 服务 ({vllm_url})...")
    vllm_client = VLLMClient(vllm_url)
    if not vllm_client.health_check():
        print(f"❌ vLLM 服务不可用!")
        print(f"   请先启动: bash scripts/start_vllm_server.sh")
        sys.exit(1)
    print("  ✅ vLLM 服务正常\n")
    
    # ---- Step 3: 加载 Mask Transformer ----
    print(">>> Step 3: 加载 Mask Transformer...")
    mask_model = load_mask_transformer(args.mask_ckpt, device=args.device)
    
    # ---- Step 4: Pipeline 推理 (LLM + Mask Transformer) ----
    print("\n>>> Step 4: Pipeline 推理 (LLM + Mask Transformer)...")
    result = run_pipeline_single(
        vllm_client,
        mask_model,
        action_text=args.action_text,
        audio_tokens=audio_tokens,
        audio_features=audio_features,
        name=args.output_name,
        temperature=args.temperature,
        top_p=args.top_p,
        generate_steps=args.generate_steps,
    )
    
    if result is None:
        print("❌ Pipeline 推理失败!")
        sys.exit(1)
    
    dense_tokens = result["dense_tokens"]
    print(f"  生成 {len(dense_tokens)} 帧 dense motion tokens")
    
    # ---- Step 5: RVQVAE 解码 ----
    print("\n>>> Step 5: RVQVAE Token 解码...")
    config = load_config_from_checkpoint(args.rvqvae_ckpt)
    rvq_model = load_model(args.rvqvae_ckpt, config, device)
    
    # 加载归一化参数
    meta_dir = os.path.join(module_dir, "meta/mta_gen_demo")
    mean = torch.tensor(np.load(os.path.join(meta_dir, "mean.npy"))).to(device)
    std = torch.tensor(np.load(os.path.join(meta_dir, "std.npy"))).to(device)
    
    # 加载占位符手部数据
    placeholder_npy = os.path.join(module_dir, "meta/xiu_joint_quat_vecs/Daiji_A_001_V001.npy")
    placeholder_motion_dict = np.load(placeholder_npy, allow_pickle=True).item()
    
    # 解码
    motion = decode_body_tokens(
        rvq_model, dense_tokens, placeholder_motion_dict,
        mean, std, device, src_fps=20.0, tgt_fps=30.0,
    )
    print(f"  解码完成: offset={motion['offset'].shape}, quat={motion['quat'].shape}")
    
    # ---- Step 6: 保存输出 ----
    print(f"\n>>> Step 6: 保存输出...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    postprocesser = MotionPostprocesser()
    
    # 保存 BVH
    bvh_path = os.path.join(args.output_dir, f"{args.output_name}.bvh")
    postprocesser.save_quat_motion_to_bvh(motion=motion, save_path=bvh_path)
    print(f"  ✅ BVH → {bvh_path}")
    
    # 保存 anim.json
    json_path = os.path.join(args.output_dir, f"{args.output_name}.json")
    anim = postprocesser.convert_quat_motion_to_ue_from_bvh(motion=motion)
    with open(json_path, "w") as f:
        json.dump(anim, f, indent=2, ensure_ascii=False)
    print(f"  ✅ JSON → {json_path}")
    
    # 复制音频
    import shutil
    wav_dst = os.path.join(args.output_dir, f"{args.output_name}.wav")
    shutil.copy(args.audio_path, wav_dst)
    print(f"  ✅ WAV → {wav_dst}")
    
    print(f"\n{'=' * 60}")
    print(f"  推理完成！输出文件:")
    print(f"    - {bvh_path}")
    print(f"    - {json_path}")
    print(f"    - {wav_dst}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
