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


def resample_fps_1d(x_np, src_fps=50.0, tgt_fps=10.0):
    """
    对 (T, D) 的特征序列进行帧率重采样（使用线性插值）。
    
    Args:
        x_np: numpy array, shape (T, D)
        src_fps: 原始帧率
        tgt_fps: 目标帧率
    
    Returns:
        numpy array, shape (T_new, D)
    """
    import torch.nn.functional as F
    x = torch.tensor(x_np, dtype=torch.float32)
    # reshape to (1, T, D, 1) for resample_fps compatible format
    x = x[None, :, :, None]  # (1, T, D, 1)
    B, T, J, D = x.shape
    new_T = max(1, int(round(T * (tgt_fps / src_fps))))
    y = x.permute(0, 2, 3, 1).contiguous().view(B, J * D, T)
    y2 = F.interpolate(y, size=new_T, mode="linear", align_corners=False)
    out = y2.view(B, J, D, new_T).permute(0, 3, 1, 2).contiguous()
    out = out.squeeze(0).squeeze(-1)  # (T_new, D)
    return out.numpy()


class ApplyKmeans(object):
    """HuBERT K-means 量化器，将连续特征映射为离散 token"""
    def __init__(self, km_path):
        import joblib
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)
        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)

    def to(self, device):
        if device == "cpu":
            self.C = self.C.cpu()
            self.Cnorm = self.Cnorm.cpu()
        elif torch.cuda.is_available():
            self.C = self.C.to(device)
            self.Cnorm = self.Cnorm.to(device)
        return self

    def feat2token(self, audio_feats):
        """将音频特征量化为 token 列表"""
        quantized_indices = self.__call__(audio_feats)
        return quantized_indices.tolist()


def extract_hubert_features_and_tokens(audio_path, device="cuda"):
    """
    从 wav 文件提取 HuBERT 特征和量化 tokens。
    
    完整流程：
    1. 加载 Chinese HuBERT 模型
    2. 提取 layer 9 特征 (hidden_states[8])，HuBERT 原始输出约 50fps
    3. 下采样 50fps → 10fps（线性插值）
    4. 使用 K-means 模型将 layer9 特征量化为离散 tokens
    
    Args:
        audio_path: wav 文件路径
        device: 计算设备
    
    Returns:
        audio_features: (T, 768) numpy array, HuBERT layer9 特征 @10fps
        audio_tokens: list of int, K-means 量化后的 audio tokens @10fps
    """
    from transformers import Wav2Vec2FeatureExtractor, HubertModel
    
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hubert_path = os.path.join(project_dir, "checkpoints", "chinese-hubert-base")
    kmeans_path = os.path.join(project_dir, "checkpoints", "hubert_kmeans", "model.mdl")
    
    # ---- 1. 加载模型 ----
    print(f"[Audio] 加载 Chinese HuBERT: {hubert_path}")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_path)
    hubert_model = HubertModel.from_pretrained(hubert_path).to(device).eval()
    
    print(f"[Audio] 加载 K-means 量化器: {kmeans_path}")
    kmeans = ApplyKmeans(kmeans_path).to(device)
    
    # ---- 2. 读取音频 ----
    wav, sr = sf.read(audio_path)
    if len(wav.shape) > 1:
        wav = wav[:, 0]  # 多声道取第一个
    # HuBERT 需要 16kHz
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    
    # ---- 3. 提取 HuBERT 特征 ----
    input_values = feature_extractor(
        wav, return_tensors="pt", sampling_rate=16000
    ).input_values.to(device)
    
    with torch.no_grad():
        outputs = hubert_model(input_values, output_hidden_states=True)
        # Layer 9 特征 (index 8 in hidden_states, 0-indexed after embedding layer)
        audio_9layer = outputs.hidden_states[8].squeeze(0).cpu().numpy()  # (T_50fps, 768)
    
    print(f"[Audio] HuBERT layer9 原始特征: shape={audio_9layer.shape} (~50fps)")
    
    # ---- 4. 下采样 50fps → 10fps ----
    audio_features_10fps = resample_fps_1d(audio_9layer, src_fps=50.0, tgt_fps=10.0)
    print(f"[Audio] 下采样后特征: shape={audio_features_10fps.shape} (10fps)")
    
    # ---- 5. K-means 量化为 tokens ----
    feats_tensor = torch.tensor(audio_features_10fps, dtype=torch.float32).to(device)
    audio_tokens = kmeans.feat2token(feats_tensor)
    
    print(f"[Audio] 特征提取完成: features={audio_features_10fps.shape}, tokens={len(audio_tokens)}")
    
    # 清理显存
    del hubert_model, feature_extractor
    torch.cuda.empty_cache()
    
    return audio_features_10fps.astype(np.float32), audio_tokens


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
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="LLM 采样温度")
    parser.add_argument("--top_p", type=float, default=0.2,
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
