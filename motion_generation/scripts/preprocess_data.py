#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
数据预处理脚本

将原始数据集(wav_data, motion_data)预处理为推理所需的中间数据：
1. audio_features_hubert_layer9_fps10: HuBERT layer9 特征 @10fps
2. audio_tokens_hubert_layer9_fps10:   K-means 量化后的音频 token @10fps
3. motion_token_data:                   RVQVAE 编码后的动作 token

前置条件：
    - 模型权重已放置到 checkpoints/ 目录
    - 数据集已放置到 data/ 目录（需要 wav_data 和 motion_data）

用法:
    # 运行全部预处理
    python scripts/preprocess_data.py --all

    # 仅处理音频特征+token
    python scripts/preprocess_data.py --audio

    # 仅处理动作token
    python scripts/preprocess_data.py --motion

@Author  :   Chuhao Jin
"""

import os
import sys
import json
import argparse
import glob
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from tqdm import tqdm

# 确保能导入本地模块
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "motion_generation"))


# ======================================================================
#  Part 1: HuBERT 特征提取 + 下采样 + K-means 量化
# ======================================================================

def resample_fps_1d(x_np, src_fps=50.0, tgt_fps=10.0):
    """对 (T, D) 的特征序列进行帧率重采样"""
    x = torch.tensor(x_np, dtype=torch.float32)
    x = x[None, :, :, None]  # (1, T, D, 1)
    B, T, J, D = x.shape
    new_T = max(1, int(round(T * (tgt_fps / src_fps))))
    y = x.permute(0, 2, 3, 1).contiguous().view(B, J * D, T)
    y2 = F.interpolate(y, size=new_T, mode="linear", align_corners=False)
    out = y2.view(B, J, D, new_T).permute(0, 3, 1, 2).contiguous()
    return out.squeeze(0).squeeze(-1).numpy()


class ApplyKmeans(object):
    """HuBERT K-means 量化器"""
    def __init__(self, km_path):
        import joblib
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)
        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)

    def to(self, device):
        self.C = self.C.to(device)
        self.Cnorm = self.Cnorm.to(device)
        return self

    def feat2token(self, x):
        if isinstance(x, torch.Tensor):
            dist = x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
            return dist.argmin(dim=1).cpu().numpy().tolist()
        else:
            dist = (x ** 2).sum(1, keepdims=True) - 2 * np.matmul(x, self.C_np) + self.Cnorm_np
            return np.argmin(dist, axis=1).tolist()


def process_audio(args):
    """处理音频: 提取 HuBERT 特征 → 下采样 → K-means 量化"""
    from transformers import Wav2Vec2FeatureExtractor, HubertModel

    device = torch.device(args.device)
    hubert_path = os.path.join(PROJECT_DIR, "checkpoints", "chinese-hubert-base")
    kmeans_path = os.path.join(PROJECT_DIR, "checkpoints", "hubert_kmeans", "model.mdl")

    wav_dir = os.path.join(args.data_dir, "wav_data")
    feat_output_dir = os.path.join(args.data_dir, "audio_features_hubert_layer9_fps10")
    token_output_dir = os.path.join(args.data_dir, "audio_tokens_hubert_layer9_fps10")

    # 加载模型
    print(f"[Audio] 加载 Chinese HuBERT: {hubert_path}")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_path)
    audio_encoder = HubertModel.from_pretrained(hubert_path).to(device).eval()

    print(f"[Audio] 加载 K-means 量化器: {kmeans_path}")
    kmeans = ApplyKmeans(kmeans_path).to(device)

    # 读取文件列表
    split_file = os.path.join(args.data_dir, "split", "all_file_list.txt")
    with open(split_file, "r") as f:
        name_list = [line.strip() for line in f if line.strip()]

    print(f"[Audio] 共 {len(name_list)} 个文件待处理")

    success, skip, fail = 0, 0, 0
    for name in tqdm(name_list, desc="处理音频特征"):
        wav_path = os.path.join(wav_dir, f"{name}.wav")
        feat_path = os.path.join(feat_output_dir, f"{name}.npy")
        token_path = os.path.join(token_output_dir, f"{name}.json")

        # 跳过已存在
        if os.path.exists(feat_path) and os.path.exists(token_path):
            skip += 1
            continue

        if not os.path.exists(wav_path):
            fail += 1
            continue

        try:
            # 1. 读取音频
            wav, sr = sf.read(wav_path)
            if len(wav.shape) > 1:
                wav = wav[:, 0]
            if sr != 16000:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)

            # 2. 提取 HuBERT layer9 特征
            input_values = feature_extractor(
                wav, return_tensors="pt", sampling_rate=16000
            ).input_values.to(device)

            with torch.no_grad():
                outputs = audio_encoder(input_values, output_hidden_states=True)
                audio_9layer = outputs.hidden_states[8].squeeze(0).cpu().numpy()  # (T_50fps, 768)

            # 3. 下采样 50fps → 10fps
            audio_feat_10fps = resample_fps_1d(audio_9layer, src_fps=50.0, tgt_fps=10.0)

            # 4. 保存特征
            os.makedirs(os.path.dirname(feat_path), exist_ok=True)
            np.save(feat_path, audio_feat_10fps.astype(np.float32))

            # 5. K-means 量化
            feats_tensor = torch.tensor(audio_feat_10fps, dtype=torch.float32).to(device)
            tokens = kmeans.feat2token(feats_tensor)

            # 6. 保存 token
            os.makedirs(os.path.dirname(token_path), exist_ok=True)
            with open(token_path, "w") as f:
                json.dump({
                    "fps": 10,
                    "num_tokens": len(tokens),
                    "tokens": tokens,
                    "name": name,
                }, f, indent=2, ensure_ascii=False)

            success += 1
        except Exception as e:
            fail += 1
            print(f"\n[ERROR] {name}: {e}")

    print(f"\n[Audio] 完成! 成功: {success}, 跳过: {skip}, 失败: {fail}")


# ======================================================================
#  Part 2: Motion Token 编码
# ======================================================================

def process_motion(args):
    """处理动作: RVQVAE 编码 motion → tokens"""
    from infer import load_config_from_checkpoint, load_model, fixseed
    from models.rvqvae import RVQVAE
    from actions.schema import MotionTokens

    device = torch.device(args.device)
    rvqvae_ckpt = os.path.join(PROJECT_DIR, "checkpoints", "rvqvae", "model", "epoch_30.pth")
    motion_dir = os.path.join(args.data_dir, "motion_data")
    token_output_dir = os.path.join(args.data_dir, "motion_token_data")
    module_dir = os.path.join(PROJECT_DIR, "motion_generation")

    # 加载模型
    print(f"[Motion] 加载 RVQVAE: {rvqvae_ckpt}")
    config = load_config_from_checkpoint(rvqvae_ckpt)
    model = load_model(rvqvae_ckpt, config, device)

    # 加载归一化参数
    mean = torch.tensor(np.load(os.path.join(module_dir, "meta/mta_gen_demo/mean.npy"))).to(device)
    std = torch.tensor(np.load(os.path.join(module_dir, "meta/mta_gen_demo/std.npy"))).to(device)

    # 读取文件列表
    split_file = os.path.join(args.data_dir, "split", "all_file_list.txt")
    with open(split_file, "r") as f:
        name_list = [line.strip() for line in f if line.strip()]

    print(f"[Motion] 共 {len(name_list)} 个文件待处理")

    success, skip, fail = 0, 0, 0
    for name in tqdm(name_list, desc="编码动作 token"):
        motion_path = os.path.join(motion_dir, f"{name}.npy")
        token_path = os.path.join(token_output_dir, f"{name}.json")

        if os.path.exists(token_path):
            skip += 1
            continue

        if not os.path.exists(motion_path):
            fail += 1
            continue

        try:
            # 1. 加载动作数据
            motion_dict = np.load(motion_path, allow_pickle=True)
            if isinstance(motion_dict, np.ndarray) and motion_dict.dtype == object:
                motion_dict = motion_dict.item()
            else:
                fail += 1
                continue

            # 2. 预处理 body motion (与 encode_motion 一致)
            body_motion = torch.tensor(motion_dict["body"], dtype=torch.float32).to(device)
            body_motion[:, 2] = body_motion[:, 2] - body_motion[0, 2]
            body_motion[1:, :3] = body_motion[1:, :3] - body_motion[:-1, :3]
            body_motion = (body_motion - mean) / std
            body_motion = body_motion.unsqueeze(0)

            # 3. RVQVAE 编码
            with torch.no_grad():
                output = model.encode(body_motion)

            body_tokens = output["code_idx"]["body"].squeeze(0).cpu().numpy().tolist()

            # 4. 保存 token
            os.makedirs(os.path.dirname(token_path), exist_ok=True)
            with open(token_path, "w") as f:
                json.dump({
                    "fps": 20,
                    "num_tokens": len(body_tokens),
                    "tokens": body_tokens,
                    "name": name,
                }, f, indent=2, ensure_ascii=False)

            success += 1
        except Exception as e:
            fail += 1
            print(f"\n[ERROR] {name}: {e}")

    print(f"\n[Motion] 完成! 成功: {success}, 跳过: {skip}, 失败: {fail}")


# ======================================================================
#  Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="数据预处理：从原始数据生成推理所需的中间数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行全部预处理
  python scripts/preprocess_data.py --all

  # 仅处理音频特征+token
  python scripts/preprocess_data.py --audio

  # 仅处理动作token
  python scripts/preprocess_data.py --motion

  # 指定数据目录和设备
  python scripts/preprocess_data.py --all --data_dir ./data --device cuda:0
        """,
    )

    parser.add_argument("--all", action="store_true", help="运行全部预处理")
    parser.add_argument("--audio", action="store_true", help="仅处理音频(HuBERT特征+K-means token)")
    parser.add_argument("--motion", action="store_true", help="仅处理动作(RVQVAE编码)")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(PROJECT_DIR, "data"),
                        help="数据集根目录 (默认: ./data)")
    parser.add_argument("--device", type=str, default="cuda:0", help="推理设备")

    args = parser.parse_args()

    if not (args.all or args.audio or args.motion):
        parser.print_help()
        print("\n请指定 --all, --audio 或 --motion")
        return

    print(f"{'=' * 60}")
    print(f"  数据预处理")
    print(f"  数据目录: {args.data_dir}")
    print(f"  设备:     {args.device}")
    print(f"{'=' * 60}\n")

    if args.all or args.audio:
        process_audio(args)

    if args.all or args.motion:
        process_motion(args)

    print(f"\n{'=' * 60}")
    print(f"  预处理完成! 生成的中间数据:")
    print(f"    - {args.data_dir}/audio_features_hubert_layer9_fps10/")
    print(f"    - {args.data_dir}/audio_tokens_hubert_layer9_fps10/")
    print(f"    - {args.data_dir}/motion_token_data/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
