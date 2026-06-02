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
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# 确保能导入本地模块
MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(MODULE_DIR)
sys.path.insert(0, MODULE_DIR)


def resolve_torch_device(device_arg: str) -> torch.device:
    """Use the requested CUDA device when available, otherwise keep CPU smoke runs usable."""
    requested = str(device_arg)
    if "cuda" in requested.lower() and not torch.cuda.is_available():
        print(f"[Device] WARN: requested {requested}, but CUDA is unavailable; falling back to cpu")
        return torch.device("cpu")
    return torch.device(requested)


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


def resample_sequence_length(x_np: np.ndarray, target_len: int) -> np.ndarray:
    """Resample a (T,D) sequence to an explicit length for token-rate alignment."""
    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")
    x_np = np.asarray(x_np, dtype=np.float32)
    if x_np.ndim != 2:
        raise ValueError(f"Expected a 2D sequence, got {x_np.shape}")
    if x_np.shape[0] == target_len:
        return x_np
    x = torch.tensor(x_np, dtype=torch.float32)
    y = x.transpose(0, 1).unsqueeze(0)
    y = F.interpolate(y, size=target_len, mode="linear", align_corners=False)
    return y.squeeze(0).transpose(0, 1).numpy()


def encoded_motion_token_length(num_motion_frames: int, stride_t: int, down_t: int) -> int:
    """Match the RVQVAE temporal encoder length for Conv1d stride downsampling."""
    length = int(num_motion_frames)
    stride_t = int(stride_t)
    kernel_size = stride_t * 2
    padding = stride_t // 2
    for _ in range(int(down_t)):
        length = (length + 2 * padding - kernel_size) // stride_t + 1
    return max(1, length)


def load_motion_frame_count(path: str) -> int:
    motion = np.load(path, allow_pickle=True)
    if isinstance(motion, np.ndarray) and motion.dtype == object:
        motion = motion.item()
    if not isinstance(motion, dict) or "body" not in motion:
        raise ValueError(f"Expected motion dict with body key: {path}")
    body = np.asarray(motion["body"])
    if body.ndim != 2:
        raise ValueError(f"Expected body motion shape (T,D), got {body.shape}: {path}")
    return int(body.shape[0])


def _normalize_rel_name(path: str, root: str, suffix: str) -> str:
    rel = os.path.relpath(path, root)
    if not rel.endswith(suffix):
        raise ValueError(f"Expected suffix {suffix}: {path}")
    return rel[: -len(suffix)].replace(os.sep, "/")


def _read_split_file(path: str) -> list[str]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _dedupe(names: list[str]) -> list[str]:
    seen = set()
    out = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _discover_names_from_dir(root: str, suffix: str) -> list[str]:
    if not os.path.isdir(root):
        return []
    names = []
    for current_root, _, files in os.walk(root):
        for file_name in files:
            if file_name.endswith(suffix):
                names.append(_normalize_rel_name(os.path.join(current_root, file_name), root, suffix))
    return sorted(names)


def resolve_preprocess_names(args, required_subdir: str, suffix: str, label: str) -> list[str]:
    """Resolve sample names for preprocessing without requiring all_file_list.txt."""
    split_dir = os.path.join(args.data_dir, "split")
    candidate_splits = []
    if args.split_file:
        if not os.path.exists(args.split_file):
            raise FileNotFoundError(f"--split_file not found: {args.split_file}")
        candidate_splits.append(args.split_file)
    else:
        candidate_splits.append(os.path.join(split_dir, "all_file_list.txt"))
        candidate_splits.extend(
            os.path.join(split_dir, file_name)
            for file_name in ("train_file_list.txt", "val_file_list.txt", "test_file_list.txt")
        )

    names: list[str] = []
    used_splits: list[str] = []
    for split_file in candidate_splits:
        split_names = _read_split_file(split_file)
        if split_names:
            names.extend(split_names)
            used_splits.append(split_file)
            if args.split_file or os.path.basename(split_file) == "all_file_list.txt":
                break

    names = _dedupe(names)
    if names:
        print(f"[{label}] 使用 split 文件解析样本: {', '.join(used_splits)}")
        return names

    data_root = os.path.join(args.data_dir, required_subdir)
    names = _discover_names_from_dir(data_root, suffix)
    if names:
        print(f"[{label}] 未找到 split 文件，改为从 {data_root} 发现 {len(names)} 个样本")
        return names

    raise FileNotFoundError(
        f"No samples found for {label}. Expected --split_file, split/all_file_list.txt, "
        f"train/val/test splits, or files under {data_root}"
    )


def resample_audio_to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return wav
    try:
        import librosa

        return librosa.resample(wav, orig_sr=sr, target_sr=16000)
    except ImportError:
        from scipy.signal import resample_poly

        gcd = int(np.gcd(sr, 16000))
        return resample_poly(wav, 16000 // gcd, sr // gcd).astype(np.float32)


def load_wav_mono(path: str) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        wav, sr = sf.read(path)
    except ImportError:
        from scipy.io import wavfile

        sr, wav = wavfile.read(path)
        wav = np.asarray(wav)
        if np.issubdtype(wav.dtype, np.integer):
            max_abs = max(abs(np.iinfo(wav.dtype).min), np.iinfo(wav.dtype).max)
            wav = wav.astype(np.float32) / float(max_abs)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav[:, 0]
    return wav, int(sr)


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
    from infer import load_config_from_checkpoint

    device = resolve_torch_device(args.device)
    hubert_path = args.hubert_path
    kmeans_path = args.kmeans_path

    wav_dir = os.path.join(args.data_dir, "wav_data")
    motion_dir = os.path.join(args.data_dir, "motion_data")
    feat_output_dir = os.path.join(args.data_dir, "audio_features_hubert_layer9_fps10")
    token_output_dir = os.path.join(args.data_dir, "audio_tokens_hubert_layer9_fps10")

    align_config = None
    downsample_factor = None
    if args.align_audio_to_motion:
        if os.path.exists(args.rvqvae_ckpt):
            align_config = load_config_from_checkpoint(args.rvqvae_ckpt)
            downsample_factor = int(align_config.model.stride_t) ** int(align_config.model.down_t)
            token_fps = float(align_config.data.fps) / max(1, downsample_factor)
            if abs(token_fps - 10.0) > 1e-3:
                print(
                    f"[Audio] WARN: RVQVAE token_fps={token_fps:.4f}; output directory name still uses fps10 "
                    "for GitHub compatibility."
                )
            print(
                f"[Audio] 将 HuBERT 特征按 motion token 长度对齐: "
                f"stride_t={align_config.model.stride_t}, down_t={align_config.model.down_t}, "
                f"downsample_factor={downsample_factor}"
            )
        else:
            print(
                f"[Audio] WARN: RVQVAE checkpoint not found ({args.rvqvae_ckpt}); "
                "audio-to-motion length alignment is disabled for this run."
            )

    # 加载模型
    print(f"[Audio] 加载 Chinese HuBERT: {hubert_path}")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_path)
    audio_encoder = HubertModel.from_pretrained(hubert_path).to(device).eval()

    print(f"[Audio] 加载 K-means 量化器: {kmeans_path}")
    kmeans = ApplyKmeans(kmeans_path).to(device)

    # 读取文件列表
    name_list = resolve_preprocess_names(args, "wav_data", ".wav", "Audio")

    print(f"[Audio] 共 {len(name_list)} 个文件待处理")

    success, skip, fail = 0, 0, 0
    for name in tqdm(name_list, desc="处理音频特征"):
        wav_path = os.path.join(wav_dir, f"{name}.wav")
        feat_path = os.path.join(feat_output_dir, f"{name}.npy")
        token_path = os.path.join(token_output_dir, f"{name}.json")

        # 跳过已存在
        if not args.overwrite and os.path.exists(feat_path) and os.path.exists(token_path):
            skip += 1
            continue

        if not os.path.exists(wav_path):
            fail += 1
            continue

        try:
            # 1. 读取音频
            wav, sr = load_wav_mono(wav_path)
            wav = resample_audio_to_16k(wav, sr).astype(np.float32)

            # 2. 提取 HuBERT layer9 特征
            input_values = feature_extractor(
                wav, return_tensors="pt", sampling_rate=16000
            ).input_values.to(device)

            with torch.no_grad():
                outputs = audio_encoder(input_values, output_hidden_states=True)
                audio_9layer = outputs.hidden_states[8].squeeze(0).cpu().numpy()  # (T_50fps, 768)

            # 3. 下采样 50fps → 10fps
            audio_feat_10fps = resample_fps_1d(audio_9layer, src_fps=50.0, tgt_fps=10.0)
            source_audio_frames = int(audio_feat_10fps.shape[0])
            aligned_to_motion = False
            target_frames = None
            motion_path = os.path.join(motion_dir, f"{name}.npy")
            if align_config is not None and os.path.exists(motion_path):
                motion_frames = load_motion_frame_count(motion_path)
                target_frames = encoded_motion_token_length(
                    motion_frames,
                    stride_t=align_config.model.stride_t,
                    down_t=align_config.model.down_t,
                )
                if target_frames != source_audio_frames:
                    audio_feat_10fps = resample_sequence_length(audio_feat_10fps, target_frames)
                    aligned_to_motion = True

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
                    "source_audio_feature_frames": source_audio_frames,
                    "aligned_to_motion_tokens": aligned_to_motion,
                    "target_motion_token_frames": target_frames,
                }, f, indent=2, ensure_ascii=False)

            success += 1
        except Exception as e:
            fail += 1
            print(f"\n[ERROR] {name}: {e}")

    print(f"\n[Audio] 完成! 成功: {success}, 跳过: {skip}, 失败: {fail}")
    return success, skip, fail


# ======================================================================
#  Part 2: Motion Token 编码
# ======================================================================

def process_motion(args):
    """处理动作: RVQVAE 编码 motion → tokens"""
    from infer import load_config_from_checkpoint, load_model, fixseed, load_motion_stats
    from models.rvqvae import RVQVAE
    from actions.schema import MotionTokens

    device = resolve_torch_device(args.device)
    rvqvae_ckpt = args.rvqvae_ckpt
    motion_dir = os.path.join(args.data_dir, "motion_data")
    token_output_dir = os.path.join(args.data_dir, "motion_token_data")

    # 加载模型
    print(f"[Motion] 加载 RVQVAE: {rvqvae_ckpt}")
    config = load_config_from_checkpoint(rvqvae_ckpt)
    model = load_model(rvqvae_ckpt, config, device)

    # 加载归一化参数：优先使用 RVQVAE checkpoint 旁边的 meta/mean.npy 和 meta/std.npy
    mean, std = load_motion_stats(rvqvae_ckpt, device)

    # 读取文件列表
    name_list = resolve_preprocess_names(args, "motion_data", ".npy", "Motion")

    print(f"[Motion] 共 {len(name_list)} 个文件待处理")

    success, skip, fail = 0, 0, 0
    for name in tqdm(name_list, desc="编码动作 token"):
        motion_path = os.path.join(motion_dir, f"{name}.npy")
        token_path = os.path.join(token_output_dir, f"{name}.json")

        if not args.overwrite and os.path.exists(token_path):
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

            downsample_factor = int(config.model.stride_t) ** int(config.model.down_t)
            token_fps = float(config.data.fps) / max(1, downsample_factor)

            # 4. 保存 token
            os.makedirs(os.path.dirname(token_path), exist_ok=True)
            with open(token_path, "w") as f:
                json.dump({
                    "source_fps": config.data.fps,
                    "token_fps": token_fps,
                    "downsample_factor": downsample_factor,
                    "num_tokens": len(body_tokens),
                    "tokens": body_tokens,
                    "name": name,
                }, f, indent=2, ensure_ascii=False)

            success += 1
        except Exception as e:
            fail += 1
            print(f"\n[ERROR] {name}: {e}")

    print(f"\n[Motion] 完成! 成功: {success}, 跳过: {skip}, 失败: {fail}")
    return success, skip, fail


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
    parser.add_argument("--split_file", type=str, default=None,
                        help="Optional sample list. Defaults to split/all_file_list.txt, train/val/test splits, or directory discovery")
    parser.add_argument("--device", type=str, default="cuda:0", help="推理设备")
    parser.add_argument("--rvqvae_ckpt", type=str,
                        default=os.path.join(PROJECT_DIR, "checkpoints", "rvqvae", "model", "epoch_30.pth"),
                        help="RVQVAE checkpoint 路径 (默认: ./checkpoints/rvqvae/model/epoch_30.pth)")
    parser.add_argument("--hubert_path", type=str,
                        default=os.path.join(PROJECT_DIR, "checkpoints", "chinese-hubert-base"),
                        help="Chinese HuBERT 模型路径")
    parser.add_argument("--kmeans_path", type=str,
                        default=os.path.join(PROJECT_DIR, "checkpoints", "hubert_kmeans", "model.mdl"),
                        help="HuBERT K-means 量化器路径")
    parser.add_argument("--strict", action="store_true",
                        help="If any requested preprocessing item fails, exit with a non-zero status")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate outputs even when preprocessed files already exist")
    parser.add_argument("--no_align_audio_to_motion", action="store_true",
                        help="Do not resample HuBERT features to the RVQVAE motion-token length when motion_data is available")

    args = parser.parse_args()
    args.align_audio_to_motion = not args.no_align_audio_to_motion

    if not (args.all or args.audio or args.motion):
        parser.print_help()
        print("\n请指定 --all, --audio 或 --motion")
        return

    print(f"{'=' * 60}")
    print(f"  数据预处理")
    print(f"  数据目录: {args.data_dir}")
    print(f"  设备:     {args.device}")
    print(f"{'=' * 60}\n")

    total_fail = 0
    if args.all or args.audio:
        _, _, fail = process_audio(args)
        total_fail += fail

    if args.all or args.motion:
        _, _, fail = process_motion(args)
        total_fail += fail

    print(f"\n{'=' * 60}")
    print(f"  预处理完成! 生成的中间数据:")
    print(f"    - {args.data_dir}/audio_features_hubert_layer9_fps10/")
    print(f"    - {args.data_dir}/audio_tokens_hubert_layer9_fps10/")
    print(f"    - {args.data_dir}/motion_token_data/")
    print(f"{'=' * 60}")

    if args.strict and total_fail > 0:
        raise SystemExit(f"Preprocessing failed for {total_fail} requested item(s)")


if __name__ == "__main__":
    main()
