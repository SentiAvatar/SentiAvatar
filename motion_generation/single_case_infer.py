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
import math
import wave
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_infer import (
    VLLMClient, load_mask_transformer,
    run_pipeline_single,
)
from reconstruct_from_tokens import decode_body_tokens, validate_placeholder_motion, validate_token_frames
from infer import load_config_from_checkpoint, load_model, fixseed
from infer import load_motion_stats
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


def read_audio_mono(audio_path):
    """Read a wav/audio file as mono float32 without making soundfile a module import dependency."""
    try:
        import soundfile as sf

        wav, sr = sf.read(audio_path)
        if len(wav.shape) > 1:
            wav = wav[:, 0]
        return wav.astype(np.float32), int(sr)
    except ImportError:
        pass

    with wave.open(audio_path, "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sampwidth == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width {sampwidth} bytes: {audio_path}")
    if channels > 1:
        audio = audio.reshape(-1, channels)[:, 0]
    return audio.astype(np.float32), int(sr)


def resample_audio_16k(wav, sr):
    if sr == 16000:
        return wav.astype(np.float32)
    try:
        import librosa

        return librosa.resample(wav, orig_sr=sr, target_sr=16000).astype(np.float32)
    except ImportError:
        pass
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise ImportError(
            f"Audio sample rate is {sr}; install librosa or scipy for 16kHz resampling"
        ) from exc
    gcd = math.gcd(int(sr), 16000)
    return resample_poly(wav, 16000 // gcd, int(sr) // gcd).astype(np.float32)


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


def extract_hubert_features_and_tokens(audio_path, device="cuda", hubert_path=None, kmeans_path=None):
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
    if hubert_path is None:
        hubert_path = os.path.join(project_dir, "checkpoints", "chinese-hubert-base")
    if kmeans_path is None:
        kmeans_path = os.path.join(project_dir, "checkpoints", "hubert_kmeans", "model.mdl")
    if not os.path.isdir(hubert_path):
        raise FileNotFoundError(f"Chinese HuBERT directory not found: {hubert_path}")
    if not os.path.isfile(kmeans_path):
        raise FileNotFoundError(f"HuBERT K-means model not found: {kmeans_path}")
    
    # ---- 1. 加载模型 ----
    print(f"[Audio] 加载 Chinese HuBERT: {hubert_path}")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_path)
    hubert_model = HubertModel.from_pretrained(hubert_path).to(device).eval()
    
    print(f"[Audio] 加载 K-means 量化器: {kmeans_path}")
    kmeans = ApplyKmeans(kmeans_path).to(device)
    
    # ---- 2. 读取音频 ----
    wav, sr = read_audio_mono(audio_path)
    # HuBERT 需要 16kHz
    if sr != 16000:
        wav = resample_audio_16k(wav, sr)
    
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


def validate_body_codec_compatibility(mask_model, rvqvae_config) -> None:
    """Ensure body Infill Transformer emits tokens decodable by the RVQVAE."""
    mask_quantizers = int(mask_model.config.num_tokens_per_frame)
    mask_codebook = int(mask_model.config.codebook_size)
    rvq_quantizers = int(rvqvae_config.model.num_quantizers)
    rvq_codebook = int(rvqvae_config.model.nb_code)
    if mask_quantizers != rvq_quantizers:
        raise ValueError(
            "Mask/RVQVAE quantizer mismatch: "
            f"mask num_tokens_per_frame={mask_quantizers}, RVQVAE num_quantizers={rvq_quantizers}"
        )
    if mask_codebook != rvq_codebook:
        raise ValueError(
            "Mask/RVQVAE codebook mismatch: "
            f"mask codebook_size={mask_codebook}, RVQVAE nb_code={rvq_codebook}"
        )
    expected_vocab = mask_codebook * mask_quantizers + 1
    if int(mask_model.config.vocab_size) != expected_vocab:
        raise ValueError(
            "Mask Transformer vocab_size does not match codebook contract: "
            f"{mask_model.config.vocab_size} vs expected {expected_vocab}"
        )


def load_continuation_inputs(args):
    """Load previous-turn artifacts produced by this script or pipeline_infer.py."""
    has_any = bool(args.prefix_audio_token_json or args.prefix_audio_feat_npy or args.prefix_motion_token_json)
    if not has_any:
        return None, None, None
    if not (args.prefix_audio_token_json and args.prefix_motion_token_json):
        raise ValueError(
            "Continuation mode requires --prefix_audio_token_json and --prefix_motion_token_json"
        )

    with open(args.prefix_audio_token_json, "r", encoding="utf-8") as f:
        prefix_audio_tokens = json.load(f).get("tokens", [])
    with open(args.prefix_motion_token_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    prefix_motion_tokens = payload.get("dense_tokens") or payload.get("tokens", [])
    if not prefix_audio_tokens:
        raise ValueError(f"Previous-turn audio tokens are empty: {args.prefix_audio_token_json}")
    if not prefix_motion_tokens:
        raise ValueError(f"Previous-turn motion tokens are empty: {args.prefix_motion_token_json}")
    prefix_audio_features = None
    if args.prefix_audio_feat_npy:
        prefix_audio_features = np.load(args.prefix_audio_feat_npy).astype(np.float32)
        if prefix_audio_features.ndim != 2:
            raise ValueError(
                f"Previous-turn audio features must be 2D, got {prefix_audio_features.shape}: "
                f"{args.prefix_audio_feat_npy}"
            )
    return prefix_audio_tokens, prefix_audio_features, prefix_motion_tokens


def save_single_case_artifacts(
    output_dir,
    output_name,
    result,
    audio_tokens,
    audio_features,
    action_text,
    audio_path,
    rvqvae_config,
):
    """Save files that make a single-case result reusable for reconstruction or continuation."""
    token_meta = {
        "name": output_name,
        "token_fps": 10.0,
        "source_audio": audio_path,
    }
    audio_token_path = os.path.join(output_dir, f"{output_name}_audio_tokens.json")
    with open(audio_token_path, "w", encoding="utf-8") as f:
        json.dump({**token_meta, "tokens": audio_tokens}, f, indent=2, ensure_ascii=False)

    audio_feat_path = os.path.join(output_dir, f"{output_name}_audio_features.npy")
    np.save(audio_feat_path, np.asarray(audio_features, dtype=np.float32))

    motion_token_path = os.path.join(output_dir, f"{output_name}_motion_tokens.json")
    motion_payload = {
        **token_meta,
        "action_text": action_text,
        "num_quantizers": int(rvqvae_config.model.num_quantizers),
        "codebook_size": int(rvqvae_config.model.nb_code),
        "tokens": result["dense_tokens"],
        "dense_tokens": result["dense_tokens"],
    }
    with open(motion_token_path, "w", encoding="utf-8") as f:
        json.dump(motion_payload, f, indent=2, ensure_ascii=False)

    pipeline_path = os.path.join(output_dir, f"{output_name}_pipeline_result.json")
    pipeline_payload = {
        **result,
        "audio_token_path": audio_token_path,
        "audio_feature_path": audio_feat_path,
        "motion_token_path": motion_token_path,
    }
    with open(pipeline_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_payload, f, indent=2, ensure_ascii=False)
    return audio_token_path, audio_feat_path, motion_token_path, pipeline_path


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
    parser.add_argument("--hubert_path", type=str, default=None,
                        help="Chinese HuBERT directory for audio feature extraction")
    parser.add_argument("--kmeans_path", type=str, default=None,
                        help="HuBERT layer9 K-means model path")
    parser.add_argument("--face_mode", type=str, default="none",
                        choices=["none", "infill"],
                        help="Face generation mode for JSON output")
    parser.add_argument("--face_rvqvae_ckpt", type=str, default=None,
                        help="Face R-VQVAE checkpoint for --face_mode infill")
    parser.add_argument("--face_infill_ckpt", type=str, default=None,
                        help="Face Infill Transformer checkpoint for --face_mode infill")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="推理设备")
    
    # 生成参数
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="LLM 采样温度")
    parser.add_argument("--top_p", type=float, default=0.2,
                        help="LLM top_p")
    parser.add_argument("--llm_max_tokens", type=int, default=1024,
                        help="vLLM 最大生成 token 数")
    parser.add_argument("--keyframe_len_policy", type=str, default="strict",
                        choices=["strict", "warn"],
                        help="LLM keyframe 数与 prompt 采样数不一致时的处理策略")
    parser.add_argument("--generate_steps", type=int, default=6,
                        help="Mask Transformer 生成步数")
    parser.add_argument("--step", type=int, default=None,
                        help="LLM keyframe interval; default is mask_model.config.num_frames - 1")
    parser.add_argument("--face_generate_steps", type=int, default=6,
                        help="Face Infill Transformer 生成步数")
    parser.add_argument("--prefix_audio_token_json", type=str, default=None,
                        help="Previous-turn audio token JSON for continuation prompt")
    parser.add_argument("--prefix_audio_feat_npy", type=str, default=None,
                        help="Previous-turn HuBERT feature NPY for continuation infill boundary")
    parser.add_argument("--prefix_motion_token_json", type=str, default=None,
                        help="Previous-turn dense/motion token JSON for continuation prompt")
    parser.add_argument("--continuation_prefix_keyframes", type=int, default=0,
                        help="Number of previous sparse keyframes to prepend to the LLM prompt")
    
    args = parser.parse_args()
    
    # ---- 默认路径 ----
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    module_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.mask_ckpt is None:
        args.mask_ckpt = os.path.join(project_dir, "checkpoints/mask_transformer")
    if args.rvqvae_ckpt is None:
        args.rvqvae_ckpt = os.path.join(project_dir, "checkpoints/rvqvae/model/epoch_30.pth")
    if args.hubert_path is None:
        args.hubert_path = os.path.join(project_dir, "checkpoints/chinese-hubert-base")
    if args.kmeans_path is None:
        args.kmeans_path = os.path.join(project_dir, "checkpoints/hubert_kmeans/model.mdl")
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
        args.audio_path,
        device=str(device),
        hubert_path=args.hubert_path,
        kmeans_path=args.kmeans_path,
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
    mask_model = load_mask_transformer(args.mask_ckpt, device=str(device))
    if args.step is None:
        args.step = int(mask_model.config.num_frames) - 1

    print("\n>>> Step 3b: 检查 RVQVAE/Mask Transformer token 契约...")
    config = load_config_from_checkpoint(args.rvqvae_ckpt)
    validate_body_codec_compatibility(mask_model, config)
    print("  RVQVAE/Mask Transformer token 契约一致")

    prefix_audio_tokens, prefix_audio_features, prefix_motion_tokens = load_continuation_inputs(args)
    if prefix_audio_tokens is not None:
        print(
            "  Continuation prefix: "
            f"audio={len(prefix_audio_tokens)}, "
            f"features={0 if prefix_audio_features is None else len(prefix_audio_features)}, "
            f"motion={len(prefix_motion_tokens)}, "
            f"keyframes={args.continuation_prefix_keyframes}"
        )
    
    # ---- Step 4: Pipeline 推理 (LLM + Mask Transformer) ----
    print("\n>>> Step 4: Pipeline 推理 (LLM + Mask Transformer)...")
    result = run_pipeline_single(
        vllm_client,
        mask_model,
        action_text=args.action_text,
        audio_tokens=audio_tokens,
        audio_features=audio_features,
        name=args.output_name,
        step=args.step,
        temperature=args.temperature,
        top_p=args.top_p,
        generate_steps=args.generate_steps,
        prefix_audio_tokens=prefix_audio_tokens,
        prefix_audio_features=prefix_audio_features,
        prefix_motion_tokens=prefix_motion_tokens,
        continuation_prefix_keyframes=args.continuation_prefix_keyframes,
        llm_max_tokens=args.llm_max_tokens,
        keyframe_len_policy=args.keyframe_len_policy,
    )
    
    if result is None:
        print("❌ Pipeline 推理失败!")
        sys.exit(1)
    
    dense_tokens = result["dense_tokens"]
    validate_token_frames(
        dense_tokens,
        num_quantizers=int(config.model.num_quantizers),
        codebook_size=int(config.model.nb_code),
        context=f"{args.output_name}.dense_tokens",
    )
    print(f"  生成 {len(dense_tokens)} 帧 dense motion tokens")
    
    # ---- Step 5: RVQVAE 解码 ----
    print("\n>>> Step 5: RVQVAE Token 解码...")
    rvq_model = load_model(args.rvqvae_ckpt, config, device)
    
    # 加载归一化参数：优先使用 RVQVAE checkpoint 旁边的 meta/mean.npy 和 meta/std.npy
    mean, std = load_motion_stats(args.rvqvae_ckpt, device)
    
    # 加载占位符手部数据
    placeholder_npy = os.path.join(module_dir, "meta/xiu_joint_quat_vecs/Daiji_A_001_V001.npy")
    placeholder_motion_dict = np.load(placeholder_npy, allow_pickle=True).item()
    validate_placeholder_motion(
        placeholder_motion_dict,
        left_dim=int(config.data.left_dim),
        right_dim=int(config.data.right_dim),
    )
    
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
    if args.face_mode == "infill":
        if args.face_rvqvae_ckpt is None or args.face_infill_ckpt is None:
            raise ValueError("--face_mode infill requires --face_rvqvae_ckpt and --face_infill_ckpt")
        from face_infill import FaceInfillPipeline, add_face_to_anim

        face_pipeline = FaceInfillPipeline(
            face_rvqvae_ckpt=args.face_rvqvae_ckpt,
            face_infill_ckpt=args.face_infill_ckpt,
            device=device,
            generate_steps=args.face_generate_steps,
        )
        anim = add_face_to_anim(anim, face_pipeline.infer_mta52(audio_features))
    with open(json_path, "w") as f:
        json.dump(anim, f, indent=2, ensure_ascii=False)
    print(f"  ✅ JSON → {json_path}")

    audio_token_path, audio_feat_path, motion_token_path, pipeline_path = save_single_case_artifacts(
        args.output_dir,
        args.output_name,
        result,
        audio_tokens,
        audio_features,
        args.action_text,
        args.audio_path,
        config,
    )
    print(f"  ✅ Audio tokens → {audio_token_path}")
    print(f"  ✅ Audio features → {audio_feat_path}")
    print(f"  ✅ Motion tokens → {motion_token_path}")
    print(f"  ✅ Pipeline result → {pipeline_path}")
    
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
    print(f"    - {pipeline_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
