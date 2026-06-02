#!/usr/bin/env python3
"""Evaluate BEATv2-style FGD / BC / Diversity for generated motion.

This is a practical repository-local evaluator. By default it uses deterministic
handcrafted motion statistics as gesture features, so the numbers are useful for
regression tests and ablations but are not the official BEATv2 benchmark scores.
Pass a TorchScript feature extractor with --feature_model_path to compute FGD and
Diversity from a project-specific or official gesture embedding model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import linalg
from scipy.io import wavfile
from scipy.signal import find_peaks, resample_poly

try:
    import librosa
except ImportError:
    librosa = None


def load_npy_object(path: Path) -> Any:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == ():
        return data.item()
    if isinstance(data, np.ndarray) and data.dtype == object:
        try:
            return data.item()
        except Exception:
            return data
    return data


def motion_name(path: Path, payload: Any, suffix: str) -> str:
    if isinstance(payload, dict) and payload.get("name"):
        return str(payload["name"])
    stem = path.name
    if suffix and stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    elif stem.endswith(".npy"):
        stem = stem[:-4]
    return stem


def load_motion_array(path: Path, parts: list[str]) -> tuple[str, np.ndarray]:
    payload = load_npy_object(path)
    name = motion_name(path, payload, "")
    if isinstance(payload, dict):
        name = str(payload.get("name") or name)
        arrays = []
        for part in parts:
            if part not in payload:
                continue
            arr = np.asarray(payload[part], dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"{path}: part {part} must have shape (T,D), got {arr.shape}")
            arrays.append(arr)
        if not arrays:
            for key in ("motion", "poses", "joints"):
                if key in payload:
                    arr = np.asarray(payload[key], dtype=np.float32)
                    if arr.ndim > 2:
                        arr = arr.reshape(arr.shape[0], -1)
                    if arr.ndim == 2:
                        arrays.append(arr)
                        break
        if not arrays:
            raise ValueError(f"{path}: no requested motion parts found: {parts}")
        min_len = min(arr.shape[0] for arr in arrays)
        motion = np.concatenate([arr[:min_len] for arr in arrays], axis=1)
        return name, motion.astype(np.float32)

    arr = np.asarray(payload, dtype=np.float32)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected motion array shape (T,D), got {arr.shape}")
    return name, arr


def read_wav_mono(path: Path, sample_rate: int) -> tuple[np.ndarray, int]:
    if librosa is not None:
        y, sr = librosa.load(str(path), sr=sample_rate)
        return y.astype(np.float32), int(sr)
    sr, y = wavfile.read(str(path))
    y = np.asarray(y)
    if y.ndim > 1:
        y = y[:, 0]
    if np.issubdtype(y.dtype, np.integer):
        max_abs = max(abs(np.iinfo(y.dtype).min), np.iinfo(y.dtype).max)
        y = y.astype(np.float32) / float(max_abs)
    else:
        y = y.astype(np.float32)
    if sr != sample_rate:
        gcd = int(np.gcd(sr, sample_rate))
        y = resample_poly(y, sample_rate // gcd, sr // gcd).astype(np.float32)
        sr = sample_rate
    return y, int(sr)


def resolve_wav(wav_dir: Path | None, name: str) -> Path | None:
    if wav_dir is None:
        return None
    candidates = [
        wav_dir / f"{name}.wav",
        wav_dir / f"{name.replace('/', '_')}.wav",
        wav_dir / f"{Path(name).name}.wav",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def audio_beats(wav_path: Path, sample_rate: int) -> np.ndarray:
    y, sr = read_wav_mono(wav_path, sample_rate)
    if y.size == 0:
        return np.array([], dtype=np.float64)
    if librosa is not None:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=512,
            backtrack=False,
            units="frames",
        )
        return librosa.frames_to_time(frames, sr=sr, hop_length=512).astype(np.float64)
    frame = 1024
    hop = 512
    if len(y) < frame:
        return np.array([], dtype=np.float64)
    rms = []
    for start in range(0, len(y) - frame + 1, hop):
        chunk = y[start : start + frame]
        rms.append(float(np.sqrt(np.mean(chunk * chunk) + 1e-12)))
    env = np.maximum(np.diff(np.asarray(rms), prepend=rms[:1]), 0.0)
    threshold = float(env.mean() + 0.5 * env.std())
    peaks, _ = find_peaks(env, distance=2, height=threshold)
    return (peaks * hop / float(sr)).astype(np.float64)


def motion_beats(motion: np.ndarray, fps: float) -> np.ndarray:
    if motion.shape[0] < 3:
        return np.array([], dtype=np.float64)
    velocity = np.linalg.norm(np.diff(motion, axis=0), axis=1)
    threshold = float(velocity.mean() + 0.2 * velocity.std())
    peaks, _ = find_peaks(velocity, distance=1, height=threshold)
    return (peaks / float(fps)).astype(np.float64)


def beat_consistency(audio_times: np.ndarray, motion_times: np.ndarray, sigma: float) -> float | None:
    """BEAT-family beat consistency: Gaussian score to nearest audio beat."""
    if len(audio_times) == 0 or len(motion_times) == 0:
        return None
    nearest = np.abs(motion_times[:, None] - audio_times[None, :]).min(axis=1)
    return float(np.exp(-(nearest**2) / (2.0 * sigma * sigma)).mean())


def temporal_stats_feature(motion: np.ndarray) -> np.ndarray:
    motion = np.asarray(motion, dtype=np.float32)
    velocity = np.diff(motion, axis=0) if motion.shape[0] > 1 else np.zeros((1, motion.shape[1]), dtype=np.float32)
    accel = np.diff(velocity, axis=0) if velocity.shape[0] > 1 else np.zeros((1, motion.shape[1]), dtype=np.float32)
    speed = np.linalg.norm(velocity, axis=1)
    acc_mag = np.linalg.norm(accel, axis=1)
    stats = [
        motion.mean(axis=0),
        motion.std(axis=0),
        velocity.mean(axis=0),
        velocity.std(axis=0),
        accel.std(axis=0),
        np.asarray([speed.mean(), speed.std(), speed.max(initial=0.0), acc_mag.mean(), acc_mag.std()], dtype=np.float32),
    ]
    return np.concatenate(stats).astype(np.float32)


def project_features(features: np.ndarray, max_dims: int, seed: int) -> np.ndarray:
    if max_dims <= 0 or features.shape[1] <= max_dims:
        return features
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, 1.0 / math.sqrt(max_dims), size=(features.shape[1], max_dims)).astype(np.float32)
    return features @ matrix


def torchscript_features(model_path: Path, motions: list[np.ndarray], device: str) -> np.ndarray:
    import torch

    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    outputs = []
    with torch.no_grad():
        for motion in motions:
            tensor = torch.tensor(motion, dtype=torch.float32, device=device).unsqueeze(0)
            out = model(tensor)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if out.dim() > 2:
                out = out.mean(dim=1)
            outputs.append(out.reshape(out.shape[0], -1)[0].detach().cpu().numpy())
    return np.stack(outputs).astype(np.float32)


def activation_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    if features.shape[0] < 2:
        return mean, np.zeros((features.shape[1], features.shape[1]), dtype=np.float64)
    return mean, np.cov(features, rowvar=False)


def frechet_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    mu1, sigma1 = activation_stats(a)
    mu2, sigma2 = activation_stats(b)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2
    cov_product = sigma1.dot(sigma2)
    if np.allclose(cov_product, 0.0):
        covmean = np.zeros_like(cov_product)
    else:
        covmean, _ = linalg.sqrtm(cov_product, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def diversity(features: np.ndarray, num_pairs: int, seed: int) -> float:
    if features.shape[0] < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    count = min(num_pairs, features.shape[0] * (features.shape[0] - 1))
    values = []
    for _ in range(count):
        i, j = rng.choice(features.shape[0], size=2, replace=False)
        values.append(float(np.linalg.norm(features[i] - features[j])))
    return float(np.mean(values)) if values else 0.0


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": round(float(arr.mean()), 6), "std": round(float(arr.std()), 6)}


def collect_pairs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray]]:
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    pred_paths = sorted(pred_dir.rglob(f"*{args.pred_suffix}"))
    if args.max_samples is not None:
        pred_paths = pred_paths[: args.max_samples]
    rows: list[dict[str, Any]] = []
    pred_motions: list[np.ndarray] = []
    gt_motions: list[np.ndarray] = []
    parts = [part.strip() for part in args.parts.split(",") if part.strip()]
    for pred_path in pred_paths:
        rel = pred_path.relative_to(pred_dir)
        base = str(rel)[: -len(args.pred_suffix)]
        gt_candidates = [
            gt_dir / f"{base}{args.gt_suffix}",
            gt_dir / rel.name.replace(args.pred_suffix, args.gt_suffix),
        ]
        gt_path = next((path for path in gt_candidates if path.exists()), None)
        if gt_path is None:
            rows.append({"name": base, "pred_path": str(pred_path), "skipped": "missing gt"})
            continue
        try:
            pred_name, pred_motion = load_motion_array(pred_path, parts)
            gt_name, gt_motion = load_motion_array(gt_path, parts)
            min_len = min(len(pred_motion), len(gt_motion))
            if min_len <= 1:
                rows.append({"name": pred_name, "pred_path": str(pred_path), "gt_path": str(gt_path), "skipped": "too short"})
                continue
            pred_motion = pred_motion[:min_len]
            gt_motion = gt_motion[:min_len]
            rows.append({
                "name": pred_name,
                "gt_name": gt_name,
                "pred_path": str(pred_path),
                "gt_path": str(gt_path),
                "num_frames": int(min_len),
                "feature_dim": int(pred_motion.shape[1]),
            })
            pred_motions.append(pred_motion)
            gt_motions.append(gt_motion)
        except Exception as exc:
            rows.append({"name": base, "pred_path": str(pred_path), "gt_path": str(gt_path), "skipped": str(exc)})
    return rows, pred_motions, gt_motions


def main(args: argparse.Namespace) -> int:
    rows, pred_motions, gt_motions = collect_pairs(args)
    if not pred_motions:
        print("No valid pred/gt motion pairs found", file=sys.stderr)
        return 1

    if args.feature_model_path:
        feature_type = "torchscript"
        pred_features = torchscript_features(Path(args.feature_model_path), pred_motions, args.device)
        gt_features = torchscript_features(Path(args.feature_model_path), gt_motions, args.device)
    else:
        feature_type = "handcrafted"
        pred_features = np.stack([temporal_stats_feature(motion) for motion in pred_motions])
        gt_features = np.stack([temporal_stats_feature(motion) for motion in gt_motions])
        combined = project_features(np.concatenate([gt_features, pred_features], axis=0), args.max_feature_dims, args.seed)
        gt_features = combined[: len(gt_features)]
        pred_features = combined[len(gt_features) :]

    fgd = frechet_distance(gt_features, pred_features)
    div_gt = diversity(gt_features, args.diversity_pairs, args.seed)
    div_pred = diversity(pred_features, args.diversity_pairs, args.seed + 1)

    bc_values: list[float] = []
    wav_dir = Path(args.wav_dir) if args.wav_dir else None
    valid_idx = 0
    for row in rows:
        if "skipped" in row:
            continue
        pred_motion = pred_motions[valid_idx]
        valid_idx += 1
        wav_path = resolve_wav(wav_dir, row["name"]) if wav_dir is not None else None
        if wav_path is None:
            row["bc_skipped"] = "missing wav" if wav_dir is not None else "wav_dir not provided"
            continue
        try:
            audio_times = audio_beats(wav_path, args.sample_rate)
            motion_times = motion_beats(pred_motion, args.fps)
            bc = beat_consistency(audio_times, motion_times, args.bc_sigma)
            row["wav_path"] = str(wav_path)
            row["num_audio_beats"] = int(len(audio_times))
            row["num_motion_beats"] = int(len(motion_times))
            row["BC"] = None if bc is None else round(float(bc), 6)
            row["BC_x10"] = None if bc is None else round(float(bc) * 10.0, 6)
            if bc is not None:
                bc_values.append(float(bc))
        except Exception as exc:
            row["bc_skipped"] = str(exc)

    metrics = {
        "feature_type": feature_type,
        "num_pairs": len(pred_motions),
        "FGD": round(float(fgd), 6),
        "Diversity_GT": round(float(div_gt), 6),
        "Diversity_Pred": round(float(div_pred), 6),
        "BC": summarize(bc_values),
        "BC_x10": summarize([v * 10.0 for v in bc_values]),
        "paper_equivalent": bool(args.feature_model_path),
    }
    payload = {"metrics": metrics, "samples": rows}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute BEATv2-style FGD/BC/Diversity metrics")
    parser.add_argument("--pred_dir", default="./output/reconstructed")
    parser.add_argument("--gt_dir", default="./output/reconstructed")
    parser.add_argument("--wav_dir", default="./data/wav_data")
    parser.add_argument("--pred_suffix", default="_pred.npy")
    parser.add_argument("--gt_suffix", default="_gt.npy")
    parser.add_argument("--parts", default="body,left,right",
                        help="Comma-separated dict keys to concatenate when loading motion npy files")
    parser.add_argument("--feature_model_path", default=None,
                        help="Optional TorchScript feature extractor for official/project-specific embeddings")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_feature_dims", type=int, default=256,
                        help="Deterministically project handcrafted statistics to this dimension; <=0 disables projection")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--bc_sigma", type=float, default=0.3)
    parser.add_argument("--diversity_pairs", type=int, default=300)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main(parse_args()))
