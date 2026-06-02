#!/usr/bin/env python3
"""Lightweight audio-motion synchronization metrics for generated motions.

This script does not require the ChronTMR retrieval model. It is intended as a
fast verification step for the paper's audio-motion sync metrics, especially
ESD from the appendix.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.io import wavfile
from scipy.signal import find_peaks
from scipy.stats import pearsonr

try:
    import librosa
except ImportError:
    librosa = None


def load_audio_mono(wav_path: Path, sr: int) -> tuple[np.ndarray, int]:
    if librosa is not None:
        y, used_sr = librosa.load(str(wav_path), sr=sr)
        return y.astype(np.float32), int(used_sr)
    used_sr, data = wavfile.read(str(wav_path))
    data = np.asarray(data)
    if data.ndim > 1:
        data = data[:, 0]
    if np.issubdtype(data.dtype, np.integer):
        max_abs = max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
        data = data.astype(np.float32) / float(max_abs)
    else:
        data = data.astype(np.float32)
    if used_sr != sr:
        from scipy.signal import resample_poly

        gcd = int(np.gcd(used_sr, sr))
        data = resample_poly(data, sr // gcd, used_sr // gcd).astype(np.float32)
        used_sr = sr
    return data, int(used_sr)


def frame_rms(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if len(y) < frame_length:
        return np.array([float(np.sqrt(np.mean(y * y) + 1e-12))], dtype=np.float32)
    values = []
    for start in range(0, len(y) - frame_length + 1, hop_length):
        frame = y[start : start + frame_length]
        values.append(float(np.sqrt(np.mean(frame * frame) + 1e-12)))
    return np.asarray(values, dtype=np.float32)


def fallback_onset_envelope(y: np.ndarray, sr: int, hop_length: int = 512) -> tuple[np.ndarray, np.ndarray]:
    rms = frame_rms(y, frame_length=1024, hop_length=hop_length)
    onset_env = np.maximum(np.diff(rms, prepend=rms[:1]), 0.0)
    times = np.arange(len(onset_env), dtype=np.float64) * hop_length / float(sr)
    return onset_env, times


def load_motion_body(path: Path) -> tuple[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.shape == ():
        raw = raw.item()
    if isinstance(raw, dict):
        name = raw.get("name") or path.stem.rsplit("_", 1)[0]
        body = np.asarray(raw.get("body"), dtype=np.float32)
    else:
        name = path.stem.rsplit("_", 1)[0]
        body = np.asarray(raw, dtype=np.float32)
    if body.ndim != 2:
        raise ValueError(f"Expected motion body shape (T,D), got {body.shape} from {path}")
    return str(name), body


def extract_audio_events(wav_path: Path, sr: int = 16000) -> np.ndarray:
    y, used_sr = load_audio_mono(wav_path, sr=sr)
    if y.size == 0:
        return np.array([], dtype=np.float64)
    if librosa is not None:
        onset_env = librosa.onset.onset_strength(y=y, sr=used_sr, hop_length=512)
        frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=used_sr,
            hop_length=512,
            backtrack=False,
            units="frames",
        )
        return librosa.frames_to_time(frames, sr=used_sr, hop_length=512).astype(np.float64)
    onset_env, times = fallback_onset_envelope(y, used_sr, hop_length=512)
    threshold = float(np.mean(onset_env) + 0.5 * np.std(onset_env))
    peaks, _ = find_peaks(onset_env, distance=2, height=threshold)
    return times[peaks].astype(np.float64)


def extract_motion_events(body: np.ndarray, fps: float) -> np.ndarray:
    if len(body) < 3:
        return np.array([], dtype=np.float64)
    velocity = np.diff(body, axis=0)
    vel_mag = np.linalg.norm(velocity, axis=1)
    threshold = float(np.mean(vel_mag) + 0.2 * np.std(vel_mag))
    peaks, _ = find_peaks(vel_mag, distance=1, height=threshold)
    return (peaks / fps).astype(np.float64)


def calculate_esd(audio_times: np.ndarray, motion_times: np.ndarray, empty_penalty: float = 2.0) -> float:
    if len(audio_times) == 0 and len(motion_times) == 0:
        return 0.0
    if len(audio_times) == 0 or len(motion_times) == 0:
        return float(empty_penalty)
    distances = np.abs(audio_times[:, None] - motion_times[None, :])
    audio_to_motion = float(distances.min(axis=1).mean())
    motion_to_audio = float(distances.min(axis=0).mean())
    return 0.5 * (audio_to_motion + motion_to_audio)


def calculate_bhr(audio_times: np.ndarray, motion_times: np.ndarray, tolerance: float) -> tuple[float, float]:
    if len(audio_times) == 0 or len(motion_times) == 0:
        return 0.0, 0.0
    nearest = np.abs(motion_times[:, None] - audio_times[None, :]).min(axis=1)
    bas = float(nearest.mean())
    bhr = float((nearest <= tolerance).mean())
    return bas, bhr


def velocity_onset_correlation(body: np.ndarray, wav_path: Path, fps: float, sr: int, sigma: float) -> float | None:
    if len(body) < 5:
        return None
    velocity = np.linalg.norm(np.diff(body, axis=0), axis=1)
    velocity = gaussian_filter1d(velocity, sigma=sigma)
    if float(velocity.std()) < 1e-8:
        return None

    y, used_sr = load_audio_mono(wav_path, sr=sr)
    if librosa is not None:
        onset_env = librosa.onset.onset_strength(y=y, sr=used_sr, hop_length=512)
        onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=used_sr, hop_length=512)
    else:
        onset_env, onset_times = fallback_onset_envelope(y, used_sr, hop_length=512)
    motion_times = np.arange(len(velocity), dtype=np.float64) / fps
    onset_resampled = np.interp(motion_times, onset_times, onset_env)
    if float(onset_resampled.std()) < 1e-8:
        return None
    corr, _ = pearsonr(velocity, onset_resampled)
    return float((corr + 1.0) * 50.0)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": round(float(arr.mean()), 6), "std": round(float(arr.std()), 6)}


def main(args: argparse.Namespace) -> int:
    motion_dir = Path(args.motion_dir)
    wav_dir = Path(args.wav_dir)
    suffix = f"_{args.motion_type}.npy"
    files = sorted(motion_dir.glob(f"*{suffix}"))
    if args.max_samples is not None:
        files = files[: args.max_samples]
    if not files:
        print(f"No *{suffix} files found under {motion_dir}", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []
    skipped = 0
    for path in files:
        try:
            name, body = load_motion_body(path)
            wav_path = wav_dir / f"{name}.wav"
            if not wav_path.exists():
                skipped += 1
                rows.append({"name": name, "path": str(path), "skipped": "missing wav"})
                continue
            audio_events = extract_audio_events(wav_path, sr=args.sample_rate)
            motion_events = extract_motion_events(body, fps=args.fps)
            esd = calculate_esd(audio_events, motion_events, empty_penalty=args.empty_penalty)
            bas, bhr = calculate_bhr(audio_events, motion_events, tolerance=args.tolerance)
            voc = velocity_onset_correlation(body, wav_path, args.fps, args.sample_rate, args.sigma)
            rows.append(
                {
                    "name": name,
                    "path": str(path),
                    "wav_path": str(wav_path),
                    "num_audio_events": int(len(audio_events)),
                    "num_motion_events": int(len(motion_events)),
                    "ESD": round(float(esd), 6),
                    "BAS": round(float(bas), 6),
                    "BHR": round(float(bhr), 6),
                    "VOC": None if voc is None else round(float(voc), 6),
                }
            )
        except Exception as exc:
            skipped += 1
            rows.append({"name": path.stem, "path": str(path), "skipped": str(exc)})

    valid = [row for row in rows if "ESD" in row]
    metrics = {
        "motion_type": args.motion_type,
        "num_files": len(files),
        "num_evaluated": len(valid),
        "num_skipped": skipped,
        "ESD": summarize([float(row["ESD"]) for row in valid]),
        "BAS": summarize([float(row["BAS"]) for row in valid]),
        "BHR": summarize([float(row["BHR"]) for row in valid]),
        "VOC": summarize([float(row["VOC"]) for row in valid if row.get("VOC") is not None]),
    }
    payload = {"metrics": metrics, "samples": rows}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if valid else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate lightweight audio-motion sync metrics")
    parser.add_argument("--motion_dir", default="./output/reconstructed")
    parser.add_argument("--wav_dir", default="./data/wav_data")
    parser.add_argument("--motion_type", choices=["pred", "gt"], default="pred")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--empty_penalty", type=float, default=2.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main(parse_args()))
