#!/usr/bin/env python3
"""Encode ARKit face sequences into Face R-VQVAE residual token JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from models.face_rvqvae import load_face_rvqvae_checkpoint
from training.data import (
    list_face_names,
    load_face_array,
    resolve_sequence_path,
    resolve_face_path,
    resample_sequence_to_length,
    set_seed,
)


def read_split_if_exists(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dedupe(names: list[str]) -> list[str]:
    seen = set()
    out = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def resolve_names(face_dir: str, args: argparse.Namespace) -> list[str]:
    split_file = Path(args.split_file) if args.split_file else None
    if split_file and split_file.exists():
        return read_split_if_exists(split_file)

    split_dir = Path(args.data_root) / "split"
    names: list[str] = []
    used: list[str] = []
    for file_name in ("all_file_list.txt", "train_file_list.txt", "val_file_list.txt", "test_file_list.txt"):
        path = split_dir / file_name
        split_names = read_split_if_exists(path)
        if split_names:
            names.extend(split_names)
            used.append(str(path))
            if file_name == "all_file_list.txt":
                break
    names = dedupe(names)
    if names:
        print(f"Using split files for face preprocessing: {', '.join(used)}")
        return names

    print(f"No split file found for face preprocessing; discovering samples under {face_dir}")
    return list_face_names(face_dir, None)


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    face_dir = args.face_dir or str(Path(args.data_root) / "arkit_data")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, mean, std, _ = load_face_rvqvae_checkpoint(args.face_rvqvae_ckpt, device)
    names = resolve_names(face_dir, args)
    if args.limit_files:
        names = names[: args.limit_files]

    success = skipped = failed = 0
    for name in tqdm(names, desc="encode face tokens"):
        try:
            out_path = output_dir / f"{name}.json"
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue
            face = load_face_array(resolve_face_path(face_dir, name), args.face_dim)
            source_face_frames = int(len(face))
            target_len = None
            if args.audio_feat_dir:
                try:
                    audio_path = resolve_sequence_path(args.audio_feat_dir, name, ".npy")
                    target_len = int(np.load(audio_path, mmap_mode="r").shape[0])
                except FileNotFoundError:
                    pass
            if target_len is not None:
                face = resample_sequence_to_length(face, target_len)
            x = torch.tensor(face, dtype=torch.float32, device=device)
            x = (x - mean) / std.clamp_min(1e-6)
            code_idx = model.encode(x.unsqueeze(0))[0].detach().cpu().numpy().astype(int).tolist()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(
                    {
                        "name": name,
                        "tokens": code_idx,
                        "num_tokens": len(code_idx),
                        "num_quantizers": model.config.num_quantizers,
                        "codebook_size": model.config.codebook_size,
                        "aligned_to_audio_features": target_len is not None,
                        "source_face_frames": source_face_frames,
                        "target_audio_feature_frames": target_len,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            success += 1
        except Exception as exc:
            failed += 1
            print(f"[ERROR] {name}: {exc}")
    print(f"face token preprocessing complete: success={success}, skipped={skipped}, failed={failed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess ARKit face data into Face R-VQVAE tokens")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--face_dir", default=None)
    parser.add_argument("--audio_feat_dir", default="./data/audio_features_hubert_layer9_fps10")
    parser.add_argument("--split_file", default="./data/split/all_file_list.txt")
    parser.add_argument("--output_dir", default="./data/face_token_data")
    parser.add_argument("--face_rvqvae_ckpt", default="./checkpoints_train/face_rvqvae/latest.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--face_dim", type=int, default=51)
    parser.add_argument("--limit_files", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
