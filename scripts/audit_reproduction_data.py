#!/usr/bin/env python3
"""Audit SentiAvatar data/checkpoint layout before training or inference."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = PROJECT_ROOT / "motion_generation"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from training.data import extract_action_text, normalize_motion_text


def read_split(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rel_name(path: Path, root: Path, suffix: str) -> str:
    return str(path.relative_to(root)).replace(os.sep, "/")[: -len(suffix)]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def token_shape(tokens: Any) -> tuple[int, int | None]:
    arr = np.asarray(tokens)
    if arr.ndim == 1:
        return int(arr.shape[0]), None
    if arr.ndim == 2:
        return int(arr.shape[0]), int(arr.shape[1])
    return int(arr.shape[0]) if arr.ndim else 0, -1


def parse_int_csv(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


class Auditor:
    def __init__(self) -> None:
        self.items: list[dict[str, str]] = []

    def add(self, level: str, message: str, evidence: str = "") -> None:
        self.items.append({"level": level, "message": message, "evidence": evidence})
        prefix = {"ok": "OK", "warn": "WARN", "error": "ERROR"}[level]
        line = f"[{prefix}] {message}"
        if evidence:
            line += f" ({evidence})"
        print(line)

    def ok(self, message: str, evidence: str = "") -> None:
        self.add("ok", message, evidence)

    def warn(self, message: str, evidence: str = "") -> None:
        self.add("warn", message, evidence)

    def error(self, message: str, evidence: str = "") -> None:
        self.add("error", message, evidence)

    def summary(self) -> dict[str, int]:
        out = {"ok": 0, "warn": 0, "error": 0}
        for item in self.items:
            out[item["level"]] += 1
        return out


def existing_sample_names(data_dir: Path, max_samples: int | None) -> list[str]:
    split = read_split(data_dir / "split" / "all_file_list.txt")
    if split:
        return split[:max_samples]
    roots = [
        (data_dir / "motion_data", ".npy"),
        (data_dir / "audio_features_hubert_layer9_fps10", ".npy"),
        (data_dir / "motion_token_data", ".json"),
    ]
    names: set[str] = set()
    for root, suffix in roots:
        if root.exists():
            names.update(rel_name(path, root, suffix) for path in root.rglob(f"*{suffix}"))
    return sorted(names)[:max_samples]


def check_motion(auditor: Auditor, path: Path, args: argparse.Namespace) -> None:
    if not path.exists():
        auditor.error("motion file missing", str(path))
        return
    try:
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object:
            data = data.item()
        if not isinstance(data, dict):
            auditor.error("motion file is not a dict", str(path))
            return
        expected = {"body": args.body_dim, "left": args.hand_dim, "right": args.hand_dim}
        for key, dim in expected.items():
            arr = np.asarray(data.get(key))
            if arr.ndim != 2 or arr.shape[1] < dim:
                auditor.error(f"motion[{key}] has invalid shape", f"{path}: {arr.shape}")
            else:
                auditor.ok(f"motion[{key}] shape", f"{path}: {arr.shape}")
    except Exception as exc:
        auditor.error("failed to load motion file", f"{path}: {exc}")


def check_motion2text(auditor: Auditor, data_dir: Path, names: list[str], args: argparse.Namespace) -> None:
    path = Path(args.motion2text_json) if args.motion2text_json else data_dir / "text_data" / "motion2text.json"
    if not path.exists():
        auditor.warn("motion2text json missing", str(path))
        return
    try:
        payload = load_json(path)
    except Exception as exc:
        auditor.error("failed to load motion2text json", f"{path}: {exc}")
        return
    if not isinstance(payload, dict):
        auditor.error("motion2text json must be an object mapping sample names to text metadata", str(path))
        return
    auditor.ok("motion2text json exists", f"{path}: {len(payload)} entries")
    missing = [name for name in names if name not in payload]
    if missing:
        auditor.warn("motion2text missing checked samples", ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else ""))
    checked = 0
    for name in names:
        if name not in payload:
            continue
        try:
            normalized = normalize_motion_text(payload[name])
            action = extract_action_text(payload[name])
        except Exception as exc:
            auditor.error("failed to normalize motion2text entry", f"{name}: {exc}")
            continue
        if not normalized:
            auditor.error("motion2text entry normalized to empty text", name)
        elif not action:
            auditor.error("motion2text entry produced empty action text", name)
        else:
            checked += 1
    if checked:
        auditor.ok("motion2text entries normalize to action text", f"{checked} checked")


def check_feature_and_tokens(auditor: Auditor, name: str, data_dir: Path, args: argparse.Namespace) -> tuple[int | None, int | None]:
    feat_len: int | None = None
    audio_token_len: int | None = None

    feat_path = data_dir / "audio_features_hubert_layer9_fps10" / f"{name}.npy"
    if feat_path.exists():
        try:
            feat = np.load(feat_path, mmap_mode="r")
            if feat.ndim != 2:
                auditor.error("audio feature has invalid rank", f"{feat_path}: {feat.shape}")
            else:
                feat_len = int(feat.shape[0])
                level = "ok" if feat.shape[1] == args.audio_feat_dim else "warn"
                getattr(auditor, level)(
                    "audio feature shape",
                    f"{feat_path}: {feat.shape}, expected dim {args.audio_feat_dim}",
                )
        except Exception as exc:
            auditor.error("failed to load audio feature", f"{feat_path}: {exc}")
    elif args.require_preprocessed:
        auditor.error("audio feature missing", str(feat_path))
    else:
        auditor.warn("audio feature missing", str(feat_path))

    audio_token_path = data_dir / "audio_tokens_hubert_layer9_fps10" / f"{name}.json"
    if audio_token_path.exists():
        try:
            payload = load_json(audio_token_path)
            audio_token_len, width = token_shape(payload.get("tokens", []))
            if width is not None:
                auditor.warn("audio tokens are nested; expected one token per frame", f"{audio_token_path}: width={width}")
            else:
                auditor.ok("audio token length", f"{audio_token_path}: {audio_token_len}")
            audio_values = np.asarray(payload.get("tokens", [])).reshape(-1)
            if audio_values.size > 0:
                bad = audio_values[(audio_values < 0) | (audio_values > args.max_audio_token)]
                if bad.size > 0:
                    auditor.error(
                        "audio token value outside planner tokenizer range",
                        f"{audio_token_path}: first_bad={int(bad[0])}, expected [0,{args.max_audio_token}]",
                    )
                else:
                    auditor.ok("audio token value range", f"{audio_token_path}: max={int(audio_values.max())}")
            if feat_len is not None and audio_token_len != feat_len:
                auditor.error("audio token length does not match audio feature length", f"{audio_token_len} vs {feat_len}: {name}")
        except Exception as exc:
            auditor.error("failed to load audio token json", f"{audio_token_path}: {exc}")
    elif args.require_preprocessed:
        auditor.error("audio token json missing", str(audio_token_path))
    else:
        auditor.warn("audio token json missing", str(audio_token_path))

    return feat_len, audio_token_len


def check_motion_tokens(
    auditor: Auditor,
    name: str,
    data_dir: Path,
    feat_len: int | None,
    args: argparse.Namespace,
    rvq_temporal: dict[str, float] | None,
) -> int | None:
    path = data_dir / "motion_token_data" / f"{name}.json"
    if not path.exists():
        if args.require_preprocessed:
            auditor.error("motion token json missing", str(path))
        else:
            auditor.warn("motion token json missing", str(path))
        return None
    try:
        payload = load_json(path)
        length, width = token_shape(payload.get("tokens", []))
        if width != args.num_body_quantizers:
            auditor.error("motion tokens must be residual groups", f"{path}: shape=({length},{width})")
        else:
            auditor.ok("motion token shape", f"{path}: ({length},{width})")
        if feat_len is not None and length != feat_len:
            auditor.error("motion token length does not match GitHub audio feature length", f"{length} vs {feat_len}: {name}")
        token_fps = payload.get("token_fps")
        source_fps = payload.get("source_fps")
        downsample_factor = payload.get("downsample_factor")
        legacy_fps = payload.get("fps")
        if token_fps is not None:
            if abs(float(token_fps) - args.expected_token_fps) > 1e-3:
                auditor.warn("motion token_fps differs from expected GitHub token rate", f"{path}: {token_fps}")
            else:
                auditor.ok("motion token_fps metadata", f"{path}: {token_fps}")
        elif legacy_fps is not None:
            auditor.warn("motion token json uses legacy fps metadata; prefer token_fps/source_fps", f"{path}: fps={legacy_fps}")
        else:
            auditor.warn("motion token json missing token_fps metadata", str(path))
        if rvq_temporal is not None:
            expected_source_fps = rvq_temporal["source_fps"]
            expected_token_fps = rvq_temporal["token_fps"]
            expected_downsample = int(rvq_temporal["downsample_factor"])
            if source_fps is None:
                auditor.warn("motion token json missing source_fps metadata", str(path))
            elif abs(float(source_fps) - expected_source_fps) > 1e-3:
                auditor.error("motion token source_fps does not match RVQVAE config", f"{source_fps} vs {expected_source_fps}: {path}")
            else:
                auditor.ok("motion token source_fps matches RVQVAE config", f"{path}: {source_fps}")
            if downsample_factor is None:
                auditor.warn("motion token json missing downsample_factor metadata", str(path))
            elif int(downsample_factor) != expected_downsample:
                auditor.error("motion token downsample_factor does not match RVQVAE config", f"{downsample_factor} vs {expected_downsample}: {path}")
            else:
                auditor.ok("motion token downsample_factor matches RVQVAE config", f"{path}: {downsample_factor}")
            if token_fps is not None:
                if abs(float(token_fps) - expected_token_fps) > 1e-3:
                    auditor.error("motion token_fps does not match RVQVAE config", f"{token_fps} vs {expected_token_fps:.4f}: {path}")
                else:
                    auditor.ok("motion token_fps matches RVQVAE config", f"{path}: {token_fps}")
        return length
    except Exception as exc:
        auditor.error("failed to load motion token json", f"{path}: {exc}")
        return None


def check_face(auditor: Auditor, name: str, data_dir: Path, feat_len: int | None, args: argparse.Namespace) -> None:
    face_path = data_dir / "arkit_data" / f"{name}.npy"
    if face_path.exists():
        try:
            arr = np.load(face_path, allow_pickle=True)
            if isinstance(arr, np.ndarray) and arr.dtype == object:
                arr = arr.item()
            if isinstance(arr, dict):
                for key in ("face", "arkit", "arkit_data", "blendshape", "blendshapes"):
                    if key in arr:
                        arr = arr[key]
                        break
            arr = np.asarray(arr)
            if arr.ndim != 2 or arr.shape[1] < args.face_dim:
                auditor.error("ARKit face data has invalid shape", f"{face_path}: {arr.shape}")
            else:
                auditor.ok("ARKit face shape", f"{face_path}: {arr.shape}")
        except Exception as exc:
            auditor.error("failed to load ARKit face data", f"{face_path}: {exc}")
    else:
        auditor.warn("ARKit face data missing", str(face_path))

    face_token_path = data_dir / "face_token_data" / f"{name}.json"
    if face_token_path.exists():
        try:
            payload = load_json(face_token_path)
            length, width = token_shape(payload.get("tokens", []))
            if width != args.num_face_quantizers:
                auditor.error(
                    "face tokens must match expected residual group width",
                    f"{face_token_path}: shape=({length},{width}), expected width={args.num_face_quantizers}",
                )
            else:
                auditor.ok("face token shape", f"{face_token_path}: ({length},{width})")
            face_values = np.asarray(payload.get("tokens", [])).reshape(-1)
            if face_values.size > 0:
                bad = face_values[(face_values < 0) | (face_values >= args.face_codebook_size)]
                if bad.size > 0:
                    auditor.error(
                        "face token value outside Face R-VQVAE codebook range",
                        f"{face_token_path}: first_bad={int(bad[0])}, expected [0,{args.face_codebook_size - 1}]",
                    )
                else:
                    auditor.ok("face token value range", f"{face_token_path}: max={int(face_values.max())}")
            meta_quantizers = payload.get("num_quantizers")
            if meta_quantizers is None:
                auditor.warn("face token json missing num_quantizers metadata", str(face_token_path))
            elif int(meta_quantizers) != width:
                auditor.error("face token num_quantizers metadata does not match token width", f"{meta_quantizers} vs {width}: {face_token_path}")
            else:
                auditor.ok("face token num_quantizers metadata", f"{face_token_path}: {meta_quantizers}")
            meta_codebook = payload.get("codebook_size")
            if meta_codebook is None:
                auditor.warn("face token json missing codebook_size metadata", str(face_token_path))
            elif int(meta_codebook) != args.face_codebook_size:
                auditor.error(
                    "face token codebook_size metadata does not match expected value",
                    f"{meta_codebook} vs {args.face_codebook_size}: {face_token_path}",
                )
            else:
                auditor.ok("face token codebook_size metadata", f"{face_token_path}: {meta_codebook}")
            if feat_len is not None and length != feat_len:
                auditor.error("face token length does not match audio feature length", f"{length} vs {feat_len}: {name}")
        except Exception as exc:
            auditor.error("failed to load face token json", f"{face_token_path}: {exc}")
    else:
        auditor.warn("face token json missing", str(face_token_path))


def check_rvqvae_checkpoint(auditor: Auditor, path: str | None) -> None:
    if not path:
        return
    ckpt = Path(path)
    if not ckpt.exists():
        auditor.error("RVQVAE checkpoint missing", str(ckpt))
        return
    auditor.ok("RVQVAE checkpoint exists", str(ckpt))
    exp_dir = ckpt.parent.parent
    for rel in ("opt.txt", "meta/mean.npy", "meta/std.npy"):
        p = exp_dir / rel
        if p.exists():
            auditor.ok(f"RVQVAE {rel} exists", str(p))
        else:
            auditor.error(f"RVQVAE {rel} missing", str(p))


def check_hf_checkpoint(auditor: Auditor, label: str, path: str | None) -> None:
    if not path:
        return
    root = Path(path)
    if not root.exists():
        auditor.error(f"{label} checkpoint directory missing", str(root))
        return
    if (root / "config.json").exists():
        auditor.ok(f"{label} config exists", str(root / "config.json"))
    else:
        auditor.error(f"{label} config missing", str(root / "config.json"))
    weight_candidates = [
        root / "model.safetensors",
        root / "pytorch_model.bin",
        root / "model.safetensors.index.json",
        root / "pytorch_model.bin.index.json",
    ]
    has_weights = any(path.exists() for path in weight_candidates) or any(root.glob("model-*.safetensors"))
    if has_weights:
        auditor.ok(f"{label} weights exist", str(root))
    else:
        auditor.error(f"{label} weights missing", str(root))


def read_rvqvae_opt_values(path: str | None) -> dict[str, str] | None:
    if not path:
        return None
    ckpt = Path(path)
    opt_path = ckpt.parent.parent / "opt.txt"
    if not opt_path.exists():
        return None
    values: dict[str, str] = {}
    for line in opt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("---") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def read_rvqvae_codec_config(path: str | None) -> tuple[int, int] | None:
    values = read_rvqvae_opt_values(path)
    if values is None:
        return None
    try:
        return int(values.get("num_quantizers", 4)), int(values.get("nb_code", 512))
    except ValueError:
        return None


def read_rvqvae_temporal_config(path: str | None) -> dict[str, float] | None:
    values = read_rvqvae_opt_values(path)
    if values is None:
        return None
    try:
        stride_t = int(values.get("stride_t", 2))
        down_t = int(values.get("down_t", 1))
        source_fps = float(values.get("fps", 20))
    except ValueError:
        return None
    downsample_factor = max(1, stride_t**down_t)
    return {
        "stride_t": float(stride_t),
        "down_t": float(down_t),
        "source_fps": source_fps,
        "downsample_factor": float(downsample_factor),
        "token_fps": source_fps / downsample_factor,
    }


def check_body_checkpoint_compatibility(
    auditor: Auditor,
    rvqvae_path: str | None,
    mask_path: str | None,
    expected_step: int | None,
) -> None:
    if not rvqvae_path or not mask_path:
        return
    rvq_codec = read_rvqvae_codec_config(rvqvae_path)
    mask_config_path = Path(mask_path) / "config.json"
    if rvq_codec is None or not mask_config_path.exists():
        return
    try:
        rvq_quantizers, rvq_codebook = rvq_codec
        mask_config = load_json(mask_config_path)
        mask_quantizers = int(mask_config.get("num_tokens_per_frame"))
        mask_codebook = int(mask_config.get("codebook_size"))
        mask_vocab = int(mask_config.get("vocab_size"))
        if rvq_quantizers != mask_quantizers:
            auditor.error("Body RVQVAE/Infill quantizer count mismatch", f"{rvq_quantizers} vs {mask_quantizers}")
        else:
            auditor.ok("Body RVQVAE/Infill quantizer count compatible", str(rvq_quantizers))
        if rvq_codebook != mask_codebook:
            auditor.error("Body RVQVAE/Infill codebook size mismatch", f"{rvq_codebook} vs {mask_codebook}")
        else:
            auditor.ok("Body RVQVAE/Infill codebook size compatible", str(rvq_codebook))
        expected_vocab = mask_codebook * mask_quantizers + 1
        if mask_vocab != expected_vocab:
            auditor.error("Body Infill vocab_size inconsistent with codebook contract", f"{mask_vocab} vs {expected_vocab}")
        else:
            auditor.ok("Body Infill vocab_size matches codebook contract", str(mask_vocab))
        if expected_step is not None:
            actual_step = int(mask_config.get("num_frames")) - 1
            if actual_step != int(expected_step):
                auditor.error("Body Infill keyframe step mismatch", f"{actual_step} vs expected {expected_step}")
            else:
                auditor.ok("Body Infill keyframe step matches planner", str(actual_step))
    except Exception as exc:
        auditor.error("failed to check Body RVQVAE/Infill compatibility", str(exc))


def check_llm_checkpoint(auditor: Auditor, label: str, path: str | None) -> None:
    if not path:
        return
    check_hf_checkpoint(auditor, label, path)
    root = Path(path)
    tokenizer_candidates = [
        root / "tokenizer.json",
        root / "tokenizer.model",
        root / "vocab.json",
    ]
    if any(path.exists() for path in tokenizer_candidates):
        auditor.ok(f"{label} tokenizer exists", str(root))
    else:
        auditor.error(f"{label} tokenizer missing", str(root))


def planner_contract_tokens(
    codebook_size: int,
    num_quantizers: int,
    max_audio_token: int,
    max_len_token: int,
    steps: list[int],
) -> list[str]:
    tokens = [f"[audio_{idx}]" for idx in range(max_audio_token + 1)]
    for q_idx in range(1, num_quantizers + 1):
        tokens.extend(f"[res{q_idx}_{idx}]" for idx in range(codebook_size))
    tokens.extend(f"[len_{idx}]" for idx in range(1, max_len_token + 1))
    tokens.extend(f"[step_{step}]" for step in sorted(set(steps)))
    return tokens


def check_llm_tokenizer_contract(
    auditor: Auditor,
    label: str,
    path: str | None,
    args: argparse.Namespace,
    rvqvae_path: str | None,
) -> None:
    if not path:
        return
    root = Path(path)
    if not root.exists():
        return
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        auditor.error(f"{label} tokenizer contract check unavailable", f"failed to import transformers: {exc}")
        return

    rvq_codec = read_rvqvae_codec_config(rvqvae_path)
    num_quantizers = args.num_body_quantizers
    codebook_size = args.body_codebook_size
    if rvq_codec is not None:
        num_quantizers, codebook_size = rvq_codec

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            root,
            trust_remote_code=True,
            local_files_only=True,
            padding_side="right",
        )
        vocab = tokenizer.get_vocab()
        expected = planner_contract_tokens(
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
            max_audio_token=args.max_audio_token,
            max_len_token=args.max_len_token,
            steps=args.step_tokens,
        )
        missing: list[str] = []
        split: list[tuple[str, list[int]]] = []
        for token in expected:
            if token not in vocab:
                missing.append(token)
                continue
            ids = tokenizer(token, add_special_tokens=False).input_ids
            if len(ids) != 1:
                split.append((token, ids))
        if missing:
            auditor.error(
                f"{label} tokenizer missing planner tokens",
                ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""),
            )
        else:
            auditor.ok(f"{label} tokenizer planner token coverage", f"{len(expected)} tokens")
        if split:
            formatted = ", ".join(f"{tok}->{ids}" for tok, ids in split[:5])
            auditor.error(
                f"{label} tokenizer has non-atomic planner tokens",
                formatted + (" ..." if len(split) > 5 else ""),
            )
        else:
            auditor.ok(f"{label} tokenizer planner tokens are atomic", str(root))
    except Exception as exc:
        auditor.error(f"{label} tokenizer contract check failed", str(exc))


def check_face_rvqvae(auditor: Auditor, path: str | None) -> None:
    if not path:
        return
    p = Path(path)
    if p.is_dir():
        p = p / "latest.pth"
    if p.exists():
        auditor.ok("Face R-VQVAE checkpoint exists", str(p))
    else:
        auditor.error("Face R-VQVAE checkpoint missing", str(p))


def check_face_checkpoint_compatibility(auditor: Auditor, face_rvqvae_path: str | None, face_infill_path: str | None) -> None:
    if not face_rvqvae_path or not face_infill_path:
        return
    rvq_path = Path(face_rvqvae_path)
    if rvq_path.is_dir():
        rvq_path = rvq_path / "latest.pth"
    infill_dir = Path(face_infill_path)
    config_path = infill_dir / "config.json"
    if not rvq_path.exists() or not config_path.exists():
        return
    try:
        import torch

        rvq_ckpt = torch.load(rvq_path, map_location="cpu", weights_only=False)
        rvq_config = rvq_ckpt.get("config", {})
        infill_config = load_json(config_path)
        rvq_quantizers = int(rvq_config.get("num_quantizers"))
        rvq_codebook = int(rvq_config.get("codebook_size"))
        infill_quantizers = int(infill_config.get("num_tokens_per_frame"))
        infill_codebook = int(infill_config.get("codebook_size"))
        infill_vocab = int(infill_config.get("vocab_size"))
        if rvq_quantizers != infill_quantizers:
            auditor.error("Face R-VQVAE/Infill quantizer count mismatch", f"{rvq_quantizers} vs {infill_quantizers}")
        else:
            auditor.ok("Face R-VQVAE/Infill quantizer count compatible", str(rvq_quantizers))
        if rvq_codebook != infill_codebook:
            auditor.error("Face R-VQVAE/Infill codebook size mismatch", f"{rvq_codebook} vs {infill_codebook}")
        else:
            auditor.ok("Face R-VQVAE/Infill codebook size compatible", str(rvq_codebook))
        expected_vocab = infill_codebook * infill_quantizers + 1
        if infill_vocab != expected_vocab:
            auditor.error("Face Infill vocab_size inconsistent with codebook contract", f"{infill_vocab} vs {expected_vocab}")
        else:
            auditor.ok("Face Infill vocab_size matches codebook contract", str(infill_vocab))
    except Exception as exc:
        auditor.error("failed to check Face R-VQVAE/Infill compatibility", str(exc))


def main(args: argparse.Namespace) -> int:
    auditor = Auditor()
    data_dir = Path(args.data_dir)

    if data_dir.exists():
        auditor.ok("data directory exists", str(data_dir))
    else:
        auditor.error("data directory missing", str(data_dir))
        return 1

    names = existing_sample_names(data_dir, args.max_samples)
    if names:
        auditor.ok("sample list resolved", f"{len(names)} checked")
    else:
        auditor.error("no samples found; expected split/all_file_list.txt or preprocessed files", str(data_dir))
        return 1
    rvq_temporal = read_rvqvae_temporal_config(args.rvqvae_ckpt)

    train_split = read_split(data_dir / "split" / "train_file_list.txt")
    if train_split:
        auditor.ok("train split exists", f"{len(train_split)} entries")
    else:
        auditor.warn("train split missing or empty", str(data_dir / "split" / "train_file_list.txt"))

    for name in names:
        print(f"\n== {name} ==")
        check_motion(auditor, data_dir / "motion_data" / f"{name}.npy", args)
        feat_len, _ = check_feature_and_tokens(auditor, name, data_dir, args)
        check_motion_tokens(auditor, name, data_dir, feat_len, args, rvq_temporal)
        if args.check_face:
            check_face(auditor, name, data_dir, feat_len, args)

    print("\n== Text Metadata ==")
    check_motion2text(auditor, data_dir, names, args)

    print("\n== Checkpoints ==")
    check_rvqvae_checkpoint(auditor, args.rvqvae_ckpt)
    check_hf_checkpoint(auditor, "Body Infill Transformer", args.mask_ckpt)
    check_body_checkpoint_compatibility(auditor, args.rvqvae_ckpt, args.mask_ckpt, args.expected_step)
    check_face_rvqvae(auditor, args.face_rvqvae_ckpt)
    check_hf_checkpoint(auditor, "Face Infill Transformer", args.face_infill_ckpt)
    check_face_checkpoint_compatibility(auditor, args.face_rvqvae_ckpt, args.face_infill_ckpt)
    check_llm_checkpoint(auditor, "Motion Foundation LLM", args.motion_foundation_dir)
    check_llm_checkpoint(auditor, "LLM planner", args.llm_dir)
    if args.check_llm_tokenizer_contract:
        check_llm_tokenizer_contract(auditor, "LLM planner", args.llm_dir, args, args.rvqvae_ckpt)

    summary = auditor.summary()
    print(f"\nAudit summary: ok={summary['ok']} warn={summary['warn']} error={summary['error']}")
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(
            json.dumps({"summary": summary, "items": auditor.items}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if summary["error"] > 0:
        return 1
    if args.strict and summary["warn"] > 0:
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit SentiAvatar reproduction data and checkpoints")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--require_preprocessed", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--motion2text_json", default=None)
    parser.add_argument("--body_dim", type=int, default=153)
    parser.add_argument("--hand_dim", type=int, default=120)
    parser.add_argument("--face_dim", type=int, default=51)
    parser.add_argument("--audio_feat_dim", type=int, default=768)
    parser.add_argument("--max_audio_token", type=int, default=2048)
    parser.add_argument("--max_len_token", type=int, default=2048)
    parser.add_argument("--num_body_quantizers", type=int, default=4)
    parser.add_argument("--body_codebook_size", type=int, default=512)
    parser.add_argument("--num_face_quantizers", type=int, default=2)
    parser.add_argument("--face_codebook_size", type=int, default=512)
    parser.add_argument("--expected_token_fps", type=float, default=10.0)
    parser.add_argument("--expected_step", type=int, default=4,
                        help="Expected planner/Infill keyframe interval; use num_frames-1 for the body Infill checkpoint")
    parser.add_argument("--step_tokens", type=parse_int_csv, default=parse_int_csv("1,2,3,4,5,6,8"),
                        help="Comma-separated planner [step_t] tokens expected in the LLM tokenizer")
    parser.add_argument("--no_check_llm_tokenizer_contract", dest="check_llm_tokenizer_contract",
                        action="store_false",
                        help="Skip planner tokenizer atomic-token coverage checks")
    parser.set_defaults(check_llm_tokenizer_contract=True)
    parser.add_argument("--check_face", action="store_true")
    parser.add_argument("--rvqvae_ckpt", default=None)
    parser.add_argument("--mask_ckpt", default=None)
    parser.add_argument("--motion_foundation_dir", default=None)
    parser.add_argument("--llm_dir", default=None)
    parser.add_argument("--face_rvqvae_ckpt", default=None)
    parser.add_argument("--face_infill_ckpt", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main(parse_args()))
