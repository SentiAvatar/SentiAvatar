#!/usr/bin/env python3
"""Dataset helpers shared by SentiAvatar training scripts."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_split(split_file: Optional[str]) -> Optional[List[str]]:
    if not split_file:
        return None
    path = Path(split_file)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def list_motion_names(motion_dir: str, split_file: Optional[str] = None) -> List[str]:
    split_names = read_split(split_file)
    if split_names is not None:
        return split_names

    root = Path(motion_dir)
    if not root.exists():
        raise FileNotFoundError(f"Motion directory not found: {root}")
    return sorted(str(path.relative_to(root)).replace(os.sep, "/")[:-4] for path in root.rglob("*.npy"))


def load_motion_dict(path: str | Path) -> dict:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        arr = arr.item()
    if not isinstance(arr, dict) or "body" not in arr:
        raise ValueError(f"Expected a motion dict with key 'body': {path}")
    return arr


def preprocess_body_motion(motion_dict: dict, body_dim: int = 153) -> np.ndarray:
    """Match inference preprocessing: root height offset and root velocity encoding."""
    body = np.asarray(motion_dict["body"], dtype=np.float32).copy()
    if body.ndim != 2 or body.shape[1] < body_dim:
        raise ValueError(f"Expected body motion shape (T,{body_dim}), got {body.shape}")
    body = body[:, :body_dim]
    body[:, 2] = body[:, 2] - body[0, 2]
    body[1:, :3] = body[1:, :3] - body[:-1, :3]
    return body


def resolve_motion_path(motion_dir: str, name: str) -> Path:
    return resolve_sequence_path(motion_dir, name, ".npy")


def compute_body_stats(
    motion_dir: str,
    names: Sequence[str],
    body_dim: int = 153,
    limit_files: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    sums = np.zeros(body_dim, dtype=np.float64)
    sq_sums = np.zeros(body_dim, dtype=np.float64)
    count = 0
    use_names = list(names[:limit_files]) if limit_files else list(names)
    for name in use_names:
        body = preprocess_body_motion(load_motion_dict(resolve_motion_path(motion_dir, name)), body_dim)
        sums += body.sum(axis=0)
        sq_sums += (body.astype(np.float64) ** 2).sum(axis=0)
        count += body.shape[0]
    if count == 0:
        raise ValueError("No frames found while computing motion statistics")
    mean = sums / count
    var = np.maximum(sq_sums / count - mean**2, 1e-12)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


@dataclass(frozen=True)
class MotionWindow:
    name: str
    start: int


class BodyMotionWindowDataset(Dataset):
    """Sliding-window body motion dataset for Motion R-VQVAE training."""

    def __init__(
        self,
        motion_dir: str,
        names: Sequence[str],
        mean: np.ndarray,
        std: np.ndarray,
        window_size: int = 64,
        stride: int = 32,
        body_dim: int = 153,
        limit_windows: Optional[int] = None,
    ) -> None:
        self.motion_dir = motion_dir
        self.names = list(names)
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)
        self.window_size = int(window_size)
        self.stride = max(1, int(stride))
        self.body_dim = body_dim
        self.cache: dict[str, np.ndarray] = {}
        self.windows: List[MotionWindow] = []

        for name in self.names:
            body = preprocess_body_motion(load_motion_dict(resolve_motion_path(motion_dir, name)), body_dim)
            length = body.shape[0]
            if length <= self.window_size:
                starts = [0]
            else:
                starts = list(range(0, length - self.window_size + 1, self.stride))
                if starts[-1] != length - self.window_size:
                    starts.append(length - self.window_size)
            self.windows.extend(MotionWindow(name, s) for s in starts)

        if limit_windows is not None:
            self.windows = self.windows[:limit_windows]
        if not self.windows:
            raise ValueError("No training windows were created")

    def __len__(self) -> int:
        return len(self.windows)

    def _load_body(self, name: str) -> np.ndarray:
        if name not in self.cache:
            body = preprocess_body_motion(load_motion_dict(resolve_motion_path(self.motion_dir, name)), self.body_dim)
            self.cache[name] = (body - self.mean) / self.std
        return self.cache[name]

    def __getitem__(self, idx: int) -> torch.Tensor:
        item = self.windows[idx]
        body = self._load_body(item.name)
        window = body[item.start : item.start + self.window_size]
        if window.shape[0] < self.window_size:
            pad = np.repeat(window[-1:], self.window_size - window.shape[0], axis=0)
            window = np.concatenate([window, pad], axis=0)
        return torch.from_numpy(window.astype(np.float32))


def _load_json_tokens(path: Path) -> List[List[int]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    tokens = data.get("tokens")
    if tokens is None:
        raise ValueError(f"Missing 'tokens' in {path}")
    return tokens


def resolve_sequence_path(root: str | Path, name: str, suffix: str) -> Path:
    root = Path(root)
    normalized = name.replace("\\", os.sep).replace("/", os.sep)
    candidates = [
        root / f"{name}{suffix}",
        root / f"{normalized}{suffix}",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"File not found for '{name}' under {root} with suffix {suffix}")


def validate_raw_token_frames(
    tokens: Sequence[Sequence[int]],
    expected_width: int,
    codebook_size: int,
    path: str | Path,
) -> None:
    arr = np.asarray(tokens)
    if arr.ndim != 2 or arr.shape[1] != expected_width:
        raise ValueError(f"Expected token shape (T,{expected_width}), got {arr.shape} in {path}")
    bad = np.argwhere((arr < 0) | (arr >= codebook_size))
    if bad.size > 0:
        frame_idx, residual_idx = bad[0].tolist()
        value = int(arr[frame_idx, residual_idx])
        raise ValueError(
            f"Token out of range in {path}: frame {frame_idx}, residual {residual_idx + 1} "
            f"is {value}, expected [0, {codebook_size - 1}]"
        )


def resolve_face_path(face_dir: str, name: str) -> Path:
    return resolve_sequence_path(face_dir, name, ".npy")


def list_face_names(face_dir: str, split_file: Optional[str] = None) -> List[str]:
    split_names = read_split(split_file)
    if split_names is not None:
        return split_names
    root = Path(face_dir)
    if not root.exists():
        raise FileNotFoundError(f"Face directory not found: {root}")
    return sorted(str(path.relative_to(root)).replace(os.sep, "/")[:-4] for path in root.rglob("*.npy"))


def load_face_array(path: str | Path, face_dim: int = 51) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        arr = arr.item()
    if isinstance(arr, dict):
        for key in ("face", "arkit", "arkit_data", "blendshape", "blendshapes"):
            if key in arr:
                arr = arr[key]
                break
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 2 or arr.shape[1] < face_dim:
        raise ValueError(f"Expected face array shape (T,{face_dim}), got {arr.shape} from {path}")
    return arr[:, :face_dim]


def resample_sequence_to_length(seq: np.ndarray, target_len: int) -> np.ndarray:
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if len(seq) == target_len:
        return seq.astype(np.float32)
    if len(seq) == 1:
        return np.repeat(seq.astype(np.float32), target_len, axis=0)
    old_x = np.linspace(0.0, 1.0, num=len(seq), dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    out = np.empty((target_len, seq.shape[1]), dtype=np.float32)
    for dim in range(seq.shape[1]):
        out[:, dim] = np.interp(new_x, old_x, seq[:, dim])
    return out


def compute_face_stats(
    face_dir: str,
    names: Sequence[str],
    face_dim: int = 51,
    limit_files: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    sums = np.zeros(face_dim, dtype=np.float64)
    sq_sums = np.zeros(face_dim, dtype=np.float64)
    count = 0
    use_names = list(names[:limit_files]) if limit_files else list(names)
    for name in use_names:
        face = load_face_array(resolve_face_path(face_dir, name), face_dim)
        sums += face.sum(axis=0)
        sq_sums += (face.astype(np.float64) ** 2).sum(axis=0)
        count += face.shape[0]
    if count == 0:
        raise ValueError("No frames found while computing face statistics")
    mean = sums / count
    var = np.maximum(sq_sums / count - mean**2, 1e-12)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


@dataclass(frozen=True)
class FaceWindow:
    name: str
    start: int


class FaceWindowDataset(Dataset):
    """Sliding-window ARKit dataset for Face R-VQVAE training."""

    def __init__(
        self,
        face_dir: str,
        names: Sequence[str],
        mean: np.ndarray,
        std: np.ndarray,
        window_size: int = 64,
        stride: int = 32,
        face_dim: int = 51,
        limit_windows: Optional[int] = None,
    ) -> None:
        self.face_dir = face_dir
        self.names = list(names)
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)
        self.window_size = int(window_size)
        self.stride = max(1, int(stride))
        self.face_dim = int(face_dim)
        self.cache: dict[str, np.ndarray] = {}
        self.windows: List[FaceWindow] = []

        for name in self.names:
            face = load_face_array(resolve_face_path(face_dir, name), face_dim)
            length = face.shape[0]
            if length <= self.window_size:
                starts = [0]
            else:
                starts = list(range(0, length - self.window_size + 1, self.stride))
                if starts[-1] != length - self.window_size:
                    starts.append(length - self.window_size)
            self.windows.extend(FaceWindow(name, s) for s in starts)
        if limit_windows is not None:
            self.windows = self.windows[:limit_windows]
        if not self.windows:
            raise ValueError("No face training windows were created")

    def __len__(self) -> int:
        return len(self.windows)

    def _load_face(self, name: str) -> np.ndarray:
        if name not in self.cache:
            face = load_face_array(resolve_face_path(self.face_dir, name), self.face_dim)
            self.cache[name] = (face - self.mean) / self.std
        return self.cache[name]

    def __getitem__(self, idx: int) -> torch.Tensor:
        item = self.windows[idx]
        face = self._load_face(item.name)
        window = face[item.start : item.start + self.window_size]
        if window.shape[0] < self.window_size:
            pad = np.repeat(window[-1:], self.window_size - window.shape[0], axis=0)
            window = np.concatenate([window, pad], axis=0)
        return torch.from_numpy(window.astype(np.float32))


def _pad_audio_window(audio: np.ndarray, start: int, length: int) -> np.ndarray:
    if audio.ndim != 2:
        raise ValueError(f"Expected audio features (T,D), got {audio.shape}")
    if start >= audio.shape[0]:
        return np.repeat(audio[-1:], length, axis=0)
    window = audio[start : start + length]
    if window.shape[0] < length:
        pad = np.repeat(window[-1:], length - window.shape[0], axis=0)
        window = np.concatenate([window, pad], axis=0)
    return window.astype(np.float32)


@dataclass(frozen=True)
class TokenWindow:
    name: str
    start: int


class AudioMotionTokenWindowDataset(Dataset):
    """Masked infill windows aligned with the existing GitHub inference contract.

    `mask_prob` selects candidate positions that receive supervised labels.
    With the default `random_replace_scope="unmasked"`, supervised positions are
    replaced by the mask token, while a subset of unsupervised interior tokens is
    replaced by random same-codebook values, matching the paper's Infill training
    procedure. `legacy_supervised` preserves the earlier BERT-style corruption of
    supervised positions.
    """

    def __init__(
        self,
        motion_token_dir: str,
        audio_feat_dir: str,
        names: Optional[Sequence[str]] = None,
        split_file: Optional[str] = None,
        num_frames: int = 5,
        num_tokens_per_frame: int = 4,
        codebook_size: int = 512,
        mask_prob: float = 0.7,
        random_replace_prob: float = 0.1,
        random_replace_scope: str = "unmasked",
        stride: int = 1,
        boundary_mode: str = "ends",
        length_mismatch_policy: str = "strict",
        limit_windows: Optional[int] = None,
    ) -> None:
        self.motion_token_dir = Path(motion_token_dir)
        self.audio_feat_dir = Path(audio_feat_dir)
        self.num_frames = int(num_frames)
        self.ntpf = int(num_tokens_per_frame)
        self.codebook_size = int(codebook_size)
        self.mask_token_id = self.codebook_size * self.ntpf
        self.mask_prob = float(mask_prob)
        self.random_replace_prob = float(random_replace_prob)
        if random_replace_scope not in {"unmasked", "legacy_supervised"}:
            raise ValueError("random_replace_scope must be 'unmasked' or 'legacy_supervised'")
        self.random_replace_scope = random_replace_scope
        self.stride = max(1, int(stride))
        self.token_cache: dict[str, List[List[int]]] = {}
        self.audio_cache: dict[str, np.ndarray] = {}
        if boundary_mode not in {"ends", "none"}:
            raise ValueError("boundary_mode must be 'ends' or 'none'")
        if length_mismatch_policy not in {"strict", "truncate"}:
            raise ValueError("length_mismatch_policy must be 'strict' or 'truncate'")
        self.boundary_mode = boundary_mode
        self.length_mismatch_policy = length_mismatch_policy

        split_names = list(names) if names is not None else read_split(split_file)
        if split_names is None:
            split_names = sorted(str(p.relative_to(self.motion_token_dir)).replace(os.sep, "/")[:-5] for p in self.motion_token_dir.rglob("*.json"))

        self.names: List[str] = []
        self.windows: List[TokenWindow] = []
        for name in split_names:
            try:
                token_path = resolve_sequence_path(self.motion_token_dir, name, ".json")
                audio_path = resolve_sequence_path(self.audio_feat_dir, name, ".npy")
            except FileNotFoundError:
                continue
            tokens = _load_json_tokens(token_path)
            validate_raw_token_frames(tokens, self.ntpf, self.codebook_size, token_path)
            audio_len = np.load(audio_path, mmap_mode="r").shape[0]
            if len(tokens) != audio_len:
                message = (
                    f"Length mismatch for {name}: motion tokens={len(tokens)}, "
                    f"audio features={audio_len}. Re-run preprocessing with audio-to-motion alignment."
                )
                if self.length_mismatch_policy == "strict":
                    raise ValueError(message)
                print(f"[WARN] {message} Truncating because length_mismatch_policy=truncate.")
            usable_len = min(len(tokens), audio_len)
            if usable_len < self.num_frames:
                continue
            self.names.append(name)
            for start in range(0, usable_len - self.num_frames + 1, self.stride):
                self.windows.append(TokenWindow(name, start))

        if limit_windows is not None:
            self.windows = self.windows[:limit_windows]
        if not self.windows:
            raise ValueError("No infill windows were created")

    def __len__(self) -> int:
        return len(self.windows)

    def _tokens_for(self, name: str) -> List[List[int]]:
        if name not in self.token_cache:
            path = resolve_sequence_path(self.motion_token_dir, name, ".json")
            tokens = _load_json_tokens(path)
            validate_raw_token_frames(tokens, self.ntpf, self.codebook_size, path)
            self.token_cache[name] = tokens
        return self.token_cache[name]

    def _audio_for(self, name: str) -> np.ndarray:
        if name not in self.audio_cache:
            self.audio_cache[name] = np.load(resolve_sequence_path(self.audio_feat_dir, name, ".npy")).astype(np.float32)
        return self.audio_cache[name]

    def _offset_tokens(self, tokens: Sequence[Sequence[int]]) -> torch.Tensor:
        arr = np.asarray(tokens, dtype=np.int64)
        if arr.ndim != 2 or arr.shape[1] != self.ntpf:
            raise ValueError(f"Expected token shape (T,{self.ntpf}), got {arr.shape}")
        validate_raw_token_frames(arr, self.ntpf, self.codebook_size, "infill token window")
        offsets = np.arange(self.ntpf, dtype=np.int64) * self.codebook_size
        return torch.from_numpy(arr + offsets[None, :]).reshape(-1)

    def __getitem__(self, idx: int) -> dict:
        item = self.windows[idx]
        tokens = self._tokens_for(item.name)[item.start : item.start + self.num_frames]
        audio = _pad_audio_window(self._audio_for(item.name), item.start, self.num_frames)

        target = self._offset_tokens(tokens)
        input_ids = target.clone()
        labels = torch.full_like(target, -100)

        if self.boundary_mode == "ends":
            frame_mask = torch.zeros(self.num_frames, self.ntpf, dtype=torch.bool)
            frame_mask[1:-1] = True
        else:
            frame_mask = torch.ones(self.num_frames, self.ntpf, dtype=torch.bool)
        candidate_positions = frame_mask.reshape(-1).nonzero(as_tuple=True)[0]

        supervise_flags = torch.rand(candidate_positions.numel()) < self.mask_prob
        if not bool(supervise_flags.any()):
            supervise_flags[random.randrange(candidate_positions.numel())] = True
        supervised_positions = candidate_positions[supervise_flags]

        labels[supervised_positions] = target[supervised_positions]

        if self.random_replace_scope == "legacy_supervised":
            if supervised_positions.numel() > 0 and self.random_replace_prob > 0:
                replace_flags = torch.rand(supervised_positions.numel()) < self.random_replace_prob
                replace_positions = supervised_positions[replace_flags]
                mask_positions = supervised_positions[~replace_flags]
            else:
                replace_positions = torch.empty(0, dtype=torch.long)
                mask_positions = supervised_positions
        else:
            mask_positions = supervised_positions
            unmasked_positions = candidate_positions[~supervise_flags]
            if unmasked_positions.numel() > 0 and self.random_replace_prob > 0:
                replace_flags = torch.rand(unmasked_positions.numel()) < self.random_replace_prob
                replace_positions = unmasked_positions[replace_flags]
            else:
                replace_positions = torch.empty(0, dtype=torch.long)

        input_ids[mask_positions] = self.mask_token_id
        for pos in replace_positions.tolist():
            quantizer_idx = pos % self.ntpf
            low = quantizer_idx * self.codebook_size
            input_ids[pos] = random.randint(low, low + self.codebook_size - 1)

        return {
            "input_ids": input_ids.long(),
            "labels": labels.long(),
            "audio_features": torch.from_numpy(audio),
        }


def token_group_to_text(tokens: Sequence[Sequence[int]], prefix: str = "res") -> str:
    parts: List[str] = []
    for frame in tokens:
        for i, value in enumerate(frame, start=1):
            parts.append(f"[{prefix}{i}_{int(value)}]")
    return "".join(parts)


def audio_tokens_to_text(audio_tokens: Sequence[int], indices: Iterable[int]) -> str:
    parts: List[str] = []
    for idx in indices:
        if idx < len(audio_tokens):
            token = audio_tokens[idx]
            if isinstance(token, list):
                token = token[0]
            parts.append(f"[audio_{int(token)}]")
    return "".join(parts)


def _first_present(mapping: dict, keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def _tag_value(prefix: str, value: Any) -> str:
    text = normalize_motion_text(value).strip()
    if not text:
        return ""
    if text.startswith(prefix):
        return text
    return f"{prefix}{text}"


def normalize_motion_text(value: Any) -> str:
    """Convert common motion2text entry variants to the released string format."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(part for part in (normalize_motion_text(item) for item in value) if part).strip()
    if isinstance(value, dict):
        action = _first_present(
            value,
            ("action", "action_text", "body_action", "bodyAction", "motion", "motion_text", "动作"),
        )
        expression = _first_present(
            value,
            ("expression", "emotion", "face", "face_expression", "facial_expression", "表情"),
        )
        dialogue = _first_present(
            value,
            ("text", "dialogue", "caption", "transcript", "speech", "content", "utterance", "sentence", "对白"),
        )
        parts: list[str] = []
        expression_text = _tag_value("表情：", expression)
        action_text = _tag_value("动作：", action)
        if expression_text:
            parts.append(f"【{expression_text}】")
        if action_text:
            parts.append(f"【{action_text}】")
        dialogue_text = normalize_motion_text(dialogue).strip()
        if dialogue_text:
            parts.append(dialogue_text)
        if parts:
            return "".join(parts)
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def extract_action_text(raw_text: Any) -> str:
    import re

    raw_text = normalize_motion_text(raw_text)
    matches = re.findall(r"【(.+?)】", raw_text or "")
    if not matches:
        return raw_text or "动作：说话"
    last_tag = matches[-1]
    if last_tag == "动作：无动作":
        for tag in matches:
            if tag.startswith("表情：") and tag != "表情：无表情":
                return "动作：" + tag.replace("表情：", "")
    return last_tag
