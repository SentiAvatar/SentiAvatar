#!/usr/bin/env python3
"""Paper-style face infill inference utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from models.face_rvqvae import load_face_rvqvae_checkpoint
from pipeline_infer import load_mask_transformer, validate_motion_token_frames
from training.data import resample_sequence_to_length, resolve_sequence_path


ARKIT_STANDARD_NAMES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]

MTA_NAMES = [
    "EyeBlinkLeft", "EyeLookDownLeft", "EyeLookInLeft", "EyeLookOutLeft", "EyeLookUpLeft", "EyeSquintLeft",
    "EyeWideLeft", "EyeBlinkRight", "EyeLookDownRight", "EyeLookInRight", "EyeLookOutRight", "EyeLookUpRight",
    "EyeSquintRight", "EyeWideRight", "JawForward", "JawLeft", "JawRight", "JawOpen", "MouthClose", "MouthFunnel",
    "MouthPucker", "MouthLeft", "MouthRight", "MouthSmileLeft", "MouthSmileRight", "MouthFrownLeft", "MouthFrownRight",
    "MouthDimpleLeft", "MouthDimpleRight", "MouthStretchLeft", "MouthStretchRight", "MouthRollLower", "MouthRollUpper",
    "MouthShrugLower", "MouthShrugUpper", "MouthPressLeft", "MouthPressRight", "MouthLowerDownLeft", "MouthLowerDownRight",
    "MouthUpperUpLeft", "MouthUpperUpRight", "BrowDownLeft", "BrowDownRight", "BrowInnerUp", "BrowOuterUpLeft",
    "BrowOuterUpRight", "CheekPuff", "CheekSquintLeft", "CheekSquintRight", "NoseSneerLeft", "NoseSneerRight", "TongueOut",
]


def arkit51_to_mta52(face_arkit: np.ndarray) -> np.ndarray:
    face_arkit = np.asarray(face_arkit, dtype=np.float32)
    lower = [name.lower() for name in ARKIT_STANDARD_NAMES]
    out = np.zeros((face_arkit.shape[0], len(MTA_NAMES)), dtype=np.float32)
    for idx, name in enumerate(MTA_NAMES):
        key = name.lower()
        if key in lower:
            out[:, idx] = face_arkit[:, lower.index(key)]
    return out


def add_face_to_anim(anim_data: Dict, face_mta52: np.ndarray) -> Dict:
    expect_frames = len(anim_data["frames"])
    face = resample_sequence_to_length(np.asarray(face_mta52, dtype=np.float32), expect_frames)
    for idx in range(expect_frames):
        anim_data["frames"][idx]["face"] = [round(float(v), 4) for v in face[idx]]
    return anim_data


class FaceInfillPipeline:
    def __init__(
        self,
        face_rvqvae_ckpt: str,
        face_infill_ckpt: str,
        device: str | torch.device = "cuda:0",
        generate_steps: int = 6,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() or "cuda" not in str(device) else "cpu")
        self.face_vqvae, self.mean, self.std, _ = load_face_rvqvae_checkpoint(face_rvqvae_ckpt, self.device)
        self.infill = load_mask_transformer(face_infill_ckpt, device=str(self.device))
        if int(self.infill.config.num_tokens_per_frame) != int(self.face_vqvae.config.num_quantizers):
            raise ValueError(
                "Face Infill num_tokens_per_frame "
                f"({self.infill.config.num_tokens_per_frame}) must match Face R-VQVAE num_quantizers "
                f"({self.face_vqvae.config.num_quantizers})"
            )
        if int(self.infill.config.codebook_size) != int(self.face_vqvae.config.codebook_size):
            raise ValueError(
                "Face Infill codebook_size "
                f"({self.infill.config.codebook_size}) must match Face R-VQVAE codebook_size "
                f"({self.face_vqvae.config.codebook_size})"
            )
        self.generate_steps = int(generate_steps)

    @torch.no_grad()
    def generate_tokens(self, audio_features: np.ndarray) -> List[List[int]]:
        cfg = self.infill.config
        ntpf = int(cfg.num_tokens_per_frame)
        codebook_size = int(cfg.codebook_size)
        mask_token_id = int(cfg.vocab_size) - 1
        num_frames = int(cfg.num_frames)
        audio = np.asarray(audio_features, dtype=np.float32)
        if audio.ndim != 2:
            raise ValueError(f"Expected audio features shape (T,D), got {audio.shape}")
        if audio.shape[0] == 0:
            raise ValueError("Expected at least one audio feature frame for face infill")

        tokens: List[List[int]] = []
        for start in range(0, audio.shape[0], num_frames):
            window = audio[start : start + num_frames]
            if window.shape[0] < num_frames:
                pad = np.repeat(window[-1:], num_frames - window.shape[0], axis=0)
                window = np.concatenate([window, pad], axis=0)
            input_ids = torch.full((1, num_frames * ntpf), mask_token_id, dtype=torch.long, device=self.device)
            audio_tensor = torch.tensor(window, dtype=torch.float32, device=self.device).unsqueeze(0)
            output = self.infill.generate_sbs(input_ids, audio_tensor, generate_steps=self.generate_steps)[0].cpu().tolist()
            for frame_idx in range(num_frames):
                raw = []
                for q in range(ntpf):
                    token_id = output[frame_idx * ntpf + q]
                    raw.append(token_id - q * codebook_size)
                tokens.append(raw)
        tokens = tokens[: audio.shape[0]]
        validate_motion_token_frames(tokens, ntpf, codebook_size, context="Face Infill output")
        return tokens

    @torch.no_grad()
    def decode_tokens_to_arkit(self, tokens: List[List[int]]) -> np.ndarray:
        idx = torch.tensor(np.asarray(tokens), dtype=torch.long, device=self.device).unsqueeze(0)
        face = self.face_vqvae.decode(idx)[0]
        face = face * self.std + self.mean
        return face.detach().cpu().numpy().astype(np.float32)

    def infer_arkit(self, audio_features: np.ndarray) -> np.ndarray:
        return self.decode_tokens_to_arkit(self.generate_tokens(audio_features))

    def infer_mta52(self, audio_features: np.ndarray) -> np.ndarray:
        return arkit51_to_mta52(self.infer_arkit(audio_features))


def load_audio_features_for_name(audio_feat_dir: Optional[str], name: str) -> Optional[np.ndarray]:
    if not audio_feat_dir:
        return None
    try:
        path = resolve_sequence_path(audio_feat_dir, name, ".npy")
        return np.load(path).astype(np.float32)
    except FileNotFoundError:
        pass
    return None
