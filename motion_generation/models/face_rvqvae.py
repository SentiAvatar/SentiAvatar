#!/usr/bin/env python3
"""Face R-VQVAE used by the paper-style face infill pathway."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .components.residual_vq import ResidualVQ


@dataclass
class FaceRVQVAEConfig:
    input_dim: int = 51
    hidden_size: int = 256
    code_dim: int = 256
    codebook_size: int = 512
    num_quantizers: int = 2
    num_res_blocks: int = 2
    dropout: float = 0.0
    quantize_dropout_prob: float = 0.0
    quantize_dropout_cutoff_index: int = 1


class FaceResBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class FaceRVQVAE(nn.Module):
    """Per-frame ARKit tokenizer with 2 residual tokens per frame by default."""

    def __init__(self, config: FaceRVQVAEConfig) -> None:
        super().__init__()
        self.config = config
        blocks = [
            nn.Conv1d(config.input_dim, config.hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        for _ in range(config.num_res_blocks):
            blocks.append(FaceResBlock(config.hidden_size, config.dropout))
        blocks.append(nn.Conv1d(config.hidden_size, config.code_dim, kernel_size=3, padding=1))
        self.encoder = nn.Sequential(*blocks)

        self.quantizer = ResidualVQ(
            num_quantizers=config.num_quantizers,
            shared_codebook=False,
            quantize_dropout_prob=config.quantize_dropout_prob,
            quantize_dropout_cutoff_index=config.quantize_dropout_cutoff_index,
            nb_code=config.codebook_size,
            code_dim=config.code_dim,
        )

        dec_blocks = [
            nn.Conv1d(config.code_dim, config.hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        for _ in range(config.num_res_blocks):
            dec_blocks.append(FaceResBlock(config.hidden_size, config.dropout))
        dec_blocks.append(nn.Conv1d(config.hidden_size, config.input_dim, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*dec_blocks)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1).float()

    def postprocess(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1)

    def forward(self, x: torch.Tensor):
        z = self.encoder(self.preprocess(x))
        z_q, code_idx, commit_loss, perplexity = self.quantizer(z, sample_codebook_temp=0.5)
        rec = self.postprocess(self.decoder(z_q))
        return rec, commit_loss, perplexity, code_idx

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(self.preprocess(x))
        return self.quantizer.quantize(z)

    @torch.no_grad()
    def decode(self, code_idx: torch.Tensor) -> torch.Tensor:
        z_q = self.quantizer.get_codebook_entry(code_idx)
        return self.postprocess(self.decoder(z_q))


def save_face_rvqvae_checkpoint(
    path: str | Path,
    model: FaceRVQVAE,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    global_step: int,
    mean: np.ndarray,
    std: np.ndarray,
    metrics: Optional[dict] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "config": asdict(model.config),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "metrics": metrics or {},
        },
        path,
    )


def load_face_rvqvae_checkpoint(path: str | Path, device: torch.device):
    path = Path(path)
    if path.is_dir():
        path = path / "latest.pth"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = FaceRVQVAEConfig(**checkpoint["config"])
    model = FaceRVQVAE(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    mean = torch.tensor(checkpoint["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(checkpoint["std"], dtype=torch.float32, device=device)
    return model, mean, std, checkpoint
