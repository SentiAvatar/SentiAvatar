#!/usr/bin/env python3
"""Train the paper-style Face R-VQVAE tokenizer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from models.face_rvqvae import FaceRVQVAE, FaceRVQVAEConfig, save_face_rvqvae_checkpoint
from training.data import FaceWindowDataset, compute_face_stats, list_face_names, set_seed


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    face_dir = args.face_dir or str(Path(args.data_root) / "arkit_data")
    names = list_face_names(face_dir, args.train_split)
    if args.limit_files:
        names = names[: args.limit_files]
    if not names:
        raise ValueError("No face files found for training")

    mean, std = compute_face_stats(face_dir, names, face_dim=args.face_dim, limit_files=args.limit_files)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    np.save(output_dir / "mean.npy", mean)
    np.save(output_dir / "std.npy", std)

    dataset = FaceWindowDataset(
        face_dir=face_dir,
        names=names,
        mean=mean,
        std=std,
        window_size=args.window_size,
        stride=args.window_stride,
        face_dim=args.face_dim,
        limit_windows=args.limit_windows,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=args.drop_last,
    )

    config = FaceRVQVAEConfig(
        input_dim=args.face_dim,
        hidden_size=args.hidden_size,
        code_dim=args.code_dim,
        codebook_size=args.codebook_size,
        num_quantizers=args.num_quantizers,
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
        quantize_dropout_prob=args.quantize_dropout_prob,
        quantize_dropout_cutoff_index=args.quantize_dropout_cutoff_index,
    )
    model = FaceRVQVAE(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.min_lr)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(loader, desc=f"face rvqvae epoch {epoch}/{args.epochs}")
        latest_metrics = {}
        for batch in progress:
            batch = batch.to(device, non_blocking=True)
            rec, commit_loss, perplexity, _ = model(batch)
            rec_loss = F.smooth_l1_loss(rec, batch)
            vel_loss = F.smooth_l1_loss(rec[:, 1:] - rec[:, :-1], batch[:, 1:] - batch[:, :-1])
            loss = args.reconstruction_weight * rec_loss + args.velocity_weight * vel_loss + args.commit_weight * commit_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            global_step += 1
            latest_metrics = {
                "epoch": epoch,
                "global_step": global_step,
                "loss": float(loss.detach().cpu()),
                "rec": float(rec_loss.detach().cpu()),
                "vel": float(vel_loss.detach().cpu()),
                "commit": float(commit_loss.detach().cpu()),
                "perplexity": float(perplexity.detach().cpu()),
                "lr": scheduler.get_last_lr()[0],
            }
            if global_step % args.log_every == 0:
                progress.set_postfix(loss=f"{latest_metrics['loss']:.4f}", rec=f"{latest_metrics['rec']:.4f}")
            if args.save_steps and global_step % args.save_steps == 0:
                save_face_rvqvae_checkpoint(output_dir / "latest.pth", model, optimizer, epoch, global_step, mean, std, latest_metrics)
            if args.max_steps and global_step >= args.max_steps:
                break

        if epoch % args.save_every_epochs == 0 or epoch == args.epochs:
            save_face_rvqvae_checkpoint(output_dir / f"epoch_{epoch}.pth", model, optimizer, epoch, global_step, mean, std, latest_metrics)
        save_face_rvqvae_checkpoint(output_dir / "latest.pth", model, optimizer, epoch, global_step, mean, std, latest_metrics)
        if args.max_steps and global_step >= args.max_steps:
            break

    print(f"Saved Face R-VQVAE to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Face R-VQVAE")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--face_dir", default=None)
    parser.add_argument("--train_split", default="./data/split/train_file_list.txt")
    parser.add_argument("--output_dir", default="./checkpoints_train/face_rvqvae")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--face_dim", type=int, default=51)
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--window_stride", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--drop_last", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--reconstruction_weight", type=float, default=1.0)
    parser.add_argument("--velocity_weight", type=float, default=1.0)
    parser.add_argument("--commit_weight", type=float, default=0.02)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--code_dim", type=int, default=256)
    parser.add_argument("--codebook_size", type=int, default=512)
    parser.add_argument("--num_quantizers", type=int, default=2)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--quantize_dropout_prob", type=float, default=0.0)
    parser.add_argument("--quantize_dropout_cutoff_index", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_every_epochs", type=int, default=2)
    parser.add_argument("--limit_files", type=int, default=None)
    parser.add_argument("--limit_windows", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

