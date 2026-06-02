#!/usr/bin/env python3
"""Train the body Motion R-VQVAE used by GitHub inference."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from configs.default_config import Config
from models.rvqvae import RVQVAE
from training.data import (
    BodyMotionWindowDataset,
    compute_body_stats,
    list_motion_names,
    set_seed,
)


def build_config(args: argparse.Namespace) -> Config:
    config = Config(
        name=args.name,
        dataset_name=args.dataset_name,
        checkpoints_dir=args.checkpoints_dir,
        log_dir=args.log_dir,
        gpu_id=args.gpu_id,
        seed=args.seed,
        debug=args.debug,
    )
    config.data.data_root = args.data_root
    config.data.body_dim = args.body_dim
    config.data.window_size = args.window_size
    config.data.batch_size = args.batch_size
    config.data.num_workers = args.num_workers
    config.data.fps = args.motion_fps
    config.model.nb_code = args.nb_code
    config.model.code_dim = args.code_dim
    config.model.down_t = args.down_t
    config.model.stride_t = args.stride_t
    config.model.width = args.width
    config.model.depth = args.depth
    config.model.dilation_growth_rate = args.dilation_growth_rate
    config.model.vq_act = args.vq_act
    config.model.vq_norm = args.vq_norm
    config.model.vq_cnn_depth = args.vq_cnn_depth
    config.model.num_quantizers = args.num_quantizers
    config.model.shared_codebook = args.shared_codebook
    config.model.quantize_dropout_prob = args.quantize_dropout_prob
    config.model.quantize_dropout_cutoff_index = args.quantize_dropout_cutoff_index
    config.train.max_epoch = args.epochs
    config.train.lr = args.lr
    config.train.weight_decay = args.weight_decay
    config.train.commit = args.commit_weight
    config.train.loss_vel = args.velocity_weight
    config.train.weight_rec = args.reconstruction_weight
    config.train.recons_loss = args.recons_loss
    config.train.log_every = args.log_every
    config.train.save_latest = args.save_latest
    config.train.save_every_e = args.save_every_e
    config.__post_init__()
    return config


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "l1":
        return F.l1_loss(pred, target)
    if mode == "mse":
        return F.mse_loss(pred, target)
    if mode == "l1_smooth":
        return F.smooth_l1_loss(pred, target)
    raise ValueError(f"Unsupported reconstruction loss: {mode}")


def root_position_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_pos = torch.cumsum(pred[..., :3], dim=1)
    tgt_pos = torch.cumsum(target[..., :3], dim=1)
    return F.smooth_l1_loss(pred_pos, tgt_pos)


def save_checkpoint(
    path: Path,
    model: RVQVAE,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "metrics": metrics,
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    config = build_config(args)
    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")

    motion_dir = os.path.join(args.data_root, "motion_data") if args.motion_dir is None else args.motion_dir
    names = list_motion_names(motion_dir, args.train_split)
    if args.limit_files:
        names = names[: args.limit_files]
    if not names:
        raise ValueError("No motion files found for RVQVAE training")

    mean, std = compute_body_stats(motion_dir, names, body_dim=args.body_dim, limit_files=args.limit_files)
    Path(config.meta_dir).mkdir(parents=True, exist_ok=True)
    Path(config.model_dir).mkdir(parents=True, exist_ok=True)
    config.save_opt()
    torch.save(vars(args), Path(config.save_root) / "train_args.pt")
    (Path(config.save_root) / "train_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    import numpy as np

    np.save(Path(config.meta_dir) / "mean.npy", mean)
    np.save(Path(config.meta_dir) / "std.npy", std)

    dataset = BodyMotionWindowDataset(
        motion_dir=motion_dir,
        names=names,
        mean=mean,
        std=std,
        window_size=args.window_size,
        stride=args.window_stride,
        body_dim=args.body_dim,
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

    model = RVQVAE(
        config=config,
        input_dim=config.data.whole_dim,
        nb_code=config.model.nb_code,
        code_dim=config.model.code_dim,
        output_dim=config.model.code_dim,
        down_t=config.model.down_t,
        stride_t=config.model.stride_t,
        width=config.model.width,
        depth=config.model.depth,
        dilation_growth_rate=config.model.dilation_growth_rate,
        activation=config.model.vq_act,
        norm=config.model.vq_norm,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.min_lr)

    global_step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        totals = {"loss": 0.0, "rec": 0.0, "vel": 0.0, "pos": 0.0, "commit": 0.0, "perplexity": 0.0}
        progress = tqdm(loader, desc=f"rvqvae epoch {epoch}/{args.epochs}")
        for batch in progress:
            batch = batch.to(device, non_blocking=True)
            pred, commit_loss, perplexity = model(batch)
            if pred.shape[1] != batch.shape[1]:
                min_len = min(pred.shape[1], batch.shape[1])
                pred = pred[:, :min_len]
                batch = batch[:, :min_len]

            rec = reconstruction_loss(pred, batch, args.recons_loss)
            vel = F.smooth_l1_loss(pred[:, 1:] - pred[:, :-1], batch[:, 1:] - batch[:, :-1])
            pos = root_position_loss(pred, batch)
            loss = (
                args.reconstruction_weight * rec
                + args.velocity_weight * vel
                + args.position_weight * pos
                + args.commit_weight * commit_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            global_step += 1
            metrics = {
                "loss": float(loss.detach().cpu()),
                "rec": float(rec.detach().cpu()),
                "vel": float(vel.detach().cpu()),
                "pos": float(pos.detach().cpu()),
                "commit": float(commit_loss.detach().cpu()),
                "perplexity": float(perplexity.detach().cpu()),
                "lr": scheduler.get_last_lr()[0],
            }
            for key in totals:
                totals[key] += metrics[key]
            if global_step % args.log_every == 0:
                progress.set_postfix({k: f"{v:.4f}" for k, v in metrics.items() if k != "lr"})
            if args.save_latest and global_step % args.save_latest == 0:
                save_checkpoint(Path(config.model_dir) / "latest.pth", model, optimizer, scheduler, epoch, global_step, metrics)
            if args.max_steps and global_step >= args.max_steps:
                break

        denom = max(1, min(len(loader), global_step if args.max_steps else len(loader)))
        epoch_metrics = {k: v / denom for k, v in totals.items()}
        if epoch % args.save_every_e == 0 or epoch == args.epochs:
            save_checkpoint(Path(config.model_dir) / f"epoch_{epoch}.pth", model, optimizer, scheduler, epoch, global_step, epoch_metrics)
        save_checkpoint(Path(config.model_dir) / "latest.pth", model, optimizer, scheduler, epoch, global_step, epoch_metrics)
        if args.max_steps and global_step >= args.max_steps:
            break

    print(f"Saved RVQVAE experiment to {config.save_root}")
    print(f"Inference checkpoint example: {Path(config.model_dir) / 'latest.pth'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Motion R-VQVAE for SentiAvatar")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--motion_dir", default=None)
    parser.add_argument("--train_split", default=None)
    parser.add_argument("--checkpoints_dir", default="./checkpoints_train")
    parser.add_argument("--log_dir", default="./logs")
    parser.add_argument("--dataset_name", default="susuinteracts")
    parser.add_argument("--name", default="rvqvae_body")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--body_dim", type=int, default=153)
    parser.add_argument("--motion_fps", type=int, default=20)
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--window_stride", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--drop_last", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--reconstruction_weight", type=float, default=5.0)
    parser.add_argument("--velocity_weight", type=float, default=50.0)
    parser.add_argument("--position_weight", type=float, default=1.0)
    parser.add_argument("--commit_weight", type=float, default=0.02)
    parser.add_argument("--recons_loss", choices=["l1_smooth", "l1", "mse"], default="l1_smooth")
    parser.add_argument("--nb_code", type=int, default=512)
    parser.add_argument("--code_dim", type=int, default=512)
    parser.add_argument("--down_t", type=int, default=1)
    parser.add_argument("--stride_t", type=int, default=2)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dilation_growth_rate", type=int, default=3)
    parser.add_argument("--vq_act", default="relu")
    parser.add_argument("--vq_norm", default=None)
    parser.add_argument("--vq_cnn_depth", type=int, default=3)
    parser.add_argument("--num_quantizers", type=int, default=4)
    parser.add_argument("--shared_codebook", action="store_true")
    parser.add_argument("--quantize_dropout_prob", type=float, default=0.8)
    parser.add_argument("--quantize_dropout_cutoff_index", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_latest", type=int, default=500)
    parser.add_argument("--save_every_e", type=int, default=2)
    parser.add_argument("--limit_files", type=int, default=None)
    parser.add_argument("--limit_windows", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

