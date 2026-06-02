#!/usr/bin/env python3
"""Train the audio-aware infill transformer used by pipeline_infer.py."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from models.audio_motion_model import AudioMotionConfig, AudioMotionTransformer
from training.data import AudioMotionTokenWindowDataset, set_seed


def collate(batch: list[dict]) -> dict:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
        "audio_features": torch.stack([x["audio_features"] for x in batch]),
    }


def save_model(model: AudioMotionTransformer, output_dir: Path, args: argparse.Namespace, metrics: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    (output_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (output_dir / "trainer_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")

    dataset = AudioMotionTokenWindowDataset(
        motion_token_dir=args.motion_token_dir,
        audio_feat_dir=args.audio_feat_dir,
        split_file=args.train_split,
        num_frames=args.num_frames,
        num_tokens_per_frame=args.num_tokens_per_frame,
        codebook_size=args.codebook_size,
        mask_prob=args.mask_prob,
        random_replace_prob=args.random_replace_prob,
        random_replace_scope=args.random_replace_scope,
        stride=args.window_stride,
        boundary_mode=args.boundary_mode,
        length_mismatch_policy=args.length_mismatch_policy,
        limit_windows=args.limit_windows,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=args.drop_last,
        collate_fn=collate,
    )

    if args.init_ckpt:
        config = AudioMotionConfig.from_pretrained(args.init_ckpt)
        model = AudioMotionTransformer.from_pretrained(args.init_ckpt, config=config)
    else:
        config = AudioMotionConfig(
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            intermediate_size=args.intermediate_size,
            max_position_embeddings=args.max_position_embeddings,
            vocab_size=args.codebook_size * args.num_tokens_per_frame + 1,
            codebook_size=args.codebook_size,
            audio_feat_dim=args.audio_feat_dim,
            num_tokens_per_frame=args.num_tokens_per_frame,
            num_frames=args.num_frames,
            dropout=args.dropout,
            cond_drop_prob=args.cond_drop_prob,
        )
        model = AudioMotionTransformer(config)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.min_lr)

    output_dir = Path(args.output_dir)
    global_step = 0
    latest_metrics = {}
    for epoch in range(1, args.epochs + 1):
        progress = tqdm(loader, desc=f"infill epoch {epoch}/{args.epochs}")
        running_loss = 0.0
        running_acc = 0.0
        for batch in progress:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            audio_features = batch["audio_features"].to(device, non_blocking=True)

            loss, _, acc = model(input_ids=input_ids, labels=labels, audio_features=audio_features)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            global_step += 1
            running_loss += float(loss.detach().cpu())
            running_acc += float(acc.detach().cpu())
            latest_metrics = {
                "epoch": epoch,
                "global_step": global_step,
                "loss": float(loss.detach().cpu()),
                "accuracy": float(acc.detach().cpu()),
                "lr": scheduler.get_last_lr()[0],
            }
            if global_step % args.log_every == 0:
                progress.set_postfix(loss=f"{latest_metrics['loss']:.4f}", acc=f"{latest_metrics['accuracy']:.4f}")
            if args.save_steps and global_step % args.save_steps == 0:
                save_model(model, output_dir / f"checkpoint-{global_step}", args, latest_metrics)
                save_model(model, output_dir, args, latest_metrics)
            if args.max_steps and global_step >= args.max_steps:
                break

        epoch_metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "loss": running_loss / max(1, len(loader)),
            "accuracy": running_acc / max(1, len(loader)),
            "lr": scheduler.get_last_lr()[0],
        }
        if epoch % args.save_every_epochs == 0 or epoch == args.epochs:
            save_model(model, output_dir / f"checkpoint-epoch-{epoch}", args, epoch_metrics)
        save_model(model, output_dir, args, epoch_metrics)
        if args.max_steps and global_step >= args.max_steps:
            break

    print(f"Saved Infill Transformer to {output_dir}")
    print(f"Use with: python motion_generation/pipeline_infer.py --mask_ckpt {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SentiAvatar audio-aware infill transformer")
    parser.add_argument("--motion_token_dir", default="./data/motion_token_data")
    parser.add_argument("--audio_feat_dir", default="./data/audio_features_hubert_layer9_fps10")
    parser.add_argument("--train_split", default="./data/split/train_file_list.txt")
    parser.add_argument("--output_dir", default="./checkpoints_train/mask_transformer")
    parser.add_argument("--init_ckpt", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--num_tokens_per_frame", type=int, default=4)
    parser.add_argument("--codebook_size", type=int, default=512)
    parser.add_argument("--audio_feat_dim", type=int, default=768)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--intermediate_size", type=int, default=1536)
    parser.add_argument("--max_position_embeddings", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--cond_drop_prob", type=float, default=0.2)
    parser.add_argument("--mask_prob", type=float, default=0.7,
                        help="Probability that an eligible token position is supervised for infill")
    parser.add_argument("--random_replace_prob", type=float, default=0.1,
                        help="Probability of replacing eligible tokens with random same-codebook values")
    parser.add_argument("--random_replace_scope", choices=["unmasked", "legacy_supervised"], default="unmasked",
                        help="unmasked matches the paper: corrupt unsupervised interior tokens; legacy_supervised corrupts supervised inputs")
    parser.add_argument("--boundary_mode", choices=["ends", "none"], default="ends")
    parser.add_argument("--length_mismatch_policy", choices=["strict", "truncate"], default="strict",
                        help="How to handle motion-token/audio-feature length mismatches; strict matches GitHub inference")
    parser.add_argument("--window_stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--drop_last", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_every_epochs", type=int, default=1)
    parser.add_argument("--limit_windows", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
