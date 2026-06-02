#!/usr/bin/env python3
"""Full-parameter SFT for the Qwen motion planner."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from training.data import set_seed


def planner_tokens(codebook_size: int, num_quantizers: int, max_audio_token: int, max_len_token: int, steps: list[int]) -> list[str]:
    tokens = []
    tokens.extend(f"[audio_{i}]" for i in range(max_audio_token + 1))
    for q in range(1, num_quantizers + 1):
        tokens.extend(f"[res{q}_{i}]" for i in range(codebook_size))
    tokens.extend(f"[len_{i}]" for i in range(1, max_len_token + 1))
    tokens.extend(f"[step_{step}]" for step in steps)
    return tokens


def discover_step_tokens(jsonl_path: str) -> list[int]:
    steps: set[int] = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "step" in row:
                steps.add(int(row["step"]))
            for match in re.findall(r"\[step_(\d+)\]", row.get("completion", "")):
                steps.add(int(match))
    return sorted(steps)


def collect_bracket_planner_tokens(jsonl_path: str) -> set[str]:
    pattern = re.compile(r"\[(?:audio|res\d+|len|step)_\d+\]")
    tokens: set[str] = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            tokens.update(pattern.findall(row.get("prompt", "")))
            tokens.update(pattern.findall(row.get("completion", "")))
    return tokens


def validate_planner_token_coverage(tokenizer, jsonl_path: str) -> None:
    """Ensure all discrete planner tokens are represented as atomic tokenizer IDs."""
    vocab = tokenizer.get_vocab()
    missing: list[str] = []
    split: list[tuple[str, list[int]]] = []
    for token in sorted(collect_bracket_planner_tokens(jsonl_path)):
        if token not in vocab:
            missing.append(token)
            continue
        ids = tokenizer(token, add_special_tokens=False).input_ids
        if len(ids) != 1:
            split.append((token, ids))
    if missing or split:
        details: list[str] = []
        if missing:
            details.append("missing tokens: " + ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""))
        if split:
            formatted = ", ".join(f"{tok}->{ids}" for tok, ids in split[:10])
            details.append("non-atomic tokens: " + formatted + (" ..." if len(split) > 10 else ""))
        raise ValueError(
            "Planner tokenizer does not cover all discrete [audio]/[res]/[len]/[step] tokens. "
            + "; ".join(details)
            + ". Increase --max_audio_token/--max_len_token or enable --add_planner_tokens."
        )


def validate_planner_jsonl_contract(
    jsonl_path: str,
    codebook_size: int,
    num_quantizers: int,
    max_audio_token: int,
    max_len_token: int,
) -> None:
    """Validate prompt/completion structure before expensive model loading."""
    bracket_re = re.compile(r"\[([a-zA-Z0-9]+)_(\d+)\]")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            name = row.get("name", f"line_{line_idx}")
            prompt = row.get("prompt")
            completion = row.get("completion")
            if not isinstance(prompt, str) or not isinstance(completion, str):
                raise ValueError(f"LLM SFT row '{name}' must contain string prompt and completion")

            for token_type, raw_value in bracket_re.findall(prompt):
                value = int(raw_value)
                if token_type == "audio" and (value < 0 or value > max_audio_token):
                    raise ValueError(f"Prompt audio token out of range in '{name}': [audio_{value}]")
                if token_type.startswith("res"):
                    q = int(token_type[3:])
                    if q < 1 or q > num_quantizers or value < 0 or value >= codebook_size:
                        raise ValueError(f"Prompt residual token out of range in '{name}': [{token_type}_{value}]")

            tokens = bracket_re.findall(completion)
            if len(tokens) < 2:
                raise ValueError(f"Completion for '{name}' must start with [step_t][len_N]")
            if tokens[0][0] != "step":
                raise ValueError(f"Completion for '{name}' must start with [step_t], got [{tokens[0][0]}_{tokens[0][1]}]")
            if tokens[1][0] != "len":
                raise ValueError(f"Completion for '{name}' must contain [len_N] immediately after [step_t]")
            step = int(tokens[0][1])
            target_len = int(tokens[1][1])
            if target_len < 1 or target_len > max_len_token:
                raise ValueError(f"Completion [len_{target_len}] out of range in '{name}', expected [1,{max_len_token}]")
            if "step" in row and int(row["step"]) != step:
                raise ValueError(f"Row '{name}' step metadata {row['step']} does not match completion [step_{step}]")
            if "num_keyframes" in row and int(row["num_keyframes"]) != target_len:
                raise ValueError(
                    f"Row '{name}' num_keyframes={row['num_keyframes']} does not match completion [len_{target_len}]"
                )

            residual_tokens = tokens[2:]
            expected_count = target_len * num_quantizers
            if len(residual_tokens) != expected_count:
                raise ValueError(
                    f"Completion for '{name}' has {len(residual_tokens)} residual tokens, "
                    f"expected {expected_count} for len={target_len}, num_quantizers={num_quantizers}"
                )
            for pos, (token_type, raw_value) in enumerate(residual_tokens):
                expected_q = pos % num_quantizers + 1
                if token_type != f"res{expected_q}":
                    raise ValueError(
                        f"Completion residual order mismatch in '{name}' at position {pos}: "
                        f"expected [res{expected_q}_*], got [{token_type}_{raw_value}]"
                    )
                value = int(raw_value)
                if value < 0 or value >= codebook_size:
                    raise ValueError(
                        f"Completion residual token out of range in '{name}': "
                        f"[{token_type}_{value}], expected [0,{codebook_size - 1}]"
                    )


class SFTJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int, truncate_long_examples: bool = False) -> None:
        self.rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))
        if not self.rows:
            raise ValueError(f"No rows found in {jsonl_path}")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncate_long_examples = truncate_long_examples

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        prompt_ids = self.tokenizer(row["prompt"], add_special_tokens=False).input_ids
        completion_ids = self.tokenizer(row["completion"], add_special_tokens=False).input_ids
        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        if len(input_ids) > self.max_length:
            if not self.truncate_long_examples:
                name = row.get("name", f"row_{idx}")
                raise ValueError(
                    f"LLM SFT example '{name}' has {len(input_ids)} tokens "
                    f"(prompt={len(prompt_ids)}, completion={len(completion_ids)}), "
                    f"which exceeds --max_length={self.max_length}. Increase --max_length "
                    "or pass --truncate_long_examples to allow tail truncation."
                )
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]
        if not any(label != -100 for label in labels):
            name = row.get("name", f"row_{idx}")
            raise ValueError(
                f"LLM SFT example '{name}' has no supervised completion tokens after tokenization/truncation. "
                "Increase --max_length or rebuild shorter planner examples."
            )
        return {"input_ids": input_ids, "labels": labels}


def make_collate(tokenizer):
    pad_id = tokenizer.pad_token_id

    def collate(batch: list[dict]) -> dict:
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = []
        labels = []
        attention_mask = []
        for item in batch:
            pad = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * pad)
            labels.append(item["labels"] + [-100] * pad)
            attention_mask.append([1] * len(item["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    return collate


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.init_model, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
    if args.validate_planner_jsonl_contract:
        validate_planner_jsonl_contract(
            args.train_jsonl,
            codebook_size=args.codebook_size,
            num_quantizers=args.num_quantizers,
            max_audio_token=args.max_audio_token,
            max_len_token=args.max_len_token,
        )

    if args.add_planner_tokens:
        configured_steps = {int(x) for x in args.step_tokens.split(",") if x}
        configured_steps.update(discover_step_tokens(args.train_jsonl))
        extra_tokens = planner_tokens(
            codebook_size=args.codebook_size,
            num_quantizers=args.num_quantizers,
            max_audio_token=args.max_audio_token,
            max_len_token=args.max_len_token,
            steps=sorted(configured_steps),
        )
        tokenizer.add_tokens([tok for tok in extra_tokens if tok not in tokenizer.get_vocab()])
    if args.validate_planner_token_coverage:
        validate_planner_token_coverage(tokenizer, args.train_jsonl)

    model = AutoModelForCausalLM.from_pretrained(
        args.init_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 and torch.cuda.is_available() else None,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.train()

    dataset = SFTJsonlDataset(
        args.train_jsonl,
        tokenizer,
        args.max_length,
        truncate_long_examples=args.truncate_long_examples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=make_collate(tokenizer),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    max_micro_steps = args.max_steps or args.epochs * len(loader)
    total_optimizer_steps = max(1, math.ceil(max_micro_steps / args.grad_accum_steps))
    warmup_steps = min(args.warmup_steps, total_optimizer_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_optimizer_steps)

    output_dir = Path(args.output_dir)
    global_step = 0
    optimizer_step = 0
    pending_accum = 0
    accum_target = args.grad_accum_steps
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        progress = tqdm(loader, desc=f"llm sft epoch {epoch}/{args.epochs}")
        for batch in progress:
            if pending_accum == 0:
                accum_target = min(args.grad_accum_steps, max_micro_steps - global_step)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / max(1, accum_target)
            loss.backward()
            pending_accum += 1
            should_step = pending_accum >= accum_target
            if should_step:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                pending_accum = 0
            global_step += 1
            if global_step % args.log_every == 0:
                progress.set_postfix(loss=f"{float(outputs.loss.detach().cpu()):.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
            if should_step and args.save_steps and optimizer_step % args.save_steps == 0:
                ckpt_dir = output_dir / f"checkpoint-{optimizer_step}"
                model.save_pretrained(ckpt_dir, safe_serialization=True)
                tokenizer.save_pretrained(ckpt_dir)
            if args.max_steps and global_step >= args.max_steps:
                break
        if args.max_steps and global_step >= args.max_steps:
            break

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"Saved planner model to {output_dir}")
    print(f"Use with: bash scripts/start_vllm_server.sh {output_dir} 8095 0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen motion planner with full-parameter SFT")
    parser.add_argument("--init_model", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--train_jsonl", default="./data/llm_sft/train_step4.jsonl")
    parser.add_argument("--output_dir", default="./checkpoints_train/llm")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max_length", type=int, default=1600)
    parser.add_argument("--truncate_long_examples", action="store_true",
                        help="Allow right truncation of examples longer than --max_length; default is to fail fast")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--add_planner_tokens", action="store_true")
    parser.add_argument("--no_validate_planner_jsonl_contract", dest="validate_planner_jsonl_contract",
                        action="store_false",
                        help="Skip fail-fast validation of [step]/[len]/residual planner JSONL structure")
    parser.add_argument("--no_validate_planner_token_coverage", dest="validate_planner_token_coverage",
                        action="store_false",
                        help="Skip fail-fast validation that JSONL planner tokens are atomic tokenizer IDs")
    parser.set_defaults(validate_planner_jsonl_contract=True, validate_planner_token_coverage=True)
    parser.add_argument("--codebook_size", type=int, default=512)
    parser.add_argument("--num_quantizers", type=int, default=4)
    parser.add_argument("--max_audio_token", type=int, default=2048)
    parser.add_argument("--max_len_token", type=int, default=2048)
    parser.add_argument("--step_tokens", default="1,2,3,4,5,6,8")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
