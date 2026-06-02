#!/usr/bin/env python3
"""Build JSONL prompt/completion data for the LLM motion planner."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from training.data import (
    audio_tokens_to_text,
    extract_action_text,
    normalize_motion_text,
    read_split,
    resolve_sequence_path,
    token_group_to_text,
    validate_raw_token_frames,
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_manifest_path(value: str | None, base_dir: Path) -> str | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return str(path)


def validate_audio_tokens(audio_tokens: list[int], max_audio_token: int, path: Path) -> None:
    for idx, token in enumerate(audio_tokens):
        if isinstance(token, list):
            if len(token) != 1:
                raise ValueError(f"Audio token at frame {idx} in {path} is nested with width {len(token)}; expected scalar")
            token = token[0]
        value = int(token)
        if value < 0 or value > max_audio_token:
            raise ValueError(
                f"Audio token out of range in {path}: frame {idx} is {value}, "
                f"expected [0, {max_audio_token}]"
            )


def audio_motion_prefix_to_text(
    audio_tokens: list[int],
    motion_tokens: list[list[int]],
    indices: list[int],
    prefix_count: int,
) -> str:
    parts: list[str] = []
    for idx, frame_tokens in zip(indices[:prefix_count], motion_tokens[:prefix_count]):
        if idx < len(audio_tokens):
            token = audio_tokens[idx]
            if isinstance(token, list):
                token = token[0]
            parts.append(f"[audio_{int(token)}]")
        parts.append(token_group_to_text([frame_tokens]))
    return "".join(parts)


def build_example(
    name: str,
    action_text: str,
    motion_tokens: list[list[int]],
    audio_tokens: list[int] | None,
    step: int,
    template: str,
    use_audio: bool = True,
    continuation_prefix_keyframes: int = 0,
    max_len_token: int = 2048,
    length_mismatch_policy: str = "strict",
) -> dict:
    if use_audio and audio_tokens is not None:
        if len(motion_tokens) != len(audio_tokens):
            message = (
                f"Example {name} has mismatched motion/audio token lengths: "
                f"{len(motion_tokens)} vs {len(audio_tokens)}"
            )
            if length_mismatch_policy == "strict":
                raise ValueError(message)
            if length_mismatch_policy != "truncate":
                raise ValueError(f"Unsupported length_mismatch_policy: {length_mismatch_policy}")
            print(f"[WARN] {message}; truncating to the shorter sequence")
        max_len = min(len(motion_tokens), len(audio_tokens))
    else:
        max_len = len(motion_tokens)
    indices = list(range(0, max_len, step))
    keyframes = [motion_tokens[i] for i in indices]
    prefix_count = 0
    prompt_body = ""
    if use_audio and audio_tokens is not None and continuation_prefix_keyframes > 0:
        prefix_count = min(continuation_prefix_keyframes, max(0, len(keyframes) - 1))
        prompt_body += audio_motion_prefix_to_text(audio_tokens, keyframes, indices, prefix_count)
    remaining_indices = indices[prefix_count:]
    remaining_keyframes = keyframes[prefix_count:]
    prompt_body += action_text
    if use_audio and audio_tokens is not None:
        prompt_body += audio_tokens_to_text(audio_tokens, remaining_indices)
    prompt = template.format(prompt=prompt_body)
    if len(remaining_keyframes) < 1:
        raise ValueError(f"Example {name} has no target keyframes after continuation split")
    if len(remaining_keyframes) > max_len_token:
        raise ValueError(
            f"Example {name} has {len(remaining_keyframes)} target keyframes, "
            f"but --max_len_token={max_len_token}; increase --max_len_token"
        )
    completion = f"[step_{step}][len_{len(remaining_keyframes)}]" + token_group_to_text(remaining_keyframes) + "<|im_end|>"
    return {
        "name": name,
        "prompt": prompt,
        "completion": completion,
        "num_keyframes": len(remaining_keyframes),
        "num_prefix_keyframes": prefix_count,
        "step": step,
    }


def select_prompt_text(raw_text, args: argparse.Namespace, *, use_audio: bool | None = None, action_text: str | None = None) -> str:
    if action_text or args.action_text:
        return action_text or args.action_text
    raw_text = normalize_motion_text(raw_text)
    mode = args.prompt_text_mode
    if mode == "auto":
        mode = "raw" if (args.no_audio if use_audio is None else not use_audio) else "action"
    if mode == "raw":
        return raw_text or "动作：说话"
    if mode == "action":
        return extract_action_text(raw_text or "动作：说话")
    raise ValueError(f"Unsupported prompt_text_mode: {args.prompt_text_mode}")


def discover_token_names(motion_token_dir: str) -> list[str]:
    return sorted(str(p.relative_to(motion_token_dir)).replace(os.sep, "/")[:-5] for p in Path(motion_token_dir).rglob("*.json"))


def source_names(motion_token_dir: str, split_file: str | None) -> list[str]:
    names = read_split(split_file)
    if names is None:
        names = discover_token_names(motion_token_dir)
    return names


def write_single_source_examples(
    out,
    args: argparse.Namespace,
    *,
    source_name: str,
    motion_token_dir: str,
    motion2text_json: str | None,
    split_file: str | None,
    audio_token_dir: str | None,
    use_audio: bool,
    step: int,
    codebook_size: int,
    num_quantizers: int,
    max_examples: int | None = None,
    action_text: str | None = None,
) -> tuple[int, int]:
    names = source_names(motion_token_dir, split_file)
    if max_examples is not None:
        names = names[: int(max_examples)]

    motion2text = load_json(Path(motion2text_json)) if motion2text_json else {}
    count = 0
    skipped = 0
    for name in names:
        try:
            motion_path = resolve_sequence_path(motion_token_dir, name, ".json")
            audio_path = (
                resolve_sequence_path(audio_token_dir, name, ".json")
                if use_audio and audio_token_dir
                else None
            )
        except FileNotFoundError:
            skipped += 1
            continue
        if use_audio and audio_path is None:
            skipped += 1
            continue
        motion_tokens = load_json(motion_path).get("tokens", [])
        audio_tokens = load_json(audio_path).get("tokens", []) if audio_path is not None and audio_path.exists() else None
        validate_raw_token_frames(motion_tokens, num_quantizers, codebook_size, motion_path)
        if audio_tokens is not None:
            validate_audio_tokens(audio_tokens, args.max_audio_token, audio_path)
        if len(motion_tokens) < 2 or (use_audio and (audio_tokens is None or len(audio_tokens) < 2)):
            skipped += 1
            continue
        prompt_text = select_prompt_text(motion2text.get(name, "动作：说话"), args, use_audio=use_audio, action_text=action_text)
        example = build_example(
            name=f"{source_name}/{name}" if source_name else name,
            action_text=prompt_text,
            motion_tokens=motion_tokens,
            audio_tokens=audio_tokens,
            step=step,
            template=args.template,
            use_audio=use_audio,
            continuation_prefix_keyframes=args.continuation_prefix_keyframes,
            max_len_token=args.max_len_token,
            length_mismatch_policy=args.length_mismatch_policy,
        )
        if source_name:
            example["source"] = source_name
            example["source_sample"] = name
        out.write(json.dumps(example, ensure_ascii=False) + "\n")
        count += 1
    return count, skipped


def iter_manifest_sources(args: argparse.Namespace) -> list[dict]:
    manifest_path = Path(args.source_manifest_json)
    manifest = load_json(manifest_path)
    if isinstance(manifest, dict):
        sources = manifest.get("sources")
    else:
        sources = manifest
    if not isinstance(sources, list) or not sources:
        raise ValueError("--source_manifest_json must contain a non-empty list or {'sources': [...]}")
    base_dir = manifest_path.parent
    resolved = []
    for idx, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(f"Manifest source {idx} must be an object")
        if not source.get("motion_token_dir"):
            raise ValueError(f"Manifest source {idx} missing motion_token_dir")
        item = dict(source)
        item["name"] = str(item.get("name") or f"source_{idx}")
        item["motion_token_dir"] = resolve_manifest_path(item.get("motion_token_dir"), base_dir)
        item["motion2text_json"] = resolve_manifest_path(item.get("motion2text_json"), base_dir)
        item["split_file"] = resolve_manifest_path(item.get("split_file"), base_dir)
        item["audio_token_dir"] = resolve_manifest_path(item.get("audio_token_dir"), base_dir)
        resolved.append(item)
    return resolved


def main(args: argparse.Namespace) -> None:
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_count = 0
    total_skipped = 0
    source_summaries = []
    with out_path.open("w", encoding="utf-8") as out:
        if args.source_manifest_json:
            for source in iter_manifest_sources(args):
                use_audio = bool(source.get("use_audio", not args.no_audio))
                count, skipped = write_single_source_examples(
                    out,
                    args,
                    source_name=source["name"],
                    motion_token_dir=source["motion_token_dir"],
                    motion2text_json=source.get("motion2text_json"),
                    split_file=source.get("split_file"),
                    audio_token_dir=source.get("audio_token_dir") or args.audio_token_dir,
                    use_audio=use_audio,
                    step=int(source.get("step", args.step)),
                    codebook_size=int(source.get("codebook_size", args.codebook_size)),
                    num_quantizers=int(source.get("num_quantizers", args.num_quantizers)),
                    max_examples=source.get("max_examples"),
                    action_text=source.get("action_text"),
                )
                total_count += count
                total_skipped += skipped
                source_summaries.append({"source": source["name"], "count": count, "skipped": skipped})
        else:
            count, skipped = write_single_source_examples(
                out,
                args,
                source_name="",
                motion_token_dir=args.motion_token_dir,
                motion2text_json=args.motion2text_json,
                split_file=args.split_file,
                audio_token_dir=args.audio_token_dir,
                use_audio=not args.no_audio,
                step=args.step,
                codebook_size=args.codebook_size,
                num_quantizers=args.num_quantizers,
                action_text=args.action_text,
            )
            total_count += count
            total_skipped += skipped
    if source_summaries:
        print(json.dumps({"sources": source_summaries}, indent=2, ensure_ascii=False))
    print(f"Wrote {total_count} examples to {out_path}; skipped {total_skipped}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LLM planner SFT JSONL")
    parser.add_argument("--motion_token_dir", default="./data/motion_token_data")
    parser.add_argument("--audio_token_dir", default="./data/audio_tokens_hubert_layer9_fps10")
    parser.add_argument("--motion2text_json", default="./data/text_data/motion2text.json")
    parser.add_argument("--split_file", default="./data/split/train_file_list.txt")
    parser.add_argument("--output_jsonl", default="./data/llm_sft/train_step4.jsonl")
    parser.add_argument("--source_manifest_json", default=None,
                        help="Optional multi-source manifest for Motion Foundation or mixed-corpus SFT data")
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--codebook_size", type=int, default=512)
    parser.add_argument("--num_quantizers", type=int, default=4)
    parser.add_argument("--max_audio_token", type=int, default=2048,
                        help="Largest discrete audio token ID expected in [audio_i] prompts")
    parser.add_argument("--max_len_token", type=int, default=2048,
                        help="Largest [len_N] token supported by the planner tokenizer")
    parser.add_argument("--action_text", default=None)
    parser.add_argument("--prompt_text_mode", choices=["auto", "action", "raw"], default="auto",
                        help="Prompt text source: auto uses raw text for --no_audio foundation data and action tags for planner SFT")
    parser.add_argument("--template", default="Human: {prompt}<|im_end|>\nAssistant:")
    parser.add_argument("--no_audio", action="store_true",
                        help="Build text-to-motion examples for Motion Foundation pre-training")
    parser.add_argument("--continuation_prefix_keyframes", type=int, default=0,
                        help="Simulate multi-turn continuation by moving the first N keyframes into the prompt")
    parser.add_argument("--length_mismatch_policy", choices=["strict", "truncate"], default="strict",
                        help="How to handle motion-token/audio-token length mismatches")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
