#!/usr/bin/env python3
"""Merge LLM SFT JSONL files while preserving row provenance."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_input_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Input label is empty in spec: {spec}")
        return label, Path(path)
    path = Path(spec)
    return path.stem, path


def read_rows(label: str, path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL input not found: {path}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("prompt"), str) or not isinstance(row.get("completion"), str):
                raise ValueError(f"{path}:{line_idx} must contain string prompt and completion")
            row = dict(row)
            row.setdefault("sft_source", label)
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def main(args: argparse.Namespace) -> int:
    all_rows: list[dict] = []
    summary = []
    for spec in args.inputs:
        label, path = parse_input_spec(spec)
        rows = read_rows(label, path)
        if args.max_rows_per_input is not None:
            rows = rows[: args.max_rows_per_input]
        all_rows.extend(rows)
        summary.append({"label": label, "path": str(path), "rows": len(rows)})
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(all_rows)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    payload = {"output_jsonl": str(out_path), "total_rows": len(all_rows), "inputs": summary}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LLM SFT JSONL files")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Input specs as label=/path/file.jsonl or plain paths")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max_rows_per_input", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
