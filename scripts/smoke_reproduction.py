#!/usr/bin/env python3
"""End-to-end synthetic smoke test for the SentiAvatar reproduction path.

The script creates a tiny local dataset, trains each trainable component for one
step, and verifies that the saved artifacts can be loaded by the GitHub-style
inference helpers. It is intentionally CPU-friendly and does not require the
real SuSuInterActs data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOTION_DIR = PROJECT_ROOT / "motion_generation"


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def unique_work_dir(base: Path, overwrite: bool) -> Path:
    if overwrite:
        if base.exists():
            shutil.rmtree(base)
        return base
    if not base.exists():
        return base
    stamp = time.strftime("%Y%m%d-%H%M%S")
    candidate = base.with_name(f"{base.name}-{stamp}")
    counter = 1
    while candidate.exists():
        counter += 1
        candidate = base.with_name(f"{base.name}-{stamp}-{counter}")
    return candidate


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_synthetic_wav(path: Path, duration: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    sample_rate = 16000
    num_samples = max(1, int(duration * sample_rate))
    audio = np.zeros(num_samples, dtype=np.float32)
    event_times = np.linspace(0.2, max(0.25, duration - 0.2), num=4)
    for event_time in event_times:
        start = int(event_time * sample_rate)
        length = min(400, num_samples - start)
        if length <= 0:
            continue
        phase = np.linspace(0.0, 30.0 * math.pi, length, dtype=np.float32)
        audio[start : start + length] += 0.65 * np.sin(phase)
    audio += rng.normal(0.0, 0.002, size=num_samples).astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes((audio * 32767.0).astype("<i2").tobytes())


def create_synthetic_data(data_root: Path, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    names = [f"demo/sample_{idx}" for idx in range(3)]

    split_dir = data_root / "split"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train_file_list.txt", "all_file_list.txt"):
        (split_dir / split_name).write_text("\n".join(names) + "\n", encoding="utf-8")

    motion2text = {}
    for idx, name in enumerate(names):
        motion_len = 32 + idx * 4
        audio_len = 16 + idx * 2
        t = np.linspace(0.0, 1.0, motion_len, dtype=np.float32)

        body = rng.normal(0.0, 0.03, size=(motion_len, 153)).astype(np.float32)
        body[:, 0] = 0.05 * np.sin(2 * np.pi * t)
        body[:, 1] = 0.04 * np.cos(2 * np.pi * t)
        body[:, 2] = 100.0 + 0.5 * np.sin(np.pi * t)
        left = rng.normal(0.0, 0.02, size=(motion_len, 120)).astype(np.float32)
        right = rng.normal(0.0, 0.02, size=(motion_len, 120)).astype(np.float32)

        motion_path = data_root / "motion_data" / f"{name}.npy"
        motion_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(motion_path, {"body": body, "left": left, "right": right})

        face = np.clip(rng.normal(0.25, 0.08, size=(motion_len, 51)), 0.0, 1.0).astype(np.float32)
        face_path = data_root / "arkit_data" / f"{name}.npy"
        face_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(face_path, face)

        audio_features = rng.normal(0.0, 1.0, size=(audio_len, 12)).astype(np.float32)
        audio_feat_path = data_root / "audio_features_hubert_layer9_fps10" / f"{name}.npy"
        audio_feat_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(audio_feat_path, audio_features)

        audio_tokens = rng.integers(0, 8, size=(audio_len,), endpoint=False).astype(int).tolist()
        write_json(data_root / "audio_tokens_hubert_layer9_fps10" / f"{name}.json", {"name": name, "tokens": audio_tokens})
        write_synthetic_wav(data_root / "wav_data" / f"{name}.wav", duration=audio_len / 10.0, seed=seed + idx)

        motion_tokens = rng.integers(0, 8, size=(audio_len, 4), endpoint=False).astype(int).tolist()
        write_json(
            data_root / "motion_token_data" / f"{name}.json",
            {
                "name": name,
                "tokens": motion_tokens,
                "num_quantizers": 4,
                "codebook_size": 8,
                "source_fps": 20,
                "token_fps": 10.0,
                "downsample_factor": 2,
            },
        )
        motion2text[name] = "【动作：挥手】"

    write_json(data_root / "text_data" / "motion2text.json", motion2text)
    return names


def verify_body_rvqvae(checkpoint_path: Path) -> None:
    sys.path.insert(0, str(MOTION_DIR))
    from actions.schema import MotionTokens
    from infer import decode_tokens, load_config_from_checkpoint, load_model, load_motion_stats

    device = torch.device("cpu")
    config = load_config_from_checkpoint(str(checkpoint_path))
    model = load_model(str(checkpoint_path), config, device)
    mean, std = load_motion_stats(str(checkpoint_path), device)
    assert mean.shape[-1] == 153 and std.shape[-1] == 153
    with torch.no_grad():
        out, _, _ = model(torch.zeros(1, 16, 153, dtype=torch.float32))
    assert out.shape[-1] == 153, f"Unexpected RVQVAE output shape: {tuple(out.shape)}"
    placeholder = {
        "body": np.zeros((32, 153), dtype=np.float32),
        "left": np.zeros((32, 120), dtype=np.float32),
        "right": np.zeros((32, 120), dtype=np.float32),
    }
    decoded = decode_tokens(
        model,
        MotionTokens(body=[[0, 0, 0, 0] for _ in range(8)]),
        placeholder,
        config,
        device,
        src_fps=20.0,
        tgt_fps=30.0,
        checkpoint_path=str(checkpoint_path),
    )
    assert decoded["offset"].shape[0] > 0 and decoded["quat"].shape[1:] == (63, 4)


def verify_llm_jsonl(path: Path) -> None:
    sys.path.insert(0, str(MOTION_DIR))
    from pipeline_infer import extract_mids_from_string
    from training.train_llm_planner import validate_planner_jsonl_contract

    validate_planner_jsonl_contract(
        str(path),
        codebook_size=8,
        num_quantizers=4,
        max_audio_token=8,
        max_len_token=32,
    )
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    parsed = extract_mids_from_string(row["completion"])
    assert parsed["step"] == [row["step"]]
    assert parsed["len"] == [row["num_keyframes"]]
    for key in ("res1", "res2", "res3", "res4"):
        assert len(parsed[key]) == row["num_keyframes"], f"{key} length mismatch"


def verify_continuation_jsonl(path: Path) -> None:
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["num_prefix_keyframes"] > 0
    assert "[audio_" in row["prompt"] and "[res1_" in row["prompt"]
    verify_llm_jsonl(path)


def verify_infill_model(checkpoint_dir: Path, num_tokens_per_frame: int, audio_feat_dim: int) -> None:
    sys.path.insert(0, str(MOTION_DIR))
    from pipeline_infer import load_mask_transformer

    model = load_mask_transformer(str(checkpoint_dir), device="cpu")
    cfg = model.config
    mask_id = int(cfg.vocab_size) - 1
    input_ids = torch.full((1, cfg.num_frames * num_tokens_per_frame), mask_id, dtype=torch.long)
    audio = torch.zeros(1, cfg.num_frames, audio_feat_dim, dtype=torch.float32)
    out = model.generate_sbs(input_ids, audio, generate_steps=2)
    assert tuple(out.shape) == tuple(input_ids.shape)


def run_pipeline_infer_smoke(body_infill_dir: Path, data_root: Path, sample_name: str) -> dict:
    sys.path.insert(0, str(MOTION_DIR))
    from pipeline_infer import load_mask_transformer, run_pipeline_single

    class FakePlanner:
        def __init__(self, keyframes: list[list[int]]) -> None:
            self.keyframes = keyframes

        def predict_motion_plan(self, *args, **kwargs):
            tokens = {
                "res_1": [frame[0] for frame in self.keyframes],
                "res_2": [frame[1] for frame in self.keyframes],
                "res_3": [frame[2] for frame in self.keyframes],
                "res_4": [frame[3] for frame in self.keyframes],
            }
            raw = "[step_4][len_{}]".format(len(self.keyframes))
            raw += "".join(
                f"[res1_{f[0]}][res2_{f[1]}][res3_{f[2]}][res4_{f[3]}]"
                for f in self.keyframes
            )
            return [{"tokens": tokens, "total_len": len(self.keyframes), "raw_output": raw}]

    motion_tokens = load_json(data_root / "motion_token_data" / f"{sample_name}.json")["tokens"]
    audio_tokens = load_json(data_root / "audio_tokens_hubert_layer9_fps10" / f"{sample_name}.json")["tokens"]
    audio_features = np.load(data_root / "audio_features_hubert_layer9_fps10" / f"{sample_name}.npy").astype(np.float32)
    step = 4
    keyframes = [motion_tokens[i] for i in range(0, min(len(motion_tokens), len(audio_features)), step)]
    mask_model = load_mask_transformer(str(body_infill_dir), device="cpu")
    result = run_pipeline_single(
        FakePlanner(keyframes),
        mask_model,
        action_text="动作：挥手",
        audio_tokens=audio_tokens,
        audio_features=audio_features,
        name=sample_name,
        step=step,
        generate_steps=2,
    )
    assert result is not None
    assert len(result["dense_tokens"]) == audio_features.shape[0]
    assert result["num_keyframes"] == len(keyframes)
    continuation_start = step + 1
    continuation_keyframes = [
        motion_tokens[i]
        for i in range(2 * step, min(len(motion_tokens), len(audio_features)), step)
    ]
    if len(continuation_keyframes) >= 1 and len(audio_features) > 2 * step:
        prefix_audio_tokens = audio_tokens[:continuation_start]
        prefix_audio_features = audio_features[:continuation_start]
        prefix_motion_tokens = motion_tokens[:continuation_start]
        continuation_result = run_pipeline_single(
            FakePlanner(continuation_keyframes),
            mask_model,
            action_text="动作：继续挥手",
            audio_tokens=audio_tokens[continuation_start:],
            audio_features=audio_features[continuation_start:],
            name=f"{sample_name}_continuation",
            step=step,
            generate_steps=2,
            prefix_audio_tokens=prefix_audio_tokens,
            prefix_audio_features=prefix_audio_features,
            prefix_motion_tokens=prefix_motion_tokens,
            continuation_prefix_keyframes=2,
        )
        assert continuation_result is not None
        assert continuation_result["continuation_boundary_used"]
        assert continuation_result["continuation_boundary_keyframe"] == motion_tokens[step]
        assert continuation_result["continuation_audio_boundary_used"]
        assert continuation_result["continuation_sampling_start"] == step - 1
        assert continuation_result["expected_keyframes"] == len(continuation_keyframes)
        assert len(continuation_result["dense_tokens"]) == audio_features[continuation_start:].shape[0]
        result["continuation_smoke"] = {
            "num_keyframes": continuation_result["num_keyframes"],
            "expected_keyframes": continuation_result["expected_keyframes"],
            "num_dense_frames": continuation_result["num_dense_frames"],
            "boundary_keyframe": continuation_result["continuation_boundary_keyframe"],
            "sampling_start": continuation_result["continuation_sampling_start"],
            "audio_boundary_used": continuation_result["continuation_audio_boundary_used"],
        }
    return result


def verify_single_case_artifact_io(
    work_dir: Path,
    data_root: Path,
    sample_name: str,
    pipeline_result: dict,
    body_ckpt: Path,
) -> Path:
    sys.path.insert(0, str(MOTION_DIR))
    from infer import load_config_from_checkpoint
    from single_case_infer import load_continuation_inputs, save_single_case_artifacts

    output_dir = work_dir / "single_case_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_tokens = load_json(data_root / "audio_tokens_hubert_layer9_fps10" / f"{sample_name}.json")["tokens"]
    audio_features = np.load(data_root / "audio_features_hubert_layer9_fps10" / f"{sample_name}.npy").astype(np.float32)
    config = load_config_from_checkpoint(str(body_ckpt))
    output_name = sample_name.replace("/", "_")
    audio_token_path, audio_feat_path, motion_token_path, pipeline_path = save_single_case_artifacts(
        str(output_dir),
        output_name,
        pipeline_result,
        audio_tokens,
        audio_features,
        "动作：挥手",
        str(data_root / "wav_data" / f"{sample_name}.wav"),
        config,
    )
    args = argparse.Namespace(
        prefix_audio_token_json=audio_token_path,
        prefix_audio_feat_npy=audio_feat_path,
        prefix_motion_token_json=motion_token_path,
    )
    prefix_audio_tokens, prefix_audio_features, prefix_motion_tokens = load_continuation_inputs(args)
    assert prefix_audio_tokens == audio_tokens
    assert prefix_audio_features.shape == audio_features.shape
    assert prefix_motion_tokens == pipeline_result["dense_tokens"]
    saved_pipeline = load_json(Path(pipeline_path))
    assert saved_pipeline["dense_tokens"] == pipeline_result["dense_tokens"]
    assert saved_pipeline["audio_token_path"] == audio_token_path
    return Path(pipeline_path)


def verify_face_pipeline(face_rvqvae_ckpt: Path, face_infill_dir: Path, audio_feat_path: Path) -> None:
    sys.path.insert(0, str(MOTION_DIR))
    from face_infill import FaceInfillPipeline, add_face_to_anim

    audio_features = np.load(audio_feat_path)
    pipe = FaceInfillPipeline(
        face_rvqvae_ckpt=str(face_rvqvae_ckpt),
        face_infill_ckpt=str(face_infill_dir),
        device="cpu",
        generate_steps=2,
    )
    face = pipe.infer_mta52(audio_features)
    assert face.shape == (audio_features.shape[0], 52), f"Unexpected face shape: {face.shape}"
    anim = {"frames": [{} for _ in range(24)]}
    add_face_to_anim(anim, face)
    assert len(anim["frames"][0]["face"]) == 52


def run_reconstruct_and_sync_eval(env: dict[str, str], data_root: Path, checkpoints: Path, work_dir: Path, sample_name: str) -> Path:
    tokens = load_json(data_root / "motion_token_data" / f"{sample_name}.json")["tokens"]
    result_path = work_dir / "pipeline_result_for_reconstruct.json"
    write_json(result_path, {"name": sample_name, "dense_tokens": tokens, "gt_tokens": tokens})
    reconstruct_dir = work_dir / "reconstructed"
    body_ckpt = checkpoints / "smoke" / "rvqvae_body" / "model" / "latest.pth"
    run(
        [
            sys.executable,
            "motion_generation/reconstruct_from_tokens.py",
            "--input_json",
            str(result_path),
            "--checkpoint_path",
            str(body_ckpt),
            "--placeholder_npy",
            str(data_root / "motion_data" / f"{sample_name}.npy"),
            "--output_dir",
            str(reconstruct_dir),
            "--device",
            "cpu",
            "--face_mode",
            "none",
            "--max_samples",
            "1",
        ],
        env,
    )
    pred_npy = reconstruct_dir / f"{sample_name.replace('/', '_')}_pred.npy"
    gt_npy = reconstruct_dir / f"{sample_name.replace('/', '_')}_gt.npy"
    assert pred_npy.exists() and gt_npy.exists(), "reconstruct_from_tokens did not save eval npy files"
    metrics_path = work_dir / "sync_metrics.json"
    run(
        [
            sys.executable,
            "scripts/eval_sync_metrics.py",
            "--motion_dir",
            str(reconstruct_dir),
            "--wav_dir",
            str(data_root / "wav_data"),
            "--motion_type",
            "pred",
            "--output_json",
            str(metrics_path),
            "--max_samples",
            "1",
        ],
        env,
    )
    metrics = load_json(metrics_path)["metrics"]
    assert metrics["num_evaluated"] == 1, "sync metric smoke did not evaluate a sample"
    return metrics_path


def create_tiny_llm_model(model_dir: Path) -> Path:
    """Create a local HF CausalLM fixture so smoke tests stay offline."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    model_dir.mkdir(parents=True, exist_ok=True)
    vocab = {
        "[UNK]": 0,
        "<|endoftext|>": 1,
        "Human": 2,
        "Assistant": 3,
        "动作": 4,
        "挥手": 5,
    }
    tokenizers_model = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizers_model.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizers_model,
        unk_token="[UNK]",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )
    tokenizer.save_pretrained(model_dir)
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=256,
        n_ctx=256,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    GPT2LMHeadModel(config).save_pretrained(model_dir, safe_serialization=True)
    return model_dir


def run_llm_train_stage(
    init_model: str,
    train_jsonl: Path,
    output_dir: Path,
    env: dict[str, str],
) -> None:
    run(
        [
            sys.executable,
            "motion_generation/training/train_llm_planner.py",
            "--init_model",
            init_model,
            "--train_jsonl",
            str(train_jsonl),
            "--output_dir",
            str(output_dir),
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--max_steps",
            "1",
            "--max_length",
            "256",
            "--batch_size",
            "1",
            "--grad_accum_steps",
            "1",
            "--num_workers",
            "0",
            "--lr",
            "1e-5",
            "--save_steps",
            "1",
            "--add_planner_tokens",
            "--codebook_size",
            "8",
            "--num_quantizers",
            "4",
            "--max_audio_token",
            "8",
            "--max_len_token",
            "32",
            "--step_tokens",
            "4",
        ],
        env,
    )
    assert (output_dir / "config.json").exists(), f"LLM smoke did not save config.json: {output_dir}"
    assert (output_dir / "tokenizer.json").exists(), f"LLM smoke did not save tokenizer.json: {output_dir}"
    assert (output_dir / "checkpoint-1").exists(), f"LLM smoke did not save checkpoint-1: {output_dir}"


def maybe_train_llm(
    args: argparse.Namespace,
    env: dict[str, str],
    foundation_jsonl: Path,
    sft_jsonl: Path,
    work_dir: Path,
) -> dict[str, str]:
    if not args.include_llm_train:
        return {}
    llm_model = args.llm_model
    if llm_model == "auto-tiny":
        llm_model = str(create_tiny_llm_model(work_dir / "fixtures" / "tiny_llm"))
    foundation_dir = work_dir / "checkpoints" / "motion_foundation"
    planner_dir = work_dir / "checkpoints" / "llm_planner"
    run_llm_train_stage(llm_model, foundation_jsonl, foundation_dir, env)
    run_llm_train_stage(str(foundation_dir), sft_jsonl, planner_dir, env)
    return {
        "motion_foundation": str(foundation_dir),
        "llm_planner": str(planner_dir),
    }


def main(args: argparse.Namespace) -> None:
    work_dir = unique_work_dir(Path(args.work_dir).resolve(), args.overwrite)
    data_root = work_dir / "data"
    checkpoints = work_dir / "checkpoints"
    logs = work_dir / "logs"
    work_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    pythonpath = [str(PROJECT_ROOT), str(MOTION_DIR)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    names = create_synthetic_data(data_root, args.seed)
    train_split = data_root / "split" / "train_file_list.txt"
    all_split = data_root / "split" / "all_file_list.txt"

    run(
        [
            sys.executable,
            "motion_generation/training/train_rvqvae.py",
            "--data_root",
            str(data_root),
            "--train_split",
            str(train_split),
            "--checkpoints_dir",
            str(checkpoints),
            "--log_dir",
            str(logs),
            "--dataset_name",
            "smoke",
            "--name",
            "rvqvae_body",
            "--device",
            "cpu",
            "--window_size",
            "16",
            "--window_stride",
            "8",
            "--batch_size",
            "2",
            "--num_workers",
            "0",
            "--epochs",
            "1",
            "--max_steps",
            "1",
            "--limit_windows",
            "2",
            "--nb_code",
            "8",
            "--code_dim",
            "32",
            "--width",
            "32",
            "--depth",
            "1",
            "--vq_cnn_depth",
            "1",
            "--num_quantizers",
            "4",
            "--quantize_dropout_prob",
            "0.0",
            "--save_latest",
            "1",
            "--save_every_e",
            "1",
            "--log_every",
            "1",
        ],
        env,
    )
    body_ckpt = checkpoints / "smoke" / "rvqvae_body" / "model" / "latest.pth"
    verify_body_rvqvae(body_ckpt)
    sync_metrics_path = run_reconstruct_and_sync_eval(env, data_root, checkpoints, work_dir, names[0])

    sft_jsonl = data_root / "llm_sft" / "train_step4.jsonl"
    foundation_jsonl = data_root / "llm_sft" / "foundation_text_only.jsonl"
    continuation_jsonl = data_root / "llm_sft" / "continuation_step4.jsonl"
    mixed_sft_jsonl = data_root / "llm_sft" / "train_step4_mixed.jsonl"
    for output, step, extra in ((sft_jsonl, "4", []), (foundation_jsonl, "1", ["--no_audio"])):
        run(
            [
                sys.executable,
                "motion_generation/training/build_llm_sft_dataset.py",
                "--motion_token_dir",
                str(data_root / "motion_token_data"),
                "--audio_token_dir",
                str(data_root / "audio_tokens_hubert_layer9_fps10"),
                "--motion2text_json",
                str(data_root / "text_data" / "motion2text.json"),
                "--split_file",
                str(train_split),
                "--output_jsonl",
                str(output),
                "--step",
                step,
                *extra,
            ],
            env,
        )
        verify_llm_jsonl(output)
    run(
        [
            sys.executable,
            "motion_generation/training/build_llm_sft_dataset.py",
            "--motion_token_dir",
            str(data_root / "motion_token_data"),
            "--audio_token_dir",
            str(data_root / "audio_tokens_hubert_layer9_fps10"),
            "--motion2text_json",
            str(data_root / "text_data" / "motion2text.json"),
            "--split_file",
            str(train_split),
            "--output_jsonl",
            str(continuation_jsonl),
            "--step",
            "4",
            "--continuation_prefix_keyframes",
            "2",
        ],
        env,
    )
    verify_continuation_jsonl(continuation_jsonl)
    run(
        [
            sys.executable,
            "scripts/merge_llm_jsonl.py",
            "--inputs",
            f"regular={sft_jsonl}",
            f"continuation={continuation_jsonl}",
            "--output_jsonl",
            str(mixed_sft_jsonl),
            "--summary_json",
            str(data_root / "llm_sft" / "train_step4_mixed_summary.json"),
            "--shuffle",
            "--seed",
            str(args.seed),
        ],
        env,
    )
    verify_llm_jsonl(mixed_sft_jsonl)
    llm_outputs = maybe_train_llm(args, env, foundation_jsonl, mixed_sft_jsonl, work_dir)

    body_infill = checkpoints / "body_infill"
    run(
        [
            sys.executable,
            "motion_generation/training/train_infill_transformer.py",
            "--motion_token_dir",
            str(data_root / "motion_token_data"),
            "--audio_feat_dir",
            str(data_root / "audio_features_hubert_layer9_fps10"),
            "--train_split",
            str(train_split),
            "--output_dir",
            str(body_infill),
            "--device",
            "cpu",
            "--num_frames",
            "5",
            "--num_tokens_per_frame",
            "4",
            "--codebook_size",
            "8",
            "--audio_feat_dim",
            "12",
            "--hidden_size",
            "32",
            "--num_layers",
            "1",
            "--num_heads",
            "4",
            "--intermediate_size",
            "64",
            "--max_position_embeddings",
            "64",
            "--dropout",
            "0.0",
            "--cond_drop_prob",
            "0.0",
            "--mask_prob",
            "1.0",
            "--random_replace_prob",
            "0.1",
            "--batch_size",
            "2",
            "--num_workers",
            "0",
            "--epochs",
            "1",
            "--max_steps",
            "1",
            "--limit_windows",
            "2",
            "--save_steps",
            "1",
            "--save_every_epochs",
            "1",
            "--log_every",
            "1",
        ],
        env,
    )
    verify_infill_model(body_infill, num_tokens_per_frame=4, audio_feat_dim=12)
    pipeline_result = run_pipeline_infer_smoke(body_infill, data_root, names[0])
    pipeline_result_path = work_dir / "pipeline_infer_smoke_result.json"
    write_json(pipeline_result_path, pipeline_result)
    single_case_artifacts_path = verify_single_case_artifact_io(
        work_dir,
        data_root,
        names[0],
        pipeline_result,
        body_ckpt,
    )

    face_rvqvae = checkpoints / "face_rvqvae"
    run(
        [
            sys.executable,
            "motion_generation/training/train_face_rvqvae.py",
            "--data_root",
            str(data_root),
            "--face_dir",
            str(data_root / "arkit_data"),
            "--train_split",
            str(train_split),
            "--output_dir",
            str(face_rvqvae),
            "--device",
            "cpu",
            "--window_size",
            "16",
            "--window_stride",
            "8",
            "--batch_size",
            "2",
            "--num_workers",
            "0",
            "--epochs",
            "1",
            "--max_steps",
            "1",
            "--limit_windows",
            "2",
            "--hidden_size",
            "32",
            "--code_dim",
            "32",
            "--codebook_size",
            "8",
            "--num_quantizers",
            "2",
            "--num_res_blocks",
            "1",
            "--save_steps",
            "1",
            "--save_every_epochs",
            "1",
            "--log_every",
            "1",
        ],
        env,
    )

    face_tokens = data_root / "face_token_data"
    run(
        [
            sys.executable,
            "motion_generation/training/preprocess_face_tokens.py",
            "--data_root",
            str(data_root),
            "--face_dir",
            str(data_root / "arkit_data"),
            "--audio_feat_dir",
            str(data_root / "audio_features_hubert_layer9_fps10"),
            "--split_file",
            str(all_split),
            "--output_dir",
            str(face_tokens),
            "--face_rvqvae_ckpt",
            str(face_rvqvae / "latest.pth"),
            "--device",
            "cpu",
            "--overwrite",
        ],
        env,
    )

    face_infill = checkpoints / "face_infill"
    run(
        [
            sys.executable,
            "motion_generation/training/train_infill_transformer.py",
            "--motion_token_dir",
            str(face_tokens),
            "--audio_feat_dir",
            str(data_root / "audio_features_hubert_layer9_fps10"),
            "--train_split",
            str(train_split),
            "--output_dir",
            str(face_infill),
            "--device",
            "cpu",
            "--num_frames",
            "5",
            "--num_tokens_per_frame",
            "2",
            "--codebook_size",
            "8",
            "--audio_feat_dim",
            "12",
            "--hidden_size",
            "32",
            "--num_layers",
            "1",
            "--num_heads",
            "4",
            "--intermediate_size",
            "64",
            "--max_position_embeddings",
            "64",
            "--dropout",
            "0.0",
            "--cond_drop_prob",
            "0.0",
            "--mask_prob",
            "1.0",
            "--random_replace_prob",
            "0.1",
            "--boundary_mode",
            "none",
            "--batch_size",
            "2",
            "--num_workers",
            "0",
            "--epochs",
            "1",
            "--max_steps",
            "1",
            "--limit_windows",
            "2",
            "--save_steps",
            "1",
            "--save_every_epochs",
            "1",
            "--log_every",
            "1",
        ],
        env,
    )
    verify_face_pipeline(face_rvqvae / "latest.pth", face_infill, data_root / "audio_features_hubert_layer9_fps10" / f"{names[0]}.npy")

    audit_cmd = [
        sys.executable,
        "scripts/audit_reproduction_data.py",
        "--data_dir",
        str(data_root),
        "--max_samples",
        "3",
        "--require_preprocessed",
        "--check_face",
        "--audio_feat_dim",
        "12",
        "--num_face_quantizers",
        "2",
        "--face_codebook_size",
        "8",
        "--rvqvae_ckpt",
        str(body_ckpt),
        "--mask_ckpt",
        str(body_infill),
        "--face_rvqvae_ckpt",
        str(face_rvqvae / "latest.pth"),
        "--face_infill_ckpt",
        str(face_infill),
    ]
    if args.include_llm_train:
        audit_cmd.extend([
            "--motion_foundation_dir",
            llm_outputs["motion_foundation"],
            "--llm_dir",
            llm_outputs["llm_planner"],
            "--max_audio_token",
            "8",
            "--max_len_token",
            "32",
            "--step_tokens",
            "4",
        ])
    run(audit_cmd, env)

    summary = {
        "work_dir": str(work_dir),
        "body_rvqvae": str(body_ckpt),
        "llm_sft_jsonl": str(sft_jsonl),
        "llm_sft_mixed_jsonl": str(mixed_sft_jsonl),
        "motion_foundation_jsonl": str(foundation_jsonl),
        "continuation_jsonl": str(continuation_jsonl),
        "pipeline_infer_result": str(pipeline_result_path),
        "single_case_artifacts": str(single_case_artifacts_path),
        "body_infill": str(body_infill),
        "face_rvqvae": str(face_rvqvae / "latest.pth"),
        "face_infill": str(face_infill),
        "sync_metrics": str(sync_metrics_path),
        "samples": names,
    }
    summary.update(llm_outputs)
    print("\nSentiAvatar synthetic reproduction smoke test passed.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic SentiAvatar reproduction smoke test")
    parser.add_argument("--work_dir", default="/tmp/sentiavatar_smoke_auto")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--overwrite", action="store_true", help="Remove and recreate work_dir before running")
    parser.add_argument("--include_llm_train", action="store_true",
                        help="Also run one-step Motion Foundation and planner CausalLM SFT smoke checks")
    parser.add_argument("--llm_model", default="auto-tiny",
                        help="Local path or model ID used with --include_llm_train; auto-tiny creates an offline fixture")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
