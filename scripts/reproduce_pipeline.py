#!/usr/bin/env python3
"""Stage runner for reproducing the trainable SentiAvatar pipeline on real data."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Stage:
    name: str
    description: str
    command: list[str]
    env: dict[str, str] | None = None
    outputs: list[Path] | None = None
    any_outputs: list[tuple[Path, ...]] | None = None


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def split_int_csv(value: str) -> list[int]:
    return [int(part) for part in split_csv(value)]


def format_command(stage: Stage) -> str:
    env_prefix = ""
    if stage.env:
        env_prefix = " ".join(f"{key}={value}" for key, value in sorted(stage.env.items())) + " "
    return env_prefix + " ".join(stage.command)


def hf_weight_outputs(root: Path) -> tuple[Path, ...]:
    return (
        root / "model.safetensors",
        root / "pytorch_model.bin",
        root / "model.safetensors.index.json",
        root / "pytorch_model.bin.index.json",
    )


def tokenizer_outputs(root: Path) -> tuple[Path, ...]:
    return (
        root / "tokenizer.json",
        root / "tokenizer.model",
        root / "vocab.json",
    )


def stage_outputs_exist(stage: Stage) -> bool:
    exact_outputs = stage.outputs or []
    if any(not output_exists(path) for path in exact_outputs):
        return False
    for alternatives in stage.any_outputs or []:
        if not any(path.exists() for path in alternatives):
            return False
    return bool(exact_outputs or stage.any_outputs)


def output_exists(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_dir():
        return any(child.is_file() for child in path.rglob("*"))
    return True


def run_stage(stage: Stage, dry_run: bool, skip_existing: bool) -> None:
    if skip_existing and stage_outputs_exist(stage):
        print(f"\n[SKIP] {stage.name}: outputs already exist")
        for path in stage.outputs or []:
            print(f"  - {path}")
        for alternatives in stage.any_outputs or []:
            found = next((path for path in alternatives if path.exists()), alternatives[0])
            print(f"  - {found}")
        return

    print(f"\n[{stage.name}] {stage.description}")
    print("$ " + format_command(stage))
    if dry_run:
        return
    env = os.environ.copy()
    if stage.env:
        env.update(stage.env)
    subprocess.run(stage.command, cwd=PROJECT_ROOT, env=env, check=True)


def build_stages(args: argparse.Namespace) -> list[Stage]:
    data_dir = Path(args.data_dir).resolve()
    checkpoints_dir = Path(args.checkpoints_dir).resolve()
    logs_dir = Path(args.logs_dir).resolve()
    dataset_name = args.dataset_name
    run_name = args.rvqvae_run_name

    train_split = Path(args.train_split).resolve() if args.train_split else data_dir / "split" / "train_file_list.txt"
    test_split = Path(args.test_split).resolve() if args.test_split else data_dir / "split" / "test_file_list.txt"
    all_split = Path(args.all_split).resolve() if args.all_split else data_dir / "split" / "all_file_list.txt"
    motion2text = Path(args.motion2text_json).resolve() if args.motion2text_json else data_dir / "text_data" / "motion2text.json"
    foundation_manifest = Path(args.foundation_manifest_json).resolve() if args.foundation_manifest_json else None
    output_dir = Path(args.output_dir).resolve()

    rvqvae_ckpt = checkpoints_dir / dataset_name / run_name / "model" / "latest.pth"
    hubert_path = Path(args.hubert_path).resolve() if args.hubert_path else PROJECT_ROOT / "checkpoints" / "chinese-hubert-base"
    kmeans_path = Path(args.kmeans_path).resolve() if args.kmeans_path else PROJECT_ROOT / "checkpoints" / "hubert_kmeans" / "model.mdl"
    mask_ckpt = checkpoints_dir / "mask_transformer"
    face_rvqvae_dir = checkpoints_dir / "face_rvqvae"
    face_rvqvae_ckpt = face_rvqvae_dir / "latest.pth"
    face_infill_ckpt = checkpoints_dir / "face_infill_transformer"
    foundation_jsonl = data_dir / "llm_sft" / "foundation_step1.jsonl"
    planner_jsonl = data_dir / "llm_sft" / f"train_step{args.step}.jsonl"
    continuation_jsonl = data_dir / "llm_sft" / f"train_step{args.step}_continuation.jsonl"
    planner_mixed_jsonl = data_dir / "llm_sft" / f"train_step{args.step}_mixed.jsonl"
    planner_train_jsonl = {
        "regular": planner_jsonl,
        "continuation": continuation_jsonl,
        "both": planner_mixed_jsonl,
    }[args.planner_sft_mix]
    motion_foundation_dir = checkpoints_dir / "motion_foundation"
    llm_dir = checkpoints_dir / "llm"
    pipeline_batch_json = output_dir / "pipeline_batch_results.json"
    reconstructed_dir = output_dir / "reconstructed"
    eval_output_dir = output_dir / "eval_results"
    sync_metrics_json = eval_output_dir / "sync_metrics.json"
    beatv2_metrics_json = eval_output_dir / "beatv2_metrics.json"
    check_face = not (args.skip_face or args.skip_face_audit)
    face_mode = args.face_mode if args.face_mode is not None else ("none" if args.skip_face or args.skip_face_audit else "infill")
    vllm_url = args.vllm_url or f"http://localhost:{args.vllm_port}"
    planner_step_tokens = sorted(set(split_int_csv(args.planner_step_tokens) + [int(args.step)]))
    planner_step_tokens_csv = ",".join(str(step) for step in planner_step_tokens)
    motion_foundation_audit_args = [] if args.no_motion_foundation else [
        "--motion_foundation_dir",
        str(motion_foundation_dir),
    ]
    face_audit_args = [] if args.skip_face or args.skip_face_audit else [
        "--face_rvqvae_ckpt",
        str(face_rvqvae_ckpt),
        "--face_infill_ckpt",
        str(face_infill_ckpt),
    ]
    infer_max_samples_args = [] if args.infer_max_samples is None else [
        "--max_samples",
        str(args.infer_max_samples),
    ]
    reconstruct_max_samples_args = [] if args.infer_max_samples is None else [
        "--max_samples",
        str(args.infer_max_samples),
    ]
    preprocess_split_args = [] if args.all_split is None and not all_split.exists() else [
        "--split_file",
        str(all_split),
    ]
    placeholder_args = [] if args.placeholder_npy is None else [
        "--placeholder_npy",
        str(Path(args.placeholder_npy).resolve()),
    ]
    face_reconstruct_args = ["--face_mode", face_mode]
    if face_mode == "infill":
        face_reconstruct_args.extend([
            "--face_rvqvae_ckpt",
            str(face_rvqvae_ckpt),
            "--face_infill_ckpt",
            str(face_infill_ckpt),
            "--audio_feat_dir",
            str(data_dir / "audio_features_hubert_layer9_fps10"),
        ])
    eval_max_samples_args = [] if args.eval_max_samples is None else [
        "--max_samples",
        str(args.eval_max_samples),
    ]
    beatv2_feature_model_args = [] if args.beatv2_feature_model_path is None else [
        "--feature_model_path",
        str(Path(args.beatv2_feature_model_path).resolve()),
        "--device",
        args.device,
    ]
    eval_gt_motion_dir = reconstructed_dir if args.eval_gt_motion_dir is None else Path(args.eval_gt_motion_dir).resolve()
    eval_real_gt_motion_dir = data_dir / "motion_data" if args.eval_real_gt_motion_dir is None else Path(args.eval_real_gt_motion_dir).resolve()
    eval_text_model_env = {} if args.eval_text_model_name is None else {"TEXT_MODEL_NAME": args.eval_text_model_name}

    python = sys.executable
    common_train_env = {
        "DATA_ROOT": str(data_dir),
        "TRAIN_SPLIT": str(train_split),
        "CHECKPOINTS_DIR": str(checkpoints_dir),
        "DATASET_NAME": dataset_name,
        "RUN_NAME": run_name,
        "DEVICE": args.device,
    }

    stages = [
        Stage(
            "audit-raw",
            "Check raw data split and motion dimensions before long training.",
            [
                python,
                "scripts/audit_reproduction_data.py",
                "--data_dir",
                str(data_dir),
                "--max_samples",
                str(args.audit_samples),
                *("--check_face".split() if check_face else []),
            ],
        ),
        Stage(
            "train-rvqvae",
            "Train Motion R-VQVAE tokenizer.",
            ["bash", "scripts/train_rvqvae.sh"],
            env={
                **common_train_env,
                "BATCH_SIZE": str(args.rvqvae_batch_size),
                "EPOCHS": str(args.rvqvae_epochs),
                "NB_CODE": str(args.body_codebook_size),
                "CODE_DIM": str(args.rvqvae_code_dim),
                "DOWN_T": str(args.rvqvae_down_t),
                "STRIDE_T": str(args.rvqvae_stride_t),
                "WIDTH": str(args.rvqvae_width),
                "DEPTH": str(args.rvqvae_depth),
                "DILATION_GROWTH_RATE": str(args.rvqvae_dilation_growth_rate),
                "VQ_CNN_DEPTH": str(args.rvqvae_vq_cnn_depth),
                "NUM_QUANTIZERS": str(args.num_body_quantizers),
            },
            outputs=[rvqvae_ckpt],
        ),
        Stage(
            "preprocess",
            "Extract HuBERT features/audio tokens and encode motion tokens with trained RVQVAE.",
            [
                "bash",
                "scripts/preprocess_data.sh",
                "--all",
                "--data_dir",
                str(data_dir),
                "--device",
                args.device,
                "--rvqvae_ckpt",
                str(rvqvae_ckpt),
                "--hubert_path",
                str(hubert_path),
                "--kmeans_path",
                str(kmeans_path),
                *preprocess_split_args,
                *(["--overwrite"] if args.overwrite_preprocess else []),
                "--strict",
            ],
            outputs=[
                data_dir / "audio_features_hubert_layer9_fps10",
                data_dir / "audio_tokens_hubert_layer9_fps10",
                data_dir / "motion_token_data",
            ],
        ),
        Stage(
            "build-foundation-jsonl",
            "Build text-to-motion Motion Foundation JSONL.",
            ["bash", "scripts/build_motion_foundation_dataset.sh"],
            env={
                "MOTION_TOKEN_DIR": str(data_dir / "motion_token_data"),
                "MOTION2TEXT_JSON": str(motion2text),
                "TRAIN_SPLIT": str(train_split),
                "OUTPUT_JSONL": str(foundation_jsonl),
                **({} if foundation_manifest is None else {"FOUNDATION_MANIFEST_JSON": str(foundation_manifest)}),
                "STEP": "1",
                "CODEBOOK_SIZE": str(args.body_codebook_size),
                "NUM_QUANTIZERS": str(args.num_body_quantizers),
                "MAX_AUDIO_TOKEN": str(args.max_audio_token),
                "MAX_LEN_TOKEN": str(args.max_len_token),
                "LENGTH_MISMATCH_POLICY": args.length_mismatch_policy,
            },
            outputs=[foundation_jsonl],
        ),
        Stage(
            "train-motion-foundation",
            "Full-parameter SFT pre-training from base Qwen on text-to-motion examples.",
            [
                "bash",
                "scripts/train_llm_planner.sh",
                "--train_jsonl",
                str(foundation_jsonl),
                "--output_dir",
                str(motion_foundation_dir),
                "--batch_size",
                str(args.llm_batch_size),
                "--grad_accum_steps",
                str(args.llm_grad_accum_steps),
                "--max_length",
                str(args.llm_max_length),
                "--step_tokens",
                planner_step_tokens_csv,
            ],
            env={
                "INIT_MODEL": args.init_llm,
                "DEVICE": args.device,
                "EPOCHS": str(args.foundation_epochs),
                "CODEBOOK_SIZE": str(args.body_codebook_size),
                "NUM_QUANTIZERS": str(args.num_body_quantizers),
                "MAX_AUDIO_TOKEN": str(args.max_audio_token),
                "MAX_LEN_TOKEN": str(args.max_len_token),
            },
            outputs=[motion_foundation_dir / "config.json"],
            any_outputs=[hf_weight_outputs(motion_foundation_dir), tokenizer_outputs(motion_foundation_dir)],
        ),
        Stage(
            "build-planner-jsonl",
            "Build SuSu planner SFT JSONL: action + sparse audio tokens -> sparse motion tokens.",
            ["bash", "scripts/build_llm_sft_dataset.sh"],
            env={
                "MOTION_TOKEN_DIR": str(data_dir / "motion_token_data"),
                "AUDIO_TOKEN_DIR": str(data_dir / "audio_tokens_hubert_layer9_fps10"),
                "MOTION2TEXT_JSON": str(motion2text),
                "TRAIN_SPLIT": str(train_split),
                "OUTPUT_JSONL": str(planner_jsonl),
                "STEP": str(args.step),
                "CODEBOOK_SIZE": str(args.body_codebook_size),
                "NUM_QUANTIZERS": str(args.num_body_quantizers),
                "MAX_AUDIO_TOKEN": str(args.max_audio_token),
                "MAX_LEN_TOKEN": str(args.max_len_token),
                "LENGTH_MISMATCH_POLICY": args.length_mismatch_policy,
            },
            outputs=[planner_jsonl],
        ),
        Stage(
            "build-continuation-jsonl",
            "Build appendix-style continuation SFT JSONL with previous keyframe prefix.",
            [
                python,
                "motion_generation/training/build_llm_sft_dataset.py",
                "--motion_token_dir",
                str(data_dir / "motion_token_data"),
                "--audio_token_dir",
                str(data_dir / "audio_tokens_hubert_layer9_fps10"),
                "--motion2text_json",
                str(motion2text),
                "--split_file",
                str(train_split),
                "--output_jsonl",
                str(continuation_jsonl),
                "--step",
                str(args.step),
                "--codebook_size",
                str(args.body_codebook_size),
                "--num_quantizers",
                str(args.num_body_quantizers),
                "--continuation_prefix_keyframes",
                str(args.continuation_prefix_keyframes),
                "--max_audio_token",
                str(args.max_audio_token),
                "--max_len_token",
                str(args.max_len_token),
                "--length_mismatch_policy",
                args.length_mismatch_policy,
            ],
            outputs=[continuation_jsonl],
        ),
        Stage(
            "merge-planner-jsonl",
            "Merge regular and continuation planner SFT JSONL files for continuation-aware planner training.",
            [
                python,
                "scripts/merge_llm_jsonl.py",
                "--inputs",
                f"regular={planner_jsonl}",
                f"continuation={continuation_jsonl}",
                "--output_jsonl",
                str(planner_mixed_jsonl),
                "--summary_json",
                str(data_dir / "llm_sft" / f"train_step{args.step}_mixed_summary.json"),
                "--shuffle",
                "--seed",
                str(args.seed),
            ],
            outputs=[planner_mixed_jsonl],
        ),
        Stage(
            "train-llm-planner",
            f"Fine-tune Qwen planner on SuSu action/audio-to-motion examples ({args.planner_sft_mix}).",
            [
                "bash",
                "scripts/train_llm_planner.sh",
                "--train_jsonl",
                str(planner_train_jsonl),
                "--output_dir",
                str(llm_dir),
                "--batch_size",
                str(args.llm_batch_size),
                "--grad_accum_steps",
                str(args.llm_grad_accum_steps),
                "--max_length",
                str(args.llm_max_length),
                "--step_tokens",
                planner_step_tokens_csv,
            ],
            env={
                "INIT_MODEL": str(args.init_llm if args.no_motion_foundation else motion_foundation_dir),
                "DEVICE": args.device,
                "EPOCHS": str(args.llm_epochs),
                "CODEBOOK_SIZE": str(args.body_codebook_size),
                "NUM_QUANTIZERS": str(args.num_body_quantizers),
                "MAX_AUDIO_TOKEN": str(args.max_audio_token),
                "MAX_LEN_TOKEN": str(args.max_len_token),
            },
            outputs=[llm_dir / "config.json"],
            any_outputs=[hf_weight_outputs(llm_dir), tokenizer_outputs(llm_dir)],
        ),
        Stage(
            "train-body-infill",
            "Train audio-aware body Infill Transformer.",
            [
                "bash",
                "scripts/train_infill_transformer.sh",
                "--num_frames",
                str(args.step + 1),
            ],
            env={
                "MOTION_TOKEN_DIR": str(data_dir / "motion_token_data"),
                "AUDIO_FEAT_DIR": str(data_dir / "audio_features_hubert_layer9_fps10"),
                "TRAIN_SPLIT": str(train_split),
                "OUTPUT_DIR": str(mask_ckpt),
                "DEVICE": args.device,
                "BATCH_SIZE": str(args.infill_batch_size),
                "EPOCHS": str(args.infill_epochs),
                "CODEBOOK_SIZE": str(args.body_codebook_size),
                "NUM_TOKENS_PER_FRAME": str(args.num_body_quantizers),
                "LENGTH_MISMATCH_POLICY": args.length_mismatch_policy,
                "RANDOM_REPLACE_SCOPE": args.infill_random_replace_scope,
            },
            outputs=[mask_ckpt / "config.json"],
            any_outputs=[hf_weight_outputs(mask_ckpt)],
        ),
        Stage(
            "train-face-rvqvae",
            "Train paper-style Face R-VQVAE tokenizer.",
            ["bash", "scripts/train_face_rvqvae.sh"],
            env={
                "DATA_ROOT": str(data_dir),
                "FACE_DIR": str(data_dir / "arkit_data"),
                "TRAIN_SPLIT": str(train_split),
                "OUTPUT_DIR": str(face_rvqvae_dir),
                "DEVICE": args.device,
                "BATCH_SIZE": str(args.face_rvqvae_batch_size),
                "EPOCHS": str(args.face_rvqvae_epochs),
                "CODEBOOK_SIZE": str(args.face_codebook_size),
                "NUM_QUANTIZERS": str(args.num_face_quantizers),
            },
            outputs=[face_rvqvae_ckpt],
        ),
        Stage(
            "preprocess-face-tokens",
            f"Encode ARKit face data into {args.num_face_quantizers}-token Face R-VQVAE residual groups.",
            ["bash", "scripts/preprocess_face_tokens.sh"],
            env={
                "DATA_ROOT": str(data_dir),
                "FACE_DIR": str(data_dir / "arkit_data"),
                "AUDIO_FEAT_DIR": str(data_dir / "audio_features_hubert_layer9_fps10"),
                "SPLIT_FILE": str(all_split),
                "OUTPUT_DIR": str(data_dir / "face_token_data"),
                "FACE_RVQVAE_CKPT": str(face_rvqvae_ckpt),
                "DEVICE": args.device,
            },
            outputs=[data_dir / "face_token_data"],
        ),
        Stage(
            "train-face-infill",
            "Train Face Infill Transformer directly from audio features.",
            ["bash", "scripts/train_face_infill_transformer.sh"],
            env={
                "FACE_TOKEN_DIR": str(data_dir / "face_token_data"),
                "AUDIO_FEAT_DIR": str(data_dir / "audio_features_hubert_layer9_fps10"),
                "TRAIN_SPLIT": str(train_split),
                "OUTPUT_DIR": str(face_infill_ckpt),
                "DEVICE": args.device,
                "BATCH_SIZE": str(args.face_infill_batch_size),
                "EPOCHS": str(args.face_infill_epochs),
                "CODEBOOK_SIZE": str(args.face_codebook_size),
                "NUM_TOKENS_PER_FRAME": str(args.num_face_quantizers),
                "LENGTH_MISMATCH_POLICY": args.length_mismatch_policy,
                "RANDOM_REPLACE_SCOPE": args.infill_random_replace_scope,
            },
            outputs=[face_infill_ckpt / "config.json"],
            any_outputs=[hf_weight_outputs(face_infill_ckpt)],
        ),
        Stage(
            "audit-final",
            "Validate preprocessed data and trained checkpoint layout.",
            [
                python,
                "scripts/audit_reproduction_data.py",
                "--data_dir",
                str(data_dir),
                "--motion2text_json",
                str(motion2text),
                "--require_preprocessed",
                "--max_samples",
                str(args.audit_samples),
                "--max_audio_token",
                str(args.max_audio_token),
                "--max_len_token",
                str(args.max_len_token),
                "--expected_step",
                str(args.step),
                "--step_tokens",
                planner_step_tokens_csv,
                "--num_body_quantizers",
                str(args.num_body_quantizers),
                "--body_codebook_size",
                str(args.body_codebook_size),
                "--num_face_quantizers",
                str(args.num_face_quantizers),
                "--face_codebook_size",
                str(args.face_codebook_size),
                "--rvqvae_ckpt",
                str(rvqvae_ckpt),
                "--mask_ckpt",
                str(mask_ckpt),
                *motion_foundation_audit_args,
                "--llm_dir",
                str(llm_dir),
                *face_audit_args,
                *("--check_face".split() if check_face else []),
            ],
        ),
        Stage(
            "infer-batch",
            "Run GitHub-compatible batch inference with the trained planner and body Infill Transformer.",
            [
                python,
                "motion_generation/pipeline_infer.py",
                "--mask_ckpt",
                str(mask_ckpt),
                "--vllm_url",
                vllm_url,
                "--device",
                args.device,
                "--mode",
                "batch",
                "--generate_steps",
                str(args.generate_steps),
                "--temperature",
                str(args.infer_temperature),
                "--top_p",
                str(args.infer_top_p),
                "--llm_max_tokens",
                str(args.llm_max_tokens),
                "--keyframe_len_policy",
                args.keyframe_len_policy,
                "--step",
                str(args.step),
                "--motion_token_dir",
                str(data_dir / "motion_token_data"),
                "--audio_token_dir",
                str(data_dir / "audio_tokens_hubert_layer9_fps10"),
                "--audio_feat_dir",
                str(data_dir / "audio_features_hubert_layer9_fps10"),
                "--val_split_file",
                str(test_split),
                "--motion2text_json",
                str(motion2text),
                *infer_max_samples_args,
                "--output_path",
                str(pipeline_batch_json),
            ],
            outputs=[pipeline_batch_json],
        ),
        Stage(
            "reconstruct-batch",
            "Decode dense motion tokens to BVH/JSON/NPY for viewing and evaluation.",
            [
                python,
                "motion_generation/reconstruct_from_tokens.py",
                "--input_json",
                str(pipeline_batch_json),
                "--checkpoint_path",
                str(rvqvae_ckpt),
                *placeholder_args,
                "--output_dir",
                str(reconstructed_dir),
                "--device",
                args.device,
                "--wave_folder",
                str(data_dir / "wav_data"),
                "--tgt_fps",
                str(args.tgt_fps),
                *face_reconstruct_args,
                *reconstruct_max_samples_args,
            ],
            outputs=[reconstructed_dir],
        ),
        Stage(
            "eval-sync",
            "Compute lightweight ESD/BAS/BHR/VOC synchronization metrics on reconstructed predictions.",
            [
                python,
                "scripts/eval_sync_metrics.py",
                "--motion_dir",
                str(reconstructed_dir),
                "--wav_dir",
                str(data_dir / "wav_data"),
                "--motion_type",
                "pred",
                "--output_json",
                str(sync_metrics_json),
                *eval_max_samples_args,
            ],
            outputs=[sync_metrics_json],
        ),
        Stage(
            "eval-beatv2",
            "Compute BEATv2-style FGD/BC/Diversity metrics from reconstructed pred/gt npy files.",
            [
                python,
                "scripts/eval_beatv2_metrics.py",
                "--pred_dir",
                str(reconstructed_dir),
                "--gt_dir",
                str(reconstructed_dir),
                "--wav_dir",
                str(data_dir / "wav_data"),
                "--output_json",
                str(beatv2_metrics_json),
                "--max_feature_dims",
                str(args.beatv2_max_feature_dims),
                "--bc_sigma",
                str(args.beatv2_bc_sigma),
                *beatv2_feature_model_args,
                *eval_max_samples_args,
            ],
            outputs=[beatv2_metrics_json],
        ),
        Stage(
            "eval-full",
            "Run the full evaluation entry point when ChronAccRet/eval dependencies are available.",
            [
                "bash",
                "scripts/run_eval.sh",
                str(reconstructed_dir),
                args.eval_gpu_id,
            ],
            env={
                "OUTPUT_DIR": str(eval_output_dir),
                "MOTION2TEXT_PATH": str(motion2text),
                "WAV_DIR": str(data_dir / "wav_data"),
                "EVAL_MODEL_PATH": str(Path(args.eval_model_path).resolve()),
                "STATS_DIR": str(Path(args.eval_stats_dir).resolve()),
                "GT_MOTION_DIR": str(eval_gt_motion_dir),
                "REAL_GT_MOTION_DIR": str(eval_real_gt_motion_dir),
                **eval_text_model_env,
                "PYTHON_BIN": python,
            },
        ),
    ]
    if args.no_motion_foundation:
        stages = [stage for stage in stages if stage.name not in {"build-foundation-jsonl", "train-motion-foundation"}]
    if args.planner_sft_mix != "both":
        stages = [stage for stage in stages if stage.name != "merge-planner-jsonl"]
    if args.skip_face:
        stages = [
            stage for stage in stages
            if stage.name not in {"train-face-rvqvae", "preprocess-face-tokens", "train-face-infill"}
        ]
    return stages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or print staged SentiAvatar reproduction commands")
    parser.add_argument("--run", action="store_true", help="Execute stages. Default is dry-run.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip stages whose declared outputs already exist")
    parser.add_argument("--overwrite_preprocess", action="store_true",
                        help="Pass --overwrite to scripts/preprocess_data.sh when running the preprocess stage")
    parser.add_argument("--stages", default="all", help="Comma-separated stage names, or 'all'")
    parser.add_argument("--list_stages", action="store_true")
    parser.add_argument("--data_dir", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--checkpoints_dir", default=str(PROJECT_ROOT / "checkpoints_train"))
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "output"))
    parser.add_argument("--logs_dir", default=str(PROJECT_ROOT / "logs"))
    parser.add_argument("--train_split", default=None)
    parser.add_argument("--test_split", default=None)
    parser.add_argument("--all_split", default=None)
    parser.add_argument("--motion2text_json", default=None)
    parser.add_argument("--foundation_manifest_json", default=None,
                        help="Optional multi-source manifest for Motion Foundation text-to-motion JSONL")
    parser.add_argument("--hubert_path", default=None)
    parser.add_argument("--kmeans_path", default=None)
    parser.add_argument("--dataset_name", default="susuinteracts")
    parser.add_argument("--rvqvae_run_name", default="rvqvae_body")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--init_llm", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--no_motion_foundation", action="store_true",
                        help="Skip Motion Foundation pre-training and fine-tune planner from --init_llm")
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--continuation_prefix_keyframes", type=int, default=2)
    parser.add_argument("--audit_samples", type=int, default=5)
    parser.add_argument("--skip_face", action="store_true",
                        help="Skip paper-style face tokenizer/infill stages and reconstruct without face animation")
    parser.add_argument("--skip_face_audit", action="store_true",
                        help="Skip face data/checkpoint audit and default reconstruction to face_mode=none")
    parser.add_argument("--rvqvae_epochs", type=int, default=100)
    parser.add_argument("--rvqvae_batch_size", type=int, default=128)
    parser.add_argument("--body_codebook_size", type=int, default=512)
    parser.add_argument("--num_body_quantizers", "--body_num_quantizers", dest="num_body_quantizers", type=int, default=4)
    parser.add_argument("--rvqvae_code_dim", type=int, default=512)
    parser.add_argument("--rvqvae_down_t", type=int, default=1)
    parser.add_argument("--rvqvae_stride_t", type=int, default=2)
    parser.add_argument("--rvqvae_width", type=int, default=512)
    parser.add_argument("--rvqvae_depth", type=int, default=3)
    parser.add_argument("--rvqvae_dilation_growth_rate", type=int, default=3)
    parser.add_argument("--rvqvae_vq_cnn_depth", type=int, default=3)
    parser.add_argument("--foundation_epochs", type=int, default=10)
    parser.add_argument("--llm_epochs", type=int, default=10)
    parser.add_argument("--llm_batch_size", type=int, default=1)
    parser.add_argument("--llm_grad_accum_steps", type=int, default=128)
    parser.add_argument("--llm_max_length", type=int, default=1600)
    parser.add_argument("--max_audio_token", type=int, default=2048)
    parser.add_argument("--max_len_token", type=int, default=2048)
    parser.add_argument("--planner_step_tokens", default="1,2,3,4,5,6,8",
                        help="Comma-separated [step_t] tokens to add/check in the LLM planner tokenizer; current --step is added automatically")
    parser.add_argument("--planner_sft_mix", choices=["regular", "continuation", "both"], default="both",
                        help="Which SuSu planner SFT JSONL to train on; both mixes normal and continuation examples")
    parser.add_argument("--length_mismatch_policy", choices=["strict", "truncate"], default="strict",
                        help="How trainer/data builders handle token/audio length mismatches; strict is the GitHub inference contract")
    parser.add_argument("--infill_epochs", type=int, default=100)
    parser.add_argument("--infill_batch_size", type=int, default=1024)
    parser.add_argument("--infill_random_replace_scope", choices=["unmasked", "legacy_supervised"], default="unmasked",
                        help="Infill training random corruption scope; unmasked matches the paper")
    parser.add_argument("--face_rvqvae_epochs", type=int, default=100)
    parser.add_argument("--face_rvqvae_batch_size", type=int, default=128)
    parser.add_argument("--face_codebook_size", type=int, default=512)
    parser.add_argument("--num_face_quantizers", "--face_num_quantizers", dest="num_face_quantizers", type=int, default=2)
    parser.add_argument("--face_infill_epochs", type=int, default=100)
    parser.add_argument("--face_infill_batch_size", type=int, default=1024)
    parser.add_argument("--vllm_url", default=None,
                        help="Full vLLM server URL for infer-batch; default uses --vllm_port")
    parser.add_argument("--vllm_port", type=int, default=8095)
    parser.add_argument("--infer_temperature", type=float, default=0.5)
    parser.add_argument("--infer_top_p", type=float, default=0.4)
    parser.add_argument("--generate_steps", type=int, default=6)
    parser.add_argument("--llm_max_tokens", type=int, default=1024)
    parser.add_argument("--keyframe_len_policy", choices=["strict", "warn"], default="strict")
    parser.add_argument("--infer_max_samples", type=int, default=None,
                        help="Limit infer-batch and reconstruct-batch to the first N test split entries")
    parser.add_argument("--placeholder_npy", default=None,
                        help="Optional placeholder motion npy for reconstruct-batch hand data")
    parser.add_argument("--face_mode", choices=["none", "infill", "legacy"], default=None,
                        help="Face mode for reconstruct-batch; default is infill unless --skip_face_audit is set")
    parser.add_argument("--tgt_fps", type=float, default=30.0)
    parser.add_argument("--eval_max_samples", type=int, default=None)
    parser.add_argument("--eval_gpu_id", default="0")
    parser.add_argument("--eval_model_path", default=str(PROJECT_ROOT / "checkpoints" / "eval_model" / "best_model.pt"))
    parser.add_argument("--eval_stats_dir", default=str(PROJECT_ROOT / "evaluation" / "stats" / "humanml3d" / "guoh3dfeats"))
    parser.add_argument("--eval_gt_motion_dir", default=None,
                        help="Directory containing *_gt.npy for FID/Diversity; default is reconstruct-batch output dir")
    parser.add_argument("--eval_real_gt_motion_dir", default=None,
                        help="Directory containing raw real GT motion .npy for GT sync upper-bound; default is data/motion_data")
    parser.add_argument("--eval_text_model_name", default=None,
                        help="Override ChronTMR text encoder/tokenizer model, e.g. a local bert-base-chinese directory")
    parser.add_argument("--beatv2_feature_model_path", default=None,
                        help="Optional TorchScript gesture feature extractor for eval-beatv2 FGD/Diversity")
    parser.add_argument("--beatv2_max_feature_dims", type=int, default=256,
                        help="Handcrafted feature projection size for eval-beatv2 when no feature model is provided")
    parser.add_argument("--beatv2_bc_sigma", type=float, default=0.3,
                        help="Gaussian sigma in seconds for eval-beatv2 beat consistency")
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    all_stages = build_stages(args)
    by_name = {stage.name: stage for stage in all_stages}
    if args.list_stages:
        for stage in all_stages:
            print(f"{stage.name}: {stage.description}")
        return 0

    if args.stages == "all":
        selected = all_stages
    else:
        requested = split_csv(args.stages)
        missing = [name for name in requested if name not in by_name]
        if missing:
            raise ValueError(f"Unknown stages: {', '.join(missing)}")
        selected = [by_name[name] for name in requested]

    dry_run = not args.run
    if dry_run:
        print("Dry run: commands are printed but not executed. Add --run to execute.")
    for stage in selected:
        run_stage(stage, dry_run=dry_run, skip_existing=args.skip_existing)
    return 0


if __name__ == "__main__":
    sys.exit(main(parse_args()))
