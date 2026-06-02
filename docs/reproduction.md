# SentiAvatar Reproduction Notes

This repository now exposes trainable entry points for the parts used by the public GitHub inference path:

1. Motion R-VQVAE tokenizes 20 FPS body motion into 4 residual codebook IDs per token frame.
2. Qwen-based LLM planner predicts sparse keyframes at step `t=4`.
3. Audio-aware Infill Transformer fills the 3 interior frames in each 5-frame token window using HuBERT layer9 features.
4. Existing reconstruction scripts decode dense body tokens to BVH/JSON and can attach either the legacy released face path or the paper-style Face Infill path.

The paper states HuBERT continuous features at 20 FPS, but the released inference code uses `audio_features_hubert_layer9_fps10` to match RVQVAE token-rate frames. The training scripts keep that GitHub-compatible contract.

## Synthetic Smoke Test

Before using the real dataset, run the CPU smoke test:

```bash
python scripts/smoke_reproduction.py --work_dir /tmp/sentiavatar_smoke_auto
python scripts/smoke_reproduction.py --work_dir /tmp/sentiavatar_smoke_auto_llm --include_llm_train
```

It creates a tiny synthetic dataset under `/tmp`, trains each component for one step, and verifies that saved artifacts are loadable by the GitHub inference helpers:

- Motion R-VQVAE checkpoint layout, local normalization stats, and `load_model()`.
- Token reconstruction to JSON/BVH plus `*_pred.npy`/`*_gt.npy` files consumed by evaluation.
- Lightweight ESD/BAS/BHR/VOC synchronization metrics from generated `*_pred.npy` and wav files.
- LLM planner JSONL for both Motion Foundation-style text-to-motion and SuSu SFT audio-aware prompts, optional two-stage tiny CausalLM training, plus `pipeline_infer.py` token parsing.
- The GitHub inference path `run_pipeline_single()` using a fake planner client and a trained body Infill Transformer.
- Body Infill Transformer and Face Infill Transformer Hugging Face-style checkpoints via `load_mask_transformer()`.
- Face R-VQVAE token preprocessing and `FaceInfillPipeline` output injection into animation JSON.

Use `--include_llm_train` to also run one-step Motion Foundation training followed by one-step SuSu planner SFT from that checkpoint. By default it creates a local `auto-tiny` Hugging Face fixture under the smoke work directory, so the check stays offline; pass `--llm_model <local_or_hf_model>` only when you explicitly want to test another base model. The smoke test does not prove paper metrics.

## Real Data Stage Runner

For full real-data reproduction, first inspect the staged plan:

```bash
python scripts/reproduce_pipeline.py --list_stages
python scripts/reproduce_pipeline.py --stages all
```

The runner defaults to dry-run and never deletes files. Add `--run` to execute stages, and use `--skip_existing` to resume around completed outputs:

```bash
python scripts/reproduce_pipeline.py \
  --run \
  --skip_existing \
  --data_dir data \
  --checkpoints_dir checkpoints_train \
  --device cuda:0 \
  --llm_max_length 1600
```

Use `--overwrite_preprocess` when refreshing audio features/tokens created before the current RVQVAE checkpoint or before audio-to-motion length alignment was enabled.

After training the LLM planner, start vLLM in a separate terminal before running inference stages:

```bash
MAX_MODEL_LEN=2400 MAX_TOKENS_LIMIT=2400 \
  bash scripts/start_vllm_server.sh checkpoints_train/llm 8095 0
```

Common partial runs:

```bash
# Build planner data and train only the body infill model.
python scripts/reproduce_pipeline.py \
  --run \
  --stages build-planner-jsonl,train-body-infill

# Skip Motion Foundation pre-training and fine-tune planner directly from Qwen.
python scripts/reproduce_pipeline.py \
  --run \
  --no_motion_foundation \
  --stages build-planner-jsonl,train-llm-planner

# Small real-data inference/evaluation pass after vLLM is running.
python scripts/reproduce_pipeline.py \
  --run \
  --skip_existing \
  --stages infer-batch,reconstruct-batch,eval-sync \
  --vllm_url http://localhost:8095 \
  --infer_max_samples 8 \
  --eval_max_samples 8

# Body-only ablation/debug path when ARKit face data is not available.
python scripts/reproduce_pipeline.py \
  --run \
  --skip_face \
  --stages train-rvqvae,preprocess,build-planner-jsonl,train-llm-planner,train-body-infill,audit-final
```

## Real Data Audit

After preprocessing real data, validate the GitHub inference contract before long training runs:

```bash
python scripts/audit_reproduction_data.py \
  --data_dir data \
  --require_preprocessed \
  --check_face \
  --rvqvae_ckpt checkpoints_train/susuinteracts/rvqvae_body/model/latest.pth \
  --mask_ckpt checkpoints_train/mask_transformer \
  --motion_foundation_dir checkpoints_train/motion_foundation \
  --llm_dir checkpoints_train/llm \
  --face_rvqvae_ckpt checkpoints_train/face_rvqvae/latest.pth \
  --face_infill_ckpt checkpoints_train/face_infill_transformer
```

The audit checks sample splits, motion dimensions, audio feature/token lengths, 4-token body residual groups, 2-token face residual groups, token value ranges, RVQVAE/HF checkpoint layout, body and face codec/Infill compatibility, LLM tokenizer coverage, and token-rate metadata. Body Infill `num_tokens_per_frame`, `codebook_size`, `vocab_size`, and `num_frames - 1` must match the RVQVAE and planner step. Face Infill `num_tokens_per_frame`, `codebook_size`, and `vocab_size` must match the Face R-VQVAE. The planner tokenizer must contain `[audio_*]`, `[res1_*]`...`[res4_*]`, `[len_*]`, and `[step_*]` as atomic tokens; align `--max_audio_token`, `--max_len_token`, `--step_tokens`, `--body_codebook_size`, `--num_body_quantizers`, `--face_codebook_size`, and `--num_face_quantizers` with the training run when using non-default K-means, codec sizes, or long clips. For GitHub inference, `motion_token_data/**/*.json` should carry `source_fps`, `token_fps`, and `downsample_factor`; with the released 20 FPS motion and 2x RVQVAE downsampling, `token_fps` should be `10.0`.
Planner JSONL building and Infill training default to strict length checks: audio tokens/features and motion tokens must share the same 10 FPS token timeline. Use `--length_mismatch_policy truncate` or `LENGTH_MISMATCH_POLICY=truncate` only when intentionally auditing legacy preprocessed data.

## Data Layout

Expected local data:

```text
data/
  motion_data/**/*.npy
  wav_data/**/*.wav
  text_data/motion2text.json
  split/train_file_list.txt
  split/test_file_list.txt
```

Each motion `.npy` must be a dict with `body` `(T,153)`, `left` `(T,120)`, and `right` `(T,120)`.
`split/all_file_list.txt` is optional: preprocessing uses it when present, otherwise it combines train/val/test splits and finally falls back to discovering files under `wav_data` or `motion_data`.
`text_data/motion2text.json` can map each sample to the released string format, such as `【表情：开心】【动作：挥手】你好`, or to a structured object with fields like `expression`, `action`, and `dialogue`; planner training and batch inference normalize both forms before extracting the action tag.

## Stage 1: Train Motion R-VQVAE

```bash
CODEBOOK_SIZE=512 NUM_QUANTIZERS=4 bash scripts/train_rvqvae.sh
```

The wrapper also exposes the structure parameters saved into `opt.txt`, for example
`DOWN_T=1 STRIDE_T=2 WIDTH=512 DEPTH=3 CODE_DIM=512`. The staged runner has matching
flags such as `--rvqvae_down_t`, `--rvqvae_stride_t`, `--rvqvae_width`, and
`--rvqvae_depth`.

Outputs:

```text
checkpoints_train/susuinteracts/rvqvae_body/
  opt.txt
  meta/mean.npy
  meta/std.npy
  model/latest.pth
```

The checkpoint layout matches `load_config_from_checkpoint()` and `load_model()`.

## Stage 2: Preprocess Tokens and Audio Features

```bash
bash scripts/preprocess_data.sh --all \
  --rvqvae_ckpt checkpoints_train/susuinteracts/rvqvae_body/model/latest.pth \
  --hubert_path checkpoints/chinese-hubert-base \
  --kmeans_path checkpoints/hubert_kmeans/model.mdl \
  --overwrite \
  --strict
```

This creates `data/motion_token_data`, `data/audio_tokens_hubert_layer9_fps10`, and `data/audio_features_hubert_layer9_fps10`. Use the RVQVAE checkpoint trained in Stage 1 so motion tokens match the decoder used by GitHub inference. Audio preprocessing defaults to resampling HuBERT features to the RVQVAE motion-token length when matching `motion_data` is available; pass `--no_align_audio_to_motion` only for audio-only preprocessing. Use `--overwrite` when regenerating from old intermediate files. `--strict` makes real-data preprocessing fail fast if any requested sample is missing or cannot be encoded. If `--device cuda:*` is requested on a CPU-only machine, preprocessing falls back to CPU for smoke/debug runs.
Downstream data builders also fail fast on length mismatches, so old preprocessed files should be regenerated with `--overwrite` rather than silently truncated.

## Stage 3: Train Motion Foundation Model and LLM Planner

```bash
# Text-to-motion pre-training data, matching the paper's Motion Foundation objective.
bash scripts/build_motion_foundation_dataset.sh
bash scripts/train_llm_planner.sh \
  --init_model Qwen/Qwen2-0.5B \
  --train_jsonl data/llm_sft/foundation_step1.jsonl \
  --output_dir checkpoints_train/motion_foundation \
  --max_length 1600 \
  --batch_size 1 \
  --grad_accum_steps 128

# SuSuInterActs planner SFT data: action label + sparse audio tokens -> sparse keyframe tokens.
bash scripts/build_llm_sft_dataset.sh
python motion_generation/training/build_llm_sft_dataset.py \
  --motion_token_dir data/motion_token_data \
  --audio_token_dir data/audio_tokens_hubert_layer9_fps10 \
  --motion2text_json data/text_data/motion2text.json \
  --split_file data/split/train_file_list.txt \
  --output_jsonl data/llm_sft/train_step4_continuation.jsonl \
  --step 4 \
  --continuation_prefix_keyframes 2
python scripts/merge_llm_jsonl.py \
  --inputs regular=data/llm_sft/train_step4.jsonl continuation=data/llm_sft/train_step4_continuation.jsonl \
  --output_jsonl data/llm_sft/train_step4_mixed.jsonl \
  --shuffle
bash scripts/train_llm_planner.sh \
  --init_model checkpoints_train/motion_foundation \
  --train_jsonl data/llm_sft/train_step4_mixed.jsonl \
  --max_length 1600 \
  --batch_size 1 \
  --grad_accum_steps 128
```

The dataset builder writes prompts in the same format consumed by `pipeline_infer.py`:

```text
Human: 动作：点头[audio_...]\nAssistant:
[step_4][len_N][res1_...][res2_...][res3_...][res4_...]...
```

By default, `build_motion_foundation_dataset.sh` uses the raw `motion2text` text for text-to-motion Motion Foundation pre-training, while `build_llm_sft_dataset.sh` extracts compact action labels for SuSu planner SFT. Override with `--prompt_text_mode raw` or `--prompt_text_mode action` only when intentionally changing that objective. `scripts/reproduce_pipeline.py` defaults to `--planner_sft_mix both`, so the planner SFT trains on regular and continuation examples; use `--planner_sft_mix regular` for the old single-turn-only path.

For the paper-scale 200K+ Motion Foundation pre-training mix, create a manifest after each external corpus has been retargeted and tokenized by the same body R-VQVAE:

```json
{
  "sources": [
    {
      "name": "embodyai",
      "motion_token_dir": "/data/foundation/embodyai/motion_token_data",
      "motion2text_json": "/data/foundation/embodyai/text_data/motion2text.json",
      "split_file": "/data/foundation/embodyai/split/train_file_list.txt"
    },
    {
      "name": "snapmogen",
      "motion_token_dir": "/data/foundation/snapmogen/motion_token_data",
      "motion2text_json": "/data/foundation/snapmogen/text_data/motion2text.json"
    },
    {
      "name": "motionx",
      "motion_token_dir": "/data/foundation/motionx/motion_token_data",
      "motion2text_json": "/data/foundation/motionx/text_data/motion2text.json"
    },
    {
      "name": "hunyuan_distill",
      "motion_token_dir": "/data/foundation/hunyuan_distill/motion_token_data",
      "motion2text_json": "/data/foundation/hunyuan_distill/text_data/motion2text.json"
    }
  ]
}
```

Then build the mixed text-to-motion JSONL:

```bash
FOUNDATION_MANIFEST_JSON=/data/foundation/foundation_manifest.json \
  bash scripts/build_motion_foundation_dataset.sh
```

`scripts/reproduce_pipeline.py` exposes the same path as `--foundation_manifest_json`. Each generated JSONL row records `source` and `source_sample` so the 200K corpus mix can be audited.

LLM training fails fast if any prompt/completion exceeds `--max_length`, because silent truncation can remove supervised motion tokens. It also validates every JSONL completion before loading the model: `[step_t]` and `[len_N]` must match metadata, residual tokens must appear as `[res1_*]...[res4_*]` groups, and all IDs must be in range. Increase `--llm_max_length` in `reproduce_pipeline.py` or `MAX_LENGTH`/`--max_length` for direct script runs when using longer clips.

Planner JSONL construction and LLM training also fail fast when discrete tokens fall outside the tokenizer vocabulary contract. Keep `MAX_AUDIO_TOKEN` aligned with the HuBERT K-means codebook IDs and `MAX_LEN_TOKEN` above the largest sparse keyframe count:

```bash
CODEBOOK_SIZE=512 NUM_QUANTIZERS=4 MAX_AUDIO_TOKEN=2048 MAX_LEN_TOKEN=2048 bash scripts/build_llm_sft_dataset.sh
CODEBOOK_SIZE=512 NUM_QUANTIZERS=4 MAX_AUDIO_TOKEN=2048 MAX_LEN_TOKEN=2048 STEP_TOKENS=1,2,3,4,5,6,8 MAX_LENGTH=1600 bash scripts/train_llm_planner.sh
```

For non-default planner intervals, include the interval in `STEP_TOKENS` for direct script runs, or pass `--planner_step_tokens` to `scripts/reproduce_pipeline.py`; the runner also adds the current `--step` automatically.

For the appendix continuation mode, simulate previous-turn context during SFT by moving the first sparse audio-motion keyframes into the prompt:

```bash
python motion_generation/training/build_llm_sft_dataset.py \
  --motion_token_dir data/motion_token_data \
  --audio_token_dir data/audio_tokens_hubert_layer9_fps10 \
  --motion2text_json data/text_data/motion2text.json \
  --split_file data/split/train_file_list.txt \
  --output_jsonl data/llm_sft/train_step4_continuation.jsonl \
  --step 4 \
  --continuation_prefix_keyframes 2
```

At inference, `pipeline_infer.py` can prepend previous-turn audio-motion keyframe pairs to the LLM prompt. When these prefix files are provided, the last sparse previous-turn motion keyframe is also used as the left boundary of the first current-turn Infill Transformer window. The current-turn sparse audio sampling starts at `step - 1`, matching the appendix setup where generation begins at the next keyframe after the context boundary; the returned dense tokens still contain only current utterance frames.

```bash
python motion_generation/pipeline_infer.py \
  --mask_ckpt checkpoints_train/mask_transformer \
  --prefix_audio_token_json data/audio_tokens_hubert_layer9_fps10/prev_turn.json \
  --prefix_audio_feat_npy data/audio_features_hubert_layer9_fps10/prev_turn.npy \
  --prefix_motion_token_json output/prev_turn_result.json \
  --continuation_prefix_keyframes 2
```

The trained model can be served by:

```bash
MAX_MODEL_LEN=2400 MAX_TOKENS_LIMIT=2400 bash scripts/start_vllm_server.sh checkpoints_train/llm 8095 0
```

## Stage 4: Train Audio-Aware Body Infill Transformer

```bash
CODEBOOK_SIZE=512 NUM_TOKENS_PER_FRAME=4 bash scripts/train_infill_transformer.sh
```

The trainer uses 5-frame windows, keeps boundary frames known, randomly masks interior tokens, randomly corrupts 10% of unmasked interior tokens, and excludes boundaries from loss. This is the default `RANDOM_REPLACE_SCOPE=unmasked`; use `legacy_supervised` only to reproduce earlier local experiments. Outputs are Hugging Face-style folders loadable by `load_mask_transformer()`.

For step ablations, train with `--num_frames t+1` and pass the matching `--step t` to `pipeline_infer.py` or `single_case_infer.py`. If `--step` is omitted, inference uses `mask_model.config.num_frames - 1`; mismatched planner step and infill checkpoint now fail early.

## Stage 5: Train Face R-VQVAE and Face Infill Transformer

The paper uses a Face R-VQVAE with 2 residual tokens per ARKit frame, followed by a Face Infill Transformer that generates face tokens directly from dense HuBERT features.

```bash
CODEBOOK_SIZE=512 NUM_QUANTIZERS=2 bash scripts/train_face_rvqvae.sh
bash scripts/preprocess_face_tokens.sh
CODEBOOK_SIZE=512 NUM_TOKENS_PER_FRAME=2 bash scripts/train_face_infill_transformer.sh
```

Expected outputs:

```text
checkpoints_train/face_rvqvae/latest.pth
data/face_token_data/**/*.json
checkpoints_train/face_infill_transformer/
```

`preprocess_face_tokens.sh` aligns `data/arkit_data` sequences to `data/audio_features_hubert_layer9_fps10` lengths before tokenization so the Face Infill Transformer uses the same GitHub-compatible audio feature rate as body inference.

## Inference and Evaluation

```bash
MAX_MODEL_LEN=2400 MAX_TOKENS_LIMIT=2400 bash scripts/start_vllm_server.sh checkpoints_train/llm 8095 0
bash scripts/run_single_infer.sh \
  --audio_path examples/demo.wav \
  --action_text "动作：张开双臂上下挥动，像鸟儿一样飞" \
  --mask_ckpt checkpoints_train/mask_transformer \
  --rvqvae_ckpt checkpoints_train/susuinteracts/rvqvae_body/model/latest.pth \
  --hubert_path checkpoints/chinese-hubert-base \
  --kmeans_path checkpoints/hubert_kmeans/model.mdl \
  --face_mode infill \
  --face_rvqvae_ckpt checkpoints_train/face_rvqvae/latest.pth \
  --face_infill_ckpt checkpoints_train/face_infill_transformer
MASK_CKPT=checkpoints_train/mask_transformer \
RVQVAE_CKPT=checkpoints_train/susuinteracts/rvqvae_body/model/latest.pth \
bash scripts/run_test.sh 8095 0
bash scripts/run_eval.sh output/reconstructed 0
```

Inference now checks the planner contract by default: parsed LLM keyframes must match the prompt's sampled audio keyframe count, and `[len_N]` must match the parsed residual groups. `single_case_infer.py` also checks that HuBERT token length equals HuBERT feature length, and that the body Infill checkpoint emits tokens compatible with the RVQVAE decoder. Use `--keyframe_len_policy warn` only when auditing legacy planner checkpoints that may emit off-length plans. For longer utterances, raise both server-side `MAX_MODEL_LEN`/`MAX_TOKENS_LIMIT` and client-side `--llm_max_tokens` so vLLM can emit all sparse keyframes instead of truncating.

For interactive multi-turn demos, each single-case run writes reusable continuation files: `<name>_audio_tokens.json`, `<name>_audio_features.npy`, `<name>_motion_tokens.json`, and `<name>_pipeline_result.json`. Pass the first three files into the next turn:

```bash
bash scripts/run_single_infer.sh \
  --audio_path examples/turn1.wav \
  --action_text "动作：点头回应" \
  --output_name turn1 \
  --output_dir output_interactive \
  --mask_ckpt checkpoints_train/mask_transformer \
  --rvqvae_ckpt checkpoints_train/susuinteracts/rvqvae_body/model/latest.pth

bash scripts/run_single_infer.sh \
  --audio_path examples/turn2.wav \
  --action_text "动作：摊手解释" \
  --output_name turn2 \
  --output_dir output_interactive \
  --mask_ckpt checkpoints_train/mask_transformer \
  --rvqvae_ckpt checkpoints_train/susuinteracts/rvqvae_body/model/latest.pth \
  --prefix_audio_token_json output_interactive/turn1_audio_tokens.json \
  --prefix_audio_feat_npy output_interactive/turn1_audio_features.npy \
  --prefix_motion_token_json output_interactive/turn1_motion_tokens.json \
  --continuation_prefix_keyframes 2
```

For batch reconstruction with the new face path, call:

```bash
python motion_generation/reconstruct_from_tokens.py \
  --input_json output/pipeline_batch_results.json \
  --checkpoint_path checkpoints_train/susuinteracts/rvqvae_body/model/latest.pth \
  --output_dir output/reconstructed \
  --face_mode infill \
  --face_rvqvae_ckpt checkpoints_train/face_rvqvae/latest.pth \
  --face_infill_ckpt checkpoints_train/face_infill_transformer \
  --audio_feat_dir data/audio_features_hubert_layer9_fps10
```

`reconstruct_from_tokens.py` saves BVH/JSON for viewing and also writes `*_pred.npy` / `*_gt.npy` feature dictionaries for evaluation. For a quick synchronization-only check that does not require the ChronTMR retrieval model, run:

```bash
python scripts/eval_sync_metrics.py \
  --motion_dir output/reconstructed \
  --wav_dir data/wav_data \
  --motion_type pred \
  --output_json output/eval_results/sync_metrics.json
```

The full `scripts/run_eval.sh` path computes retrieval, FID, Diversity, BAS, VOC, and ESD when the evaluation checkpoint, real `motion2text.json`, wav files, and Python dependencies from `requirements.txt` are available. The script is fail-fast: missing paths or missing modules such as `hydra`/`librosa` stop the run instead of printing a false success. Override paths with environment variables when evaluating an external output directory:

```bash
MOTION2TEXT_PATH=data/text_data/motion2text.json \
WAV_DIR=data/wav_data \
EVAL_MODEL_PATH=checkpoints/eval_model/best_model.pt \
STATS_DIR=evaluation/stats/humanml3d/guoh3dfeats \
GT_MOTION_DIR=output/reconstructed \
REAL_GT_MOTION_DIR=data/motion_data \
TEXT_MODEL_NAME=bert-base-chinese \
OUTPUT_DIR=output/eval_results \
bash scripts/run_eval.sh output/reconstructed 0
```

`GT_MOTION_DIR` should contain `*_gt.npy` files for FID/Diversity; it defaults to the reconstruction directory because `reconstruct_from_tokens.py` writes `*_pred.npy` and `*_gt.npy` side by side. `REAL_GT_MOTION_DIR` is used only for the optional real-motion BAS/ESD upper bound. Set `TEXT_MODEL_NAME` to a local BERT directory when running offline.

For BEATv2-style reporting, run:

```bash
python scripts/eval_beatv2_metrics.py \
  --pred_dir output/reconstructed \
  --gt_dir output/reconstructed \
  --wav_dir data/wav_data \
  --output_json output/eval_results/beatv2_metrics.json
```

This computes FGD, beat consistency (`BC` and `BC_x10`), and Diversity. Without `--feature_model_path`, FGD/Diversity use deterministic handcrafted motion statistics and are suitable for regression tests, not for claiming the official BEATv2 table. Pass a TorchScript gesture feature extractor with `--feature_model_path` to align FGD/Diversity with an official or project-specific embedding model.

When using `scripts/reproduce_pipeline.py`, the same settings are exposed as `--eval_gt_motion_dir`, `--eval_real_gt_motion_dir`, `--eval_text_model_name`, `--beatv2_feature_model_path`, `--beatv2_max_feature_dims`, and `--beatv2_bc_sigma`. The staged runner includes `eval-beatv2` after reconstruction.

## Remaining Fidelity Notes

The released checkpoints still use the legacy `checkpoints/face_vqvae` helper. Use `--face_mode legacy` for released-checkpoint compatibility and `--face_mode infill` for the paper-style reproduced face pipeline. Full paper-scale claims still require training on the real SuSuInterActs split and running the provided evaluation metrics.
