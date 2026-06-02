#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

python "${PROJECT_DIR}/motion_generation/training/train_llm_planner.py" \
  --init_model "${INIT_MODEL:-Qwen/Qwen2-0.5B}" \
  --train_jsonl "${TRAIN_JSONL:-${PROJECT_DIR}/data/llm_sft/train_step4.jsonl}" \
  --output_dir "${OUTPUT_DIR:-${PROJECT_DIR}/checkpoints_train/llm}" \
  --device "${DEVICE:-cuda:0}" \
  --epochs "${EPOCHS:-10}" \
  --max_length "${MAX_LENGTH:-1600}" \
  --codebook_size "${CODEBOOK_SIZE:-512}" \
  --num_quantizers "${NUM_QUANTIZERS:-4}" \
  --max_audio_token "${MAX_AUDIO_TOKEN:-2048}" \
  --max_len_token "${MAX_LEN_TOKEN:-2048}" \
  --step_tokens "${STEP_TOKENS:-1,2,3,4,5,6,8}" \
  --add_planner_tokens \
  "$@"
