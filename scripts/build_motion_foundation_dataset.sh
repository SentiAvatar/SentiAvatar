#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

EXTRA_ARGS=()
if [[ -n "${SOURCE_MANIFEST_JSON:-${FOUNDATION_MANIFEST_JSON:-}}" ]]; then
  EXTRA_ARGS+=(--source_manifest_json "${SOURCE_MANIFEST_JSON:-${FOUNDATION_MANIFEST_JSON:-}}")
fi

python "${PROJECT_DIR}/motion_generation/training/build_llm_sft_dataset.py" \
  --motion_token_dir "${MOTION_TOKEN_DIR:-${PROJECT_DIR}/data/motion_token_data}" \
  --motion2text_json "${MOTION2TEXT_JSON:-${PROJECT_DIR}/data/text_data/motion2text.json}" \
  --split_file "${TRAIN_SPLIT:-${PROJECT_DIR}/data/split/train_file_list.txt}" \
  --output_jsonl "${OUTPUT_JSONL:-${PROJECT_DIR}/data/llm_sft/foundation_step1.jsonl}" \
  --step "${STEP:-1}" \
  --codebook_size "${CODEBOOK_SIZE:-512}" \
  --num_quantizers "${NUM_QUANTIZERS:-4}" \
  --max_audio_token "${MAX_AUDIO_TOKEN:-2048}" \
  --max_len_token "${MAX_LEN_TOKEN:-2048}" \
  --length_mismatch_policy "${LENGTH_MISMATCH_POLICY:-strict}" \
  --no_audio \
  "${EXTRA_ARGS[@]}" \
  "$@"
