#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

python "${PROJECT_DIR}/motion_generation/training/train_infill_transformer.py" \
  --motion_token_dir "${FACE_TOKEN_DIR:-${PROJECT_DIR}/data/face_token_data}" \
  --audio_feat_dir "${AUDIO_FEAT_DIR:-${PROJECT_DIR}/data/audio_features_hubert_layer9_fps10}" \
  --train_split "${TRAIN_SPLIT:-${PROJECT_DIR}/data/split/train_file_list.txt}" \
  --output_dir "${OUTPUT_DIR:-${PROJECT_DIR}/checkpoints_train/face_infill_transformer}" \
  --device "${DEVICE:-cuda:0}" \
  --batch_size "${BATCH_SIZE:-1024}" \
  --epochs "${EPOCHS:-100}" \
  --num_tokens_per_frame "${NUM_TOKENS_PER_FRAME:-2}" \
  --codebook_size "${CODEBOOK_SIZE:-512}" \
  --length_mismatch_policy "${LENGTH_MISMATCH_POLICY:-strict}" \
  --random_replace_scope "${RANDOM_REPLACE_SCOPE:-unmasked}" \
  --boundary_mode none \
  "$@"
