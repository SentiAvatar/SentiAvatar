#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

python "${PROJECT_DIR}/motion_generation/training/preprocess_face_tokens.py" \
  --data_root "${DATA_ROOT:-${PROJECT_DIR}/data}" \
  --face_dir "${FACE_DIR:-${PROJECT_DIR}/data/arkit_data}" \
  --audio_feat_dir "${AUDIO_FEAT_DIR:-${PROJECT_DIR}/data/audio_features_hubert_layer9_fps10}" \
  --split_file "${SPLIT_FILE:-${PROJECT_DIR}/data/split/all_file_list.txt}" \
  --output_dir "${OUTPUT_DIR:-${PROJECT_DIR}/data/face_token_data}" \
  --face_rvqvae_ckpt "${FACE_RVQVAE_CKPT:-${PROJECT_DIR}/checkpoints_train/face_rvqvae/latest.pth}" \
  --device "${DEVICE:-cuda:0}" \
  "$@"

