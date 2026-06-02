#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

python "${PROJECT_DIR}/motion_generation/training/train_face_rvqvae.py" \
  --data_root "${DATA_ROOT:-${PROJECT_DIR}/data}" \
  --face_dir "${FACE_DIR:-${PROJECT_DIR}/data/arkit_data}" \
  --train_split "${TRAIN_SPLIT:-${PROJECT_DIR}/data/split/train_file_list.txt}" \
  --output_dir "${OUTPUT_DIR:-${PROJECT_DIR}/checkpoints_train/face_rvqvae}" \
  --device "${DEVICE:-cuda:0}" \
  --batch_size "${BATCH_SIZE:-128}" \
  --epochs "${EPOCHS:-100}" \
  --codebook_size "${CODEBOOK_SIZE:-512}" \
  --num_quantizers "${NUM_QUANTIZERS:-2}" \
  "$@"
