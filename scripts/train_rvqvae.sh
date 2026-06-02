#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

EXTRA_ARGS=()
if [[ -n "${VQ_NORM:-}" ]]; then
  EXTRA_ARGS+=(--vq_norm "${VQ_NORM}")
fi
if [[ "${SHARED_CODEBOOK:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--shared_codebook)
fi

python "${PROJECT_DIR}/motion_generation/training/train_rvqvae.py" \
  --data_root "${DATA_ROOT:-${PROJECT_DIR}/data}" \
  --train_split "${TRAIN_SPLIT:-${PROJECT_DIR}/data/split/train_file_list.txt}" \
  --checkpoints_dir "${CHECKPOINTS_DIR:-${PROJECT_DIR}/checkpoints_train}" \
  --dataset_name "${DATASET_NAME:-susuinteracts}" \
  --name "${RUN_NAME:-rvqvae_body}" \
  --device "${DEVICE:-cuda:0}" \
  --batch_size "${BATCH_SIZE:-128}" \
  --epochs "${EPOCHS:-100}" \
  --nb_code "${NB_CODE:-${CODEBOOK_SIZE:-512}}" \
  --code_dim "${CODE_DIM:-512}" \
  --down_t "${DOWN_T:-1}" \
  --stride_t "${STRIDE_T:-2}" \
  --width "${WIDTH:-512}" \
  --depth "${DEPTH:-3}" \
  --dilation_growth_rate "${DILATION_GROWTH_RATE:-3}" \
  --vq_act "${VQ_ACT:-relu}" \
  --vq_cnn_depth "${VQ_CNN_DEPTH:-3}" \
  --num_quantizers "${NUM_QUANTIZERS:-4}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
