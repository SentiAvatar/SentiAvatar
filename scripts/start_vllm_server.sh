#!/bin/bash
set -euo pipefail
# ============================================================
# 启动 vLLM 服务（LLM Motion Token Plan 预测）
#
# 用法:
#   bash scripts/start_vllm_server.sh [model_path] [port] [gpu_id]
# ============================================================

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH=${1:-"$PROJECT_DIR/checkpoints/llm"}
PORT=${2:-8095}
GPU_ID=${3:-0}

echo "============================================"
echo "  启动 vLLM 服务"
echo "  Model:  ${MODEL_PATH}"
echo "  Port:   ${PORT}"
echo "  GPU:    ${GPU_ID}"
echo "  MaxLen: ${MAX_MODEL_LEN:-1600}"
echo "============================================"

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

MODEL_PATH=$(cd "${PROJECT_DIR}" && realpath "${MODEL_PATH}")

cd "$PROJECT_DIR/motion_generation"

python -u vllm_server.py \
    --port ${PORT} \
    --model_path "${MODEL_PATH}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION:-0.2}" \
    --max_model_len "${MAX_MODEL_LEN:-1600}" \
    --max_num_batched_tokens "${MAX_NUM_BATCHED_TOKENS:-${MAX_MODEL_LEN:-1600}}" \
    --max_num_seqs "${MAX_NUM_SEQS:-1}" \
    --max_tokens_limit "${MAX_TOKENS_LIMIT:-2048}" \
    --default_max_tokens "${DEFAULT_MAX_TOKENS:-512}"
