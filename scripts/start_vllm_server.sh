#!/bin/bash
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
echo "============================================"

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export CUDA_DEVICE_ORDER=PCI_BUS_ID

cd "$PROJECT_DIR/motion_generation"

python -u vllm_server.py \
    --port ${PORT} \
    --model_path ${MODEL_PATH}
