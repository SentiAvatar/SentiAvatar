#!/bin/bash
# ============================================================
# 测试集推理脚本 (Batch Mode)
#
# 前提:
#   1. 已运行数据预处理 (python scripts/preprocess_data.py --all)
#   2. 确保 vLLM 服务已启动 (bash scripts/start_vllm_server.sh)
#
# 流程:
#   1. Pipeline 推理: LLM + Mask Transformer → dense motion tokens
#   2. Token 重建: RVQVAE 解码 → BVH/JSON
#
# 用法:
#   bash scripts/run_test.sh [vllm_port] [gpu_id]
# ============================================================

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VLLM_PORT=${1:-8095}
GPU_ID=${2:-0}

# Mask Transformer checkpoint
MASK_CKPT="${PROJECT_DIR}/checkpoints/mask_transformer"

# RVQVAE checkpoint
RVQVAE_CKPT="${PROJECT_DIR}/checkpoints/rvqvae/model/epoch_30.pth"

# 数据路径
DATA_DIR="${PROJECT_DIR}/data"
OUTPUT_DIR="${PROJECT_DIR}/output"

echo "============================================"
echo "  测试集推理 (Batch Mode)"
echo "  vLLM Port:      ${VLLM_PORT}"
echo "  Mask Ckpt:      ${MASK_CKPT}"
echo "  RVQVAE Ckpt:    ${RVQVAE_CKPT}"
echo "  Output:         ${OUTPUT_DIR}"
echo "============================================"

mkdir -p "${OUTPUT_DIR}"

# Step 1: Pipeline 推理 (LLM + Mask Transformer)
echo ""
echo ">>> Step 1: Pipeline 推理..."
cd "${PROJECT_DIR}/motion_generation"

CUDA_VISIBLE_DEVICES=${GPU_ID} python pipeline_infer.py \
    --mask_ckpt ${MASK_CKPT} \
    --vllm_port ${VLLM_PORT} \
    --mode batch \
    --generate_steps 6 \
    --temperature 0.5 \
    --top_p 0.4 \
    --motion_token_dir "${DATA_DIR}/motion_token_data" \
    --audio_token_dir "${DATA_DIR}/audio_tokens_hubert_layer9_fps10" \
    --audio_feat_dir "${DATA_DIR}/audio_features_hubert_layer9_fps10" \
    --val_split_file "${DATA_DIR}/split/test_file_list.txt" \
    --motion2text_json "${DATA_DIR}/text_data/motion2text.json" \
    --output_path "${OUTPUT_DIR}/pipeline_batch_results.json"

# Step 2: Token 重建 (RVQVAE → BVH/JSON)
echo ""
echo ">>> Step 2: Token 重建..."

CUDA_VISIBLE_DEVICES=${GPU_ID} python reconstruct_from_tokens.py \
    --input_json "${OUTPUT_DIR}/pipeline_batch_results.json" \
    --checkpoint_path ${RVQVAE_CKPT} \
    --output_dir "${OUTPUT_DIR}/reconstructed" \
    --wave_folder "${DATA_DIR}/wav_data" \
    --tgt_fps 30.0

echo ""
echo "============================================"
echo "  测试集推理完成！"
echo "  结果保存在: ${OUTPUT_DIR}/reconstructed"
echo "============================================"
