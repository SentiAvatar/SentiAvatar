#!/bin/bash
# ============================================================
# 评测脚本
#
# 评估生成动作的质量 (R@K, FID, Diversity, BAS, VOC, ESD)
#
# 用法:
#   bash scripts/run_eval.sh [motion_dir] [gpu_id]
# ============================================================

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MOTION_DIR=${1:-"${PROJECT_DIR}/output/reconstructed"}
GPU_ID=${2:-0}

OUTPUT_DIR="${PROJECT_DIR}/output/eval_results"
mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "  评测生成动作"
echo "  Motion Dir:  ${MOTION_DIR}"
echo "  Output Dir:  ${OUTPUT_DIR}"
echo "============================================"

cd "${PROJECT_DIR}/evaluation"

CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluate_pred_motion_v2.py \
    +eval.motion_dir=${MOTION_DIR} \
    +eval.output_dir=${OUTPUT_DIR} \
    +eval.motion2text_path="${PROJECT_DIR}/data/text_data/motion2text.json" \
    +eval.wav_dir="${PROJECT_DIR}/data/wav_data" \
    +eval.model_path="${PROJECT_DIR}/checkpoints/eval_model/best_model.pt" \
    +eval.stats_dir="${PROJECT_DIR}/evaluation/stats/humanml3d/guoh3dfeats"

echo ""
echo "  评测完成！结果保存在: ${OUTPUT_DIR}"
