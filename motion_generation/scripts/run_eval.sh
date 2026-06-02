#!/bin/bash
set -euo pipefail
# ============================================================
# 评测脚本
#
# 评估生成动作的质量 (R@K, FID, Diversity, BAS, VOC, ESD)
#
# 用法:
#   bash scripts/run_eval.sh [motion_dir] [gpu_id]
# ============================================================

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
MOTION_DIR=${1:-"${PROJECT_DIR}/output/reconstructed"}
GPU_ID=${2:-0}
PYTHON_BIN="${PYTHON_BIN:-python}"

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output/eval_results}"
MOTION2TEXT_PATH="${MOTION2TEXT_PATH:-${PROJECT_DIR}/data/text_data/motion2text.json}"
WAV_DIR="${WAV_DIR:-${PROJECT_DIR}/data/wav_data}"
EVAL_MODEL_PATH="${EVAL_MODEL_PATH:-${PROJECT_DIR}/checkpoints/eval_model/best_model.pt}"
STATS_DIR="${STATS_DIR:-${PROJECT_DIR}/evaluation/stats/humanml3d/guoh3dfeats}"
GT_MOTION_DIR="${GT_MOTION_DIR:-${MOTION_DIR}}"
REAL_GT_MOTION_DIR="${REAL_GT_MOTION_DIR:-${PROJECT_DIR}/data/motion_data}"
TEXT_MODEL_NAME="${TEXT_MODEL_NAME:-}"
mkdir -p "${OUTPUT_DIR}"

[[ -d "${MOTION_DIR}" ]] || { echo "Missing motion dir: ${MOTION_DIR}" >&2; exit 1; }
[[ -f "${MOTION2TEXT_PATH}" ]] || { echo "Missing motion2text json: ${MOTION2TEXT_PATH}" >&2; exit 1; }
[[ -d "${WAV_DIR}" ]] || { echo "Missing wav dir: ${WAV_DIR}" >&2; exit 1; }
[[ -f "${EVAL_MODEL_PATH}" ]] || { echo "Missing eval model checkpoint: ${EVAL_MODEL_PATH}" >&2; exit 1; }
[[ -d "${STATS_DIR}" ]] || { echo "Missing stats dir: ${STATS_DIR}" >&2; exit 1; }

if ! "${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys

missing = [
    module
    for module in ("hydra", "omegaconf", "librosa", "torch", "transformers", "clip")
    if importlib.util.find_spec(module) is None
]
if missing:
    print("Missing Python modules for full evaluation: " + ", ".join(missing), file=sys.stderr)
    sys.exit(1)
PY
then
    echo "Install the pinned environment first, for example: pip install -r requirements.txt" >&2
    exit 1
fi

EVAL_ARGS=(
    "+eval.motion_dir=${MOTION_DIR}"
    "+eval.gt_motion_dir=${GT_MOTION_DIR}"
    "+eval.real_gt_motion_dir=${REAL_GT_MOTION_DIR}"
    "+eval.output_dir=${OUTPUT_DIR}"
    "+eval.motion2text_path=${MOTION2TEXT_PATH}"
    "+eval.wav_dir=${WAV_DIR}"
    "+eval.model_path=${EVAL_MODEL_PATH}"
    "+eval.stats_dir=${STATS_DIR}"
)
if [[ -n "${TEXT_MODEL_NAME}" ]]; then
    EVAL_ARGS+=("+eval.text_model_name=${TEXT_MODEL_NAME}")
fi

echo "============================================"
echo "  评测生成动作"
echo "  Motion Dir:  ${MOTION_DIR}"
echo "  GT Dir:      ${GT_MOTION_DIR}"
echo "  Output Dir:  ${OUTPUT_DIR}"
echo "============================================"

cd "${PROJECT_DIR}/evaluation"

CUDA_VISIBLE_DEVICES=${GPU_ID} "${PYTHON_BIN}" evaluate_pred_motion_v2.py "${EVAL_ARGS[@]}"

echo ""
echo "  评测完成！结果保存在: ${OUTPUT_DIR}"
