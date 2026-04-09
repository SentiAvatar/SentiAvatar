#!/bin/bash
# ============================================================
# 单条样本推理脚本 (Single Case Inference)
#
# 用户传入音频文件和动作标签，生成动作(BVH+anim.json)
#
# 前置: 确保 vLLM 服务已启动:
#   bash scripts/start_vllm_server.sh
#
# 用法:
#   # Demo 模式（使用自带示例音频，快速验证环境）
#   bash scripts/run_single_infer.sh
#
#   # 自定义推理
#   bash scripts/run_single_infer.sh \
#       --audio_path /path/to/audio.wav \
#       --action_text "动作：点头" \
#       --output_dir ./output_single
# ============================================================

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd "${PROJECT_DIR}/motion_generation"

# 如果没有传参数，使用 demo 模式
if [ $# -eq 0 ]; then
    echo "============================================"
    echo "  🚀 Demo 模式：使用自带示例音频"
    echo "============================================"
    python single_case_infer.py \
        --audio_path "${PROJECT_DIR}/examples/demo.wav" \
        --action_text "动作：张开双臂上下挥动，像鸟儿一样飞" \
        --output_dir "${PROJECT_DIR}/output_demo"
else
    python single_case_infer.py "$@"
fi
