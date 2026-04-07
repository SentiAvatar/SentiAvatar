#!/bin/bash
# ============================================================
# 单条样本推理脚本 (Single Case Inference)
#
# 用户传入音频文件和动作标签，生成动作(BVH+anim.json)
#
# 前置: 确保 vLLM 服务已启动 (bash scripts/start_vllm_server.sh)
#
# 用法:
#   bash scripts/run_single_infer.sh \
#       --audio_path <wav_file> \
#       --action_text "动作：点头" \
#       --output_dir ./output_single
# ============================================================

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "${PROJECT_DIR}/motion_generation"

python single_case_infer.py "$@"
