#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

python "${PROJECT_DIR}/motion_generation/scripts/preprocess_data.py" "$@"
