#!/usr/bin/env python3
"""Repository-root wrapper for motion_generation/scripts/preprocess_data.py."""

from pathlib import Path
import runpy

SCRIPT = Path(__file__).resolve().parents[1] / "motion_generation" / "scripts" / "preprocess_data.py"
runpy.run_path(str(SCRIPT), run_name="__main__")
