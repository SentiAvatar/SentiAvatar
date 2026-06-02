import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def _motion(num_frames: int, phase: float) -> dict:
    t = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    body = np.stack([np.sin(t * 6.0 + phase), np.cos(t * 3.0), t], axis=1).astype(np.float32)
    left = np.stack([t, t * 0.5], axis=1).astype(np.float32)
    right = np.stack([t * 0.25, np.sin(t * 4.0)], axis=1).astype(np.float32)
    return {"name": "", "body": body, "left": left, "right": right}


def _write_wav(path):
    sr = 16000
    y = np.zeros(sr, dtype=np.float32)
    y[1000:1030] = 0.8
    y[5000:5030] = 0.8
    wavfile.write(path, sr, (y * 32767).astype(np.int16))


def test_eval_beatv2_metrics_cli(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    pred_dir = tmp_path / "reconstructed"
    wav_dir = tmp_path / "wav_data" / "demo"
    pred_dir.mkdir()
    wav_dir.mkdir(parents=True)
    for idx, frames in enumerate([20, 24]):
        name = f"demo/sample_{idx}"
        pred = _motion(frames, phase=0.1 * idx)
        gt = _motion(frames, phase=0.2 * idx)
        pred["name"] = name
        gt["name"] = name
        np.save(pred_dir / f"demo_sample_{idx}_pred.npy", pred)
        np.save(pred_dir / f"demo_sample_{idx}_gt.npy", gt)
        _write_wav(wav_dir / f"sample_{idx}.wav")

    output_json = tmp_path / "beatv2_metrics.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/eval_beatv2_metrics.py",
            "--pred_dir",
            str(pred_dir),
            "--gt_dir",
            str(pred_dir),
            "--wav_dir",
            str(tmp_path / "wav_data"),
            "--max_feature_dims",
            "8",
            "--output_json",
            str(output_json),
        ],
        check=True,
        cwd=str(repo_root),
        text=True,
        capture_output=True,
    )
    assert "FGD" in proc.stdout
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["metrics"]["num_pairs"] == 2
    assert payload["metrics"]["feature_type"] == "handcrafted"
    assert payload["metrics"]["paper_equivalent"] is False
    assert "BC_x10" in payload["metrics"]
