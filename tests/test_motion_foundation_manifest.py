import json
import subprocess
import sys
from pathlib import Path


def _write_source(root: Path, source: str, text: str):
    token_dir = root / source / "motion_token_data"
    text_dir = root / source / "text_data"
    split_dir = root / source / "split"
    token_dir.mkdir(parents=True)
    text_dir.mkdir(parents=True)
    split_dir.mkdir(parents=True)
    (token_dir / "sample.json").write_text(
        json.dumps({"tokens": [[0, 1, 2, 3], [4, 5, 6, 7]]}),
        encoding="utf-8",
    )
    (text_dir / "motion2text.json").write_text(json.dumps({"sample": text}, ensure_ascii=False), encoding="utf-8")
    (split_dir / "train_file_list.txt").write_text("sample\n", encoding="utf-8")


def test_motion_foundation_manifest_builds_mixed_jsonl(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    _write_source(tmp_path, "embodyai", "walk forward")
    _write_source(tmp_path, "motionx", "wave hand")
    manifest = {
        "sources": [
            {
                "name": "embodyai",
                "motion_token_dir": "embodyai/motion_token_data",
                "motion2text_json": "embodyai/text_data/motion2text.json",
                "split_file": "embodyai/split/train_file_list.txt",
            },
            {
                "name": "motionx",
                "motion_token_dir": "motionx/motion_token_data",
                "motion2text_json": "motionx/text_data/motion2text.json",
                "split_file": "motionx/split/train_file_list.txt",
                "max_examples": 1,
            },
        ]
    }
    manifest_path = tmp_path / "foundation_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "foundation.jsonl"

    subprocess.run(
        [
            sys.executable,
            "motion_generation/training/build_llm_sft_dataset.py",
            "--source_manifest_json",
            str(manifest_path),
            "--output_jsonl",
            str(output),
            "--no_audio",
            "--step",
            "1",
            "--codebook_size",
            "8",
            "--num_quantizers",
            "4",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["source"] for row in rows] == ["embodyai", "motionx"]
    assert [row["source_sample"] for row in rows] == ["sample", "sample"]
    assert rows[0]["name"] == "embodyai/sample"
    assert "walk forward" in rows[0]["prompt"]
    assert rows[0]["completion"].startswith("[step_1][len_2]")
