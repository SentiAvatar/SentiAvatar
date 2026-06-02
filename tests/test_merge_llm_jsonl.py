import json
import subprocess
import sys
from pathlib import Path


def _write_jsonl(path: Path, name: str):
    path.write_text(
        json.dumps(
            {
                "name": name,
                "prompt": f"Human: {name}<|im_end|>\nAssistant:",
                "completion": "[step_4][len_1][res1_0][res2_1][res3_2][res4_3]<|im_end|>",
                "step": 4,
                "num_keyframes": 1,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def test_merge_llm_jsonl_preserves_provenance(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    regular = tmp_path / "regular.jsonl"
    continuation = tmp_path / "continuation.jsonl"
    output = tmp_path / "mixed.jsonl"
    summary = tmp_path / "summary.json"
    _write_jsonl(regular, "regular_sample")
    _write_jsonl(continuation, "continuation_sample")

    subprocess.run(
        [
            sys.executable,
            "scripts/merge_llm_jsonl.py",
            "--inputs",
            f"regular={regular}",
            f"continuation={continuation}",
            "--output_jsonl",
            str(output),
            "--summary_json",
            str(summary),
        ],
        cwd=repo_root,
        check=True,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["sft_source"] for row in rows] == ["regular", "continuation"]
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["total_rows"] == 2
    assert [item["label"] for item in payload["inputs"]] == ["regular", "continuation"]
