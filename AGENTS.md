# Repository Guidelines

## Project Structure & Module Organization

`motion_generation/` is the main Python package for inference, vLLM serving, RVQVAE and infill models, preprocessing, metadata, and training helpers. `evaluation/` contains metric models, datasets, stats, and Hydra config for motion evaluation. Use top-level `scripts/` for common entry points; mirrored scripts under `motion_generation/scripts/` support running from inside the package. `tools/` contains utilities such as motion visualization. `examples/` stores demo inputs, `assets/` stores README figures, `checkpoints/` documents expected model-weight layout, and runtime datasets are expected under `data/`.

## Build, Test, and Development Commands

```bash
conda create -n sentiavatar python=3.10 -y
conda activate sentiavatar
pip install -r requirements.txt
python scripts/preprocess_data.py --all --device cuda:0
bash scripts/start_vllm_server.sh checkpoints/llm 8095 0
bash scripts/run_single_infer.sh
bash scripts/run_test.sh 8095 0
bash scripts/run_eval.sh output/reconstructed 0
python -m compileall motion_generation evaluation tools
```

Use Python 3.10 and the pinned requirements. Start the vLLM planner before inference. `run_single_infer.sh` is the fastest demo using `examples/demo.wav`; `run_test.sh` performs batch generation; `run_eval.sh` evaluates reconstructed motion. Training entry points include `scripts/train_rvqvae.sh`, `scripts/train_llm_planner.sh`, `scripts/train_infill_transformer.sh`, and face-specific training scripts.

## Coding Style & Naming Conventions

Write Python with 4-space indentation, `snake_case` for functions and variables, and `PascalCase` for classes. Keep CLIs in `argparse`, prefer repo-relative defaults, and avoid hard-coded local paths. In shell scripts, quote path variables, keep environment variables uppercase, and expose GPU IDs, ports, and checkpoint paths as arguments or env overrides.

## Testing Guidelines

There is no formal unit-test suite yet. For utility changes, add focused `pytest` tests under `tests/` using `test_*.py` names. For pipeline changes, run `python -m compileall motion_generation evaluation tools`; when checkpoints, data, and CUDA are available, also run `bash scripts/run_single_infer.sh`, `bash scripts/run_test.sh 8095 0`, and relevant evaluation or training smoke commands.

## Commit & Pull Request Guidelines

Recent commit subjects are short lowercase messages such as `update ack` and `update requirements`; keep subjects concise but make them specific, for example `fix single inference output path`. Pull requests should state purpose, commands run, checkpoint/data assumptions, linked issues, and include sample outputs or screenshots when generated BVH, JSON, or visual assets change.

## Security & Configuration Tips

Do not commit generated outputs, logs, dataset dumps, secrets, or large new checkpoints unless they are intentional release artifacts. Document any new CUDA, port, Hugging Face offline, or checkpoint-path assumptions in the relevant README or script comments.
