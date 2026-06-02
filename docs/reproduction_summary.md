# SentiAvatar 复现进展总结

本文档记录当前仓库相对论文 [SentiAvatar](https://arxiv.org/abs/2604.02908) 的复现完成度。当前状态是：训练、推理、重建、审计和评估的工程链路已经基本补齐，并通过合成数据 smoke test 验证；但尚未使用真实 SuSuInterActs / BEATv2 / 多来源 foundation 数据完成论文级指标复现。

## 当前完成度

当前仓库已经从主要推理逻辑扩展为可训练的复现框架，覆盖以下模块：

- Body Motion R-VQVAE 训练与 GitHub 推理兼容加载。
- HuBERT audio feature、K-means audio token、RVQVAE motion token 预处理。
- Motion Foundation text-to-motion 预训练 JSONL 构建。
- 多来源 foundation manifest，用于混合 EmbodyAI、SnapMoGen、Motion-X、Hunyuan Distill 等外部语料。
- LLM planner SFT，包括普通 planner 样本和 continuation-aware 样本混合训练。
- Body Infill Transformer 训练与推理。
- Face R-VQVAE、face token preprocessing、Face Infill Transformer。
- vLLM planner server、batch/single inference、continuation 推理。
- token reconstruction 到 BVH / JSON / NPY。
- 数据、checkpoint、tokenizer 和评估前置条件审计。
- SuSu-style 与 BEATv2-style 评估入口。

## 训练链路

### Body Motion R-VQVAE

训练入口：

```bash
CODEBOOK_SIZE=512 NUM_QUANTIZERS=4 bash scripts/train_rvqvae.sh
```

训练产物会保存为 GitHub 推理可加载的结构：

```text
checkpoints_train/susuinteracts/rvqvae_body/
  opt.txt
  meta/mean.npy
  meta/std.npy
  model/latest.pth
```

`opt.txt`、`mean.npy`、`std.npy` 和 `latest.pth` 与现有 `load_config_from_checkpoint()` / `load_model()` 路径对齐。

### 预处理

入口：

```bash
bash scripts/preprocess_data.sh --all \
  --rvqvae_ckpt checkpoints_train/susuinteracts/rvqvae_body/model/latest.pth \
  --hubert_path checkpoints/chinese-hubert-base \
  --kmeans_path checkpoints/hubert_kmeans/model.mdl \
  --overwrite \
  --strict
```

预处理会生成：

- `data/audio_features_hubert_layer9_fps10`
- `data/audio_tokens_hubert_layer9_fps10`
- `data/motion_token_data`

当前逻辑会将 HuBERT 特征对齐到 RVQVAE motion-token 长度，并在默认 strict 模式下拒绝 audio/motion token 长度不一致的数据。

### Motion Foundation 与 Planner

Motion Foundation 入口：

```bash
bash scripts/build_motion_foundation_dataset.sh
bash scripts/train_llm_planner.sh \
  --init_model Qwen/Qwen2-0.5B \
  --train_jsonl data/llm_sft/foundation_step1.jsonl \
  --output_dir checkpoints_train/motion_foundation
```

多来源 foundation 语料可以通过 manifest 构建：

```bash
FOUNDATION_MANIFEST_JSON=/data/foundation/foundation_manifest.json \
  bash scripts/build_motion_foundation_dataset.sh
```

Planner SFT 默认支持普通样本和 continuation 样本混合训练：

```bash
python scripts/merge_llm_jsonl.py \
  --inputs regular=data/llm_sft/train_step4.jsonl continuation=data/llm_sft/train_step4_continuation.jsonl \
  --output_jsonl data/llm_sft/train_step4_mixed.jsonl \
  --shuffle

bash scripts/train_llm_planner.sh \
  --train_jsonl data/llm_sft/train_step4_mixed.jsonl \
  --output_dir checkpoints_train/llm
```

`scripts/reproduce_pipeline.py` 默认使用 `--planner_sft_mix both`，即普通 planner SFT 与 continuation SFT 混合训练。

### Body Infill Transformer

入口：

```bash
CODEBOOK_SIZE=512 NUM_TOKENS_PER_FRAME=4 bash scripts/train_infill_transformer.sh
```

当前训练逻辑已对齐论文中的关键设定：

- 默认 5-frame window，对应 `step=4`。
- 首尾关键帧保持已知。
- 只监督内部帧 token。
- 内部 token 随机 mask。
- 未 mask 的内部 token 以 10% 概率替换为随机 same-codebook token。
- 支持 step ablation：训练时使用 `--num_frames t+1`，推理时使用 `--step t`。

### Face 分支

入口：

```bash
CODEBOOK_SIZE=512 NUM_QUANTIZERS=2 bash scripts/train_face_rvqvae.sh
bash scripts/preprocess_face_tokens.sh
CODEBOOK_SIZE=512 NUM_TOKENS_PER_FRAME=2 bash scripts/train_face_infill_transformer.sh
```

当前已经补齐 paper-style face path：

- Face R-VQVAE。
- ARKit face token preprocessing。
- Face Infill Transformer。
- `reconstruct_from_tokens.py --face_mode infill` 注入 face animation。

## 推理与重建

vLLM server：

```bash
MAX_MODEL_LEN=2400 MAX_TOKENS_LIMIT=2400 \
  bash scripts/start_vllm_server.sh checkpoints_train/llm 8095 0
```

batch inference 与重建：

```bash
python scripts/reproduce_pipeline.py \
  --run \
  --skip_existing \
  --stages infer-batch,reconstruct-batch \
  --vllm_url http://localhost:8095
```

推理链路已增强：

- 检查 LLM 输出 `[step_t]`、`[len_N]` 和 residual token 数量。
- 检查 token range 和 residual group width。
- 检查 audio token / feature 长度与维度。
- 支持 continuation prefix。
- 保存 single-case artifacts，便于多轮续接调试。

## 评估链路

当前包含三类评估入口。

### SuSu-style 完整评估

```bash
bash scripts/run_eval.sh output/reconstructed 0
```

该路径覆盖：

- R@K
- FID
- Diversity
- BAS
- VOC
- ESD

需要 ChronTMR/eval checkpoint、真实 `motion2text.json`、wav 文件和对应 Python 依赖。

### 轻量同步指标

```bash
python scripts/eval_sync_metrics.py \
  --motion_dir output/reconstructed \
  --wav_dir data/wav_data \
  --motion_type pred \
  --output_json output/eval_results/sync_metrics.json
```

覆盖：

- ESD
- BAS
- BHR
- VOC

该路径不依赖 ChronTMR checkpoint，适合 smoke 和快速回归测试。

### BEATv2-style 指标

```bash
python scripts/eval_beatv2_metrics.py \
  --pred_dir output/reconstructed \
  --gt_dir output/reconstructed \
  --wav_dir data/wav_data \
  --output_json output/eval_results/beatv2_metrics.json
```

覆盖：

- FGD
- BC / BC_x10
- Diversity

默认 FGD/Diversity 使用 deterministic handcrafted motion features，只适合回归测试。若要对齐官方 BEATv2 指标，需要通过 `--feature_model_path <torchscript.pt>` 接入官方或同协议 gesture feature extractor。

## 一键分阶段复现

主入口：

```bash
python scripts/reproduce_pipeline.py --list_stages
python scripts/reproduce_pipeline.py --stages all
python scripts/reproduce_pipeline.py --run --skip_existing
```

该 runner 默认 dry-run，不删除文件。主要 stages 包括：

- `audit-raw`
- `train-rvqvae`
- `preprocess`
- `build-foundation-jsonl`
- `train-motion-foundation`
- `build-planner-jsonl`
- `build-continuation-jsonl`
- `merge-planner-jsonl`
- `train-llm-planner`
- `train-body-infill`
- `train-face-rvqvae`
- `preprocess-face-tokens`
- `train-face-infill`
- `audit-final`
- `infer-batch`
- `reconstruct-batch`
- `eval-sync`
- `eval-beatv2`
- `eval-full`

## 已验证内容

当前已经通过以下验证：

```bash
python -m compileall motion_generation evaluation tools scripts tests
bash -n scripts/*.sh motion_generation/scripts/*.sh
python -m pytest tests/test_merge_llm_jsonl.py tests/test_motion_foundation_manifest.py tests/test_eval_beatv2_metrics.py tests/test_infill_masking.py -q
```

合成数据 smoke test 已多次跑通。最新完整 smoke 包含：

- Body R-VQVAE one-step train/load/decode。
- Motion Foundation JSONL 构建。
- Planner SFT JSONL 构建。
- Continuation SFT JSONL 构建。
- Regular + continuation mixed JSONL 合并。
- Body Infill one-step train/load/infer。
- Face R-VQVAE one-step train/load。
- Face token preprocessing。
- Face Infill one-step train/load。
- Reconstruction 到 BVH / JSON / NPY。
- Sync eval。
- Final audit。

最新 smoke final audit：

```text
ok=69 warn=0 error=0
```

## 当前未完成部分

还不能声明已经复现论文结果，原因是以下实证条件仍缺失：

- 未接入真实 SuSuInterActs 数据。
- 未接入真实 BEATv2 数据。
- 未接入真实 EmbodyAI / SnapMoGen / Motion-X / Hunyuan Distill 多来源 foundation 数据。
- 未完成真实规模 GPU 长训练。
- 未得到真实可用的 Motion Foundation / Planner / Infill checkpoint。
- 未启动真实 vLLM planner checkpoint 做端到端推理评估。
- 未使用官方或同协议 BEATv2 feature extractor 计算最终 FGD。
- 未复现论文表格中的最终 R@K、FID、Diversity、ESD、FGD、BC 等指标。

## 结论

当前仓库已经具备较完整的工程复现路径：训练代码、推理代码、重建代码、审计代码和评估入口都已补齐，并且通过合成数据验证可执行。下一步需要接入真实数据和训练资源，执行长训练并跑完整指标，才能从“工程路径已复现”推进到“论文结果已复现”。
