#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
完整推理流水线：LLM Motion Token Plan + Mask Transformer 插帧 (Hubert Features Version)

流程：
1. 使用 vLLM 服务预测稀疏 motion token plan（间隔4采样关键帧）
2. 使用 Mask Transformer 进行滑动窗口插帧，生成完整的 dense motion tokens

输入：
- 动作标签（文本）
- 音频 token（hubert 量化后的，用于 LLM prompt）
- 音频特征（hubert layer9 npy，用于 Mask Transformer）

输出：
- 完整的 dense motion tokens（每帧4个残差 token）

支持模式：
- demo: 单样本推理（从数据集中选取）
- batch: 批量推理（验证集）

前置条件：
- vLLM 服务已启动（参考 gen_vllm_infer/scripts/at2m_v4_vllm_sft.sh）

数据来源：
- motion token: merged_processed_all_data/motion_token_data/{name}.json
- audio token (for LLM): merged_processed_all_data/audio_tokens_hubert_layer9_fps10/{name}.json
- audio feature (for Mask Transformer): merged_processed_all_data/audio_features_hubert_layer9_fps10/{name}.npy

用法:
    python pipeline_infer.py --mask_ckpt <checkpoint_path> [--vllm_port 8095] [--mode demo]

@File    :   pipeline_infer.py
@Time    :   2025/07/16
"""

import os
import sys
import json
import time
import argparse
import re
import numpy as np
import requests
import torch
from tqdm import tqdm
from safetensors import safe_open
import random

# 确保能导入本地模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.audio_motion_model import AudioMotionTransformer, AudioMotionConfig


# ======================================================================
#  Part 1: vLLM 客户端（LLM Motion Token Plan 预测）
# ======================================================================

def extract_mids_from_string(input_str):
    """从 LLM 输出字符串中提取所有 [type_value] 格式的 token"""
    pattern = r'\[(.*?)\]'
    matches = re.findall(pattern, input_str)
    id_dict = {}
    for match in matches:
        if '_' in match:
            id_type, value = match.split('_', 1)
            try:
                value = int(value)
            except ValueError:
                pass
        else:
            id_type = match
            value = None
        if id_type not in id_dict:
            id_dict[id_type] = []
        if value is None:
            continue
        id_dict[id_type].append(value)
    return id_dict


class VLLMClient:
    """vLLM 服务客户端，用于调用 LLM 预测稀疏 motion token plan"""

    def __init__(self, base_url="http://localhost:8095"):
        self.base_url = base_url

    def health_check(self):
        """检查 vLLM 服务是否可用"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def predict_motion_plan(self, text_list, temperature=0.5, top_p=0.4,
                            max_tokens=1024, stop=None, base_token_start=0):
        """
        调用 vLLM 服务预测 motion token plan

        Args:
            text_list: 已格式化的 prompt 列表（含 Human:/Assistant: 模板）
            temperature: 采样温度
            top_p: 核采样参数
            max_tokens: 最大生成 token 数
            stop: 停止 token 列表
            base_token_start: token 偏移量

        Returns:
            list of dict, 每个 dict 包含:
                - tokens: {"res_1": [...], "res_2": [...], "res_3": [...], "res_4": [...]}
                - total_len: LLM 预测的总帧数（如有）
                - raw_output: LLM 原始输出字符串
        """
        if stop is None:
            stop = ["<|im_end|>"]

        payload = {
            "text_list": text_list,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stop": stop,
        }
        response = requests.post(
            f"{self.base_url}/text_to_motion",
            json=payload, timeout=120
        )
        assert response.status_code == 200, (
            f"vLLM 请求失败，状态码: {response.status_code}, "
            f"响应: {response.text}"
        )

        result = response.json()
        results_list = []

        for i, (text, motion) in enumerate(result["motion_sequence_list"]):
            # 解析 LLM 输出
            motion_clean = motion.replace("[res_", "[res")
            # 尝试按 [step_4] 或 [STEP_4] 分割，取最后一段
            motion_sequence = motion_clean
            for sep in ["[step_4]", "[STEP_4]", "[step_1]", "[STEP_1]"]:
                if sep.lower() in motion_sequence.lower():
                    idx = motion_sequence.lower().rfind(sep.lower())
                    motion_sequence = motion_sequence[idx + len(sep):]
                    break

            parsed = extract_mids_from_string(motion_sequence)
            full_parsed = extract_mids_from_string(motion_clean)

            motion_tokens = {
                "res_1": parsed.get("res1", []),
                "res_2": parsed.get("res2", []),
                "res_3": parsed.get("res3", []),
                "res_4": parsed.get("res4", []),
            }
            # 截断到最短长度，确保四层对齐
            lengths = [len(v) for v in motion_tokens.values()]
            min_len = min(lengths) if lengths else 0
            motion_tokens = {
                k: [tk - base_token_start for tk in v[:min_len]]
                for k, v in motion_tokens.items()
            }

            total_len = full_parsed.get("len", [None])[0] if "len" in full_parsed else None

            results_list.append({
                "tokens": motion_tokens,
                "total_len": total_len,
                "raw_output": motion,
            })

        return results_list


# ======================================================================
#  Part 2: Mask Transformer 模型加载与插帧
# ======================================================================

def load_mask_transformer(ckpt_path, device="cuda"):
    """加载训练好的 Mask Transformer 模型"""
    print(f"[Mask Transformer] 加载模型: {ckpt_path}")

    config = AudioMotionConfig.from_pretrained(ckpt_path)
    model = AudioMotionTransformer(config)

    safetensors_path = os.path.join(ckpt_path, "model.safetensors")
    bin_path = os.path.join(ckpt_path, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        state_dict = {}
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        model.load_state_dict(state_dict, strict=True)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    else:
        raise FileNotFoundError(
            f"在 {ckpt_path} 中未找到模型权重文件 "
            "(model.safetensors 或 pytorch_model.bin)"
        )

    model.to(device)
    model.eval()
    print(f"[Mask Transformer] 模型加载完成，设备: {device}")
    return model


def ensure_length(dense_tokens, target_length):
    """
    确保 dense motion tokens 序列长度与目标长度（音频帧数）一致。

    - 如果过短：重复最后一帧进行填充
    - 如果过长：截断到目标长度

    Args:
        dense_tokens: list of [4 ints]，当前的 dense motion tokens
        target_length: int，目标帧数（= 音频特征帧数）

    Returns:
        list of [4 ints]，长度恰好为 target_length
    """
    current_length = len(dense_tokens)
    if current_length == target_length:
        return dense_tokens

    if current_length > target_length:
        return dense_tokens[:target_length]

    if current_length == 0:
        return [[0, 0, 0, 0]] * target_length

    last_frame = dense_tokens[-1]
    pad_count = target_length - current_length
    return dense_tokens + [list(last_frame) for _ in range(pad_count)]


def sparse_to_keyframes(motion_tokens_dict):
    """
    将 LLM 输出的稀疏 motion tokens 转换为关键帧列表

    Input:  {"res_1": [v0, v1, ...], "res_2": [...], "res_3": [...], "res_4": [...]}
    Output: [[res1_0, res2_0, res3_0, res4_0], [res1_1, res2_1, res3_1, res4_1], ...]
    """
    num_keyframes = len(motion_tokens_dict["res_1"])
    keyframes = []
    for i in range(num_keyframes):
        keyframes.append([
            motion_tokens_dict["res_1"][i],
            motion_tokens_dict["res_2"][i],
            motion_tokens_dict["res_3"][i],
            motion_tokens_dict["res_4"][i],
        ])
    return keyframes


def interpolate_sequence(model, keyframe_tokens, audio_features, generate_steps=6):
    """
    使用 Mask Transformer 对稀疏关键帧进行滑动窗口插帧。

    每个窗口包含 5 帧：frame1(已知) + frame2,3,4(mask) + frame5(已知)，
    模型预测中间 3 帧的 motion token。

    Args:
        model: AudioMotionTransformer 模型
        keyframe_tokens: list of [4 ints]，关键帧的 motion tokens
        audio_features: (num_frames_full, 768) numpy array，完整序列的 hubert 音频特征
        generate_steps: 每个窗口的生成步数（越多越精确，但越慢）

    Returns:
        list of [4 ints]，插帧后的完整 dense motion tokens
    """
    ntpf = model.config.num_tokens_per_frame
    codebook_size = model.config.codebook_size
    offsets = [codebook_size * i for i in range(ntpf)]
    mask_token_id = model.config.vocab_size - 1
    device = next(model.parameters()).device

    num_keyframes = len(keyframe_tokens)
    if num_keyframes < 2:
        return keyframe_tokens

    all_output_frames = []

    for i in range(num_keyframes - 1):
        frame1_tokens = keyframe_tokens[i]
        frame5_tokens = keyframe_tokens[i + 1]

        # 构建 5 帧窗口输入: frame1(4 tokens) + mask(12 tokens) + frame5(4 tokens)
        input_tokens = []
        for j in range(ntpf):
            input_tokens.append(frame1_tokens[j] + offsets[j])
        for _ in range(3 * ntpf):
            input_tokens.append(mask_token_id)
        for j in range(ntpf):
            input_tokens.append(frame5_tokens[j] + offsets[j])

        # 获取对应 5 帧的 audio 特征
        start_audio_idx = i * 4
        end_audio_idx = start_audio_idx + 5
        if end_audio_idx <= audio_features.shape[0]:
            window_audio = audio_features[start_audio_idx:end_audio_idx]
        else:
            # 边界处理：start_audio_idx 可能超出音频长度
            if start_audio_idx >= audio_features.shape[0]:
                # 完全超出范围，用最后一帧填充
                last_frame = audio_features[-1:]  # (1, feat_dim)
                window_audio = np.tile(last_frame, (5, 1))
            else:
                available = audio_features[start_audio_idx:]
                pad_len = 5 - available.shape[0]
                if pad_len > 0:
                    padding = np.tile(available[-1:], (pad_len, 1))
                    window_audio = np.concatenate([available, padding], axis=0)
                else:
                    window_audio = available[:5]

        input_ids = torch.tensor([input_tokens], device=device)
        audio_feat = torch.tensor(
            window_audio, dtype=torch.float32, device=device
        ).unsqueeze(0)

        with torch.no_grad():
            output = model.generate_sbs(
                input_ids, audio_feat, generate_steps=generate_steps
            )

        output = output[0].cpu().tolist()

        # 解析中间 3 帧 (f=1,2,3)
        interp_frames = []
        for f in range(1, 4):
            frame_tokens = []
            for j in range(ntpf):
                token_id = output[f * ntpf + j]
                frame_tokens.append(token_id - offsets[j])
            interp_frames.append(frame_tokens)

        # 组装输出
        if i == 0:
            all_output_frames.append(list(frame1_tokens))
            all_output_frames.extend(interp_frames)
            all_output_frames.append(list(frame5_tokens))
        else:
            all_output_frames.extend(interp_frames)
            all_output_frames.append(list(frame5_tokens))

    return all_output_frames


# ======================================================================
#  Part 3: 数据加载与 Prompt 构建
# ======================================================================

def load_motion_tokens(name, motion_token_dir):
    """加载 motion token JSON 文件"""
    json_path = os.path.join(motion_token_dir, name + ".json")
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["tokens"]  # list of [4 ints] per frame


def load_audio_tokens(name, audio_token_dir):
    """加载 hubert audio token JSON 文件"""
    json_path = os.path.join(audio_token_dir, name + ".json")
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["tokens"]  # list of int


def load_audio_features(name, audio_feat_dir):
    """加载 hubert 音频特征 npy 文件"""
    feat_path = os.path.join(audio_feat_dir, name + ".npy")
    if os.path.exists(feat_path):
        return np.load(feat_path).astype(np.float32)
    return None


def format_audio_tokens_for_prompt(audio_tokens, indices):
    """
    将指定帧索引处的 audio token 格式化为 LLM prompt 字符串

    Args:
        audio_tokens: 完整的 audio token 列表，每个元素是 int
        indices: 采样帧索引列表

    Returns:
        如 "[audio_183][audio_507][audio_956]..."
    """
    parts = []
    for idx in indices:
        if idx < len(audio_tokens):
            token = audio_tokens[idx]
            if isinstance(token, list) and len(token) > 0:
                parts.append(f"[audio_{token[0]}]")
            elif isinstance(token, int):
                parts.append(f"[audio_{token}]")
    return "".join(parts)


def construct_llm_prompt(action_text, audio_tokens, offset=0, step=4):
    """
    构建 LLM 推理的 prompt

    Args:
        action_text: 动作标签，如 "动作：抬头仰望"
        audio_tokens: 完整的 audio token 列表（每帧一个）
        offset: 采样偏移 (0-3)
        step: 采样间隔 (默认 4)

    Returns:
        (prompt_str, sampled_indices)
    """
    total_frames = len(audio_tokens)
    sampled_indices = list(range(offset, total_frames, step))
    audio_str = format_audio_tokens_for_prompt(audio_tokens, sampled_indices)
    prompt = f"{action_text}{audio_str}"
    return prompt, sampled_indices


def extract_description(text: str):
    """
    从 motion2text.json 的文本中提取描述内容作为 action_text

    优先使用最后一个【】中的内容（通常是动作标签）。
    如果动作为"无动作"，则尝试使用表情标签作为替代。

    例:
      "【表情：认真聆听】【动作：缓慢点头】嗯" -> "动作：缓慢点头"
      "【表情：担忧】【动作：无动作】爬坡前..." -> "动作：担忧"
      "【表情：无表情】【动作：无动作】你脸盲..." -> "动作：无动作"
    """
    pattern = r'【(.+?)】'
    matches = re.findall(pattern, text)
    if not matches:
        return None

    last_tag = matches[-1]

    # 如果动作为"无动作"，尝试用表情标签替代
    if last_tag == "动作：无动作":
        # 查找表情标签
        for tag in matches:
            if tag.startswith("表情：") and tag != "表情：无表情":
                expression = tag.replace("表情：", "")
                if "动作" in expression:
                    return expression
                else:
                    return f"动作：{expression}"

    return last_tag


def build_action_text_from_motion2text(motion2text_json_path):
    """
    从 motion2text.json 构建 name → action_text 的映射

    Args:
        motion2text_json_path: motion2text.json 文件路径

    Returns:
        dict: {name: action_text}
    """
    assert os.path.exists(motion2text_json_path), (
        f"motion2text.json 不存在: {motion2text_json_path}"
    )

    with open(motion2text_json_path, "r", encoding="utf-8") as f:
        motion2text = json.load(f)

    name_to_action = {}
    for name, text in motion2text.items():
        desc = extract_description(text)
        if desc:
            name_to_action[name] = desc

    print(f"[数据] 从 motion2text.json 构建动作标签映射: {len(name_to_action)} 条")
    return name_to_action


# ======================================================================
#  Part 4: 完整推理流水线
# ======================================================================

def run_pipeline_single(
    vllm_client,
    mask_model,
    action_text,
    audio_tokens,
    audio_features,
    name="unknown",
    offset=0,
    step=4,
    temperature=0.5,
    top_p=0.4,
    generate_steps=6,
    template="Human: {prompt}<|im_end|>\nAssistant:",
):
    """
    单样本完整推理流水线

    Args:
        vllm_client: VLLMClient 实例
        mask_model: AudioMotionTransformer 模型
        action_text: 动作标签文本
        audio_tokens: 完整的 audio token 列表（每帧一个 int）
        audio_features: (num_frames, 768) numpy array
        name: 样本名称
        offset: 采样偏移
        step: 采样间隔
        temperature: LLM 采样温度
        top_p: LLM top_p
        generate_steps: Mask Transformer 生成步数
        template: LLM prompt 模板

    Returns:
        dict: 包含 sparse_tokens, dense_tokens, timing 等信息；失败返回 None
    """
    result = {"name": name}

    # ---- Step 1: 构建 LLM prompt ----
    prompt, sampled_indices = construct_llm_prompt(
        action_text, audio_tokens, offset, step
    )
    text_list = [template.format(prompt=prompt)]

    print(f"  [Step 1] LLM 推理 | 关键帧数: {len(sampled_indices)}")

    # ---- Step 2: 调用 vLLM 预测稀疏 motion token plan ----
    t0 = time.time()
    llm_results = vllm_client.predict_motion_plan(
        text_list=text_list,
        temperature=temperature,
        top_p=top_p,
        max_tokens=1024,
        stop=["<|im_end|>"],
        base_token_start=0,
    )
    t1 = time.time()

    if llm_results is None or len(llm_results) == 0:
        print("  ❌ LLM 推理失败")
        return None

    llm_result = llm_results[0]
    sparse_tokens = llm_result["tokens"]
    num_keyframes = len(sparse_tokens["res_1"])
    print(f"  [Step 1] LLM 完成 | 预测关键帧: {num_keyframes}, 耗时: {t1 - t0:.3f}s")

    if num_keyframes < 2:
        print("  ❌ LLM 输出关键帧数不足 (<2)，无法插帧")
        return None

    # ---- Step 3: 转换为关键帧格式 ----
    keyframes = sparse_to_keyframes(sparse_tokens)

    # ---- Step 4: Mask Transformer 插帧 ----
    print(
        f"  [Step 2] Mask Transformer 插帧 | "
        f"关键帧: {num_keyframes}, audio 特征: {audio_features.shape}"
    )

    t2 = time.time()
    dense_tokens = interpolate_sequence(
        mask_model, keyframes, audio_features, generate_steps=generate_steps
    )
    t3 = time.time()

    # ---- Step 5: 长度对齐 —— 确保 motion tokens 与音频帧数严格一致 ----
    target_length = audio_features.shape[0]
    raw_length = len(dense_tokens)
    dense_tokens = ensure_length(dense_tokens, target_length)

    if raw_length != target_length:
        diff = target_length - raw_length
        op = "填充" if diff > 0 else "截断"
        print(
            f"  [对齐] 插帧输出 {raw_length} 帧, 音频 {target_length} 帧, "
            f"{op} {abs(diff)} 帧 → 最终 {len(dense_tokens)} 帧"
        )
    else:
        print(
            f"  [Step 2] 插帧完成 | 输出帧数: {len(dense_tokens)} (与音频等长), "
            f"耗时: {t3 - t2:.3f}s"
        )
        
    assert len(dense_tokens) == target_length, (
        f"长度对齐失败: dense_tokens={len(dense_tokens)}, audio={target_length}"
    )

    result.update({
        "action_text": action_text,
        "sparse_tokens": sparse_tokens,
        "dense_tokens": dense_tokens,
        "num_keyframes": num_keyframes,
        "num_dense_frames": len(dense_tokens),
        "num_audio_frames": target_length,
        "length_adjusted": raw_length != target_length,
        "length_adjustment": target_length - raw_length,
        "llm_raw_output": llm_result.get("raw_output", ""),
        "llm_total_len": llm_result.get("total_len"),
        "timing": {
            "llm_time": round(t1 - t0, 4),
            "interp_time": round(t3 - t2, 4),
            "total_time": round(t3 - t0, 4),
        },
    })

    return result


# ======================================================================
#  Part 5: Demo / Batch 模式
# ======================================================================

def demo_mode(args, vllm_client, mask_model, name_to_action,
              motion_token_dir, audio_token_dir, audio_feat_dir):
    """Demo 模式：单样本推理并展示结果"""
    # 加载 val split
    with open(args.val_split_file, "r") as f:
        val_names = [line.strip() for line in f if line.strip()]

    # 选择样本
    if args.sample_idx < len(val_names):
        target_name = val_names[args.sample_idx]
    else:
        print(f"[WARN] sample_idx={args.sample_idx} 超出范围 ({len(val_names)})，使用第 0 个")
        target_name = val_names[0]

    # 加载 motion tokens (GT)
    motion_tokens = load_motion_tokens(target_name, motion_token_dir)
    if motion_tokens is None:
        print(f"❌ 找不到 motion token 文件: {target_name}")
        return

    # 加载 audio tokens (for LLM prompt)
    audio_tokens = load_audio_tokens(target_name, audio_token_dir)
    if audio_tokens is None:
        print(f"❌ 找不到 audio token 文件: {target_name}")
        return

    # 确定动作标签
    if args.action_text:
        action_text = args.action_text
    else:
        action_text = name_to_action.get(target_name, "动作：说话")

    print(f"\n{'=' * 70}")
    print(f"  Demo 推理")
    print(f"  样本: {target_name}")
    print(f"  动作标签: {action_text}")
    print(f"  总帧数 (GT): {len(motion_tokens)}")
    print(f"  Audio token 数: {len(audio_tokens)}")
    print(f"{'=' * 70}")

    # 加载音频特征
    audio_feat = load_audio_features(target_name, audio_feat_dir)
    if audio_feat is None:
        print(f"❌ 找不到音频特征文件: {target_name}")
        return

    print(f"  Audio 特征 shape: {audio_feat.shape}")

    # 运行流水线
    result = run_pipeline_single(
        vllm_client,
        mask_model,
        action_text=action_text,
        audio_tokens=audio_tokens,
        audio_features=audio_feat,
        name=target_name,
        offset=args.offset,
        temperature=args.temperature,
        top_p=args.top_p,
        generate_steps=args.generate_steps,
    )

    if result is None:
        return

    # 对比 GT
    gt_tokens = motion_tokens
    print(f"\n  总帧数 — GT: {len(gt_tokens)}, 预测: {result['num_dense_frames']}")
    print(f"  关键帧数: {result['num_keyframes']}")
    print(
        f"  耗时 — LLM: {result['timing']['llm_time']:.3f}s, "
        f"插帧: {result['timing']['interp_time']:.3f}s, "
        f"总计: {result['timing']['total_time']:.3f}s"
    )

    if gt_tokens:
        num_show = min(12, max(len(gt_tokens), len(result["dense_tokens"])))
        print(f"\n  GT vs Pred (前 {num_show} 帧):")
        for i in range(num_show):
            gt = gt_tokens[i] if i < len(gt_tokens) else "N/A"
            pred = result["dense_tokens"][i] if i < len(result["dense_tokens"]) else "N/A"
            match = "✓" if gt == pred else "✗"
            print(f"    Frame {i:3d}: GT={str(gt):>30s}  Pred={str(pred):>30s}  {match}")

    # 保存结果
    result["gt_tokens"] = gt_tokens
    output_path = args.output_path
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存到: {output_path}")


def batch_mode(args, vllm_client, mask_model, name_to_action,
               motion_token_dir, audio_token_dir, audio_feat_dir):
    """Batch 模式：批量推理验证集"""
    with open(args.val_split_file, "r") as f:
        val_names = [line.strip() for line in f if line.strip()]
    # val_names = random.sample(val_names, min(100, len(val_names)))
    
    results = []
    success_count = 0
    fail_count = 0
    total_llm_time = 0.0
    total_interp_time = 0.0

    for idx, name in enumerate(tqdm(val_names, desc="Pipeline 推理")):
        # 加载 motion tokens (GT)
        motion_tokens = load_motion_tokens(name, motion_token_dir)
        if motion_tokens is None:
            fail_count += 1
            continue

        # 加载 audio tokens (for LLM prompt)
        audio_tokens = load_audio_tokens(name, audio_token_dir)
        if audio_tokens is None:
            fail_count += 1
            continue

        # 加载音频特征
        audio_feat = load_audio_features(name, audio_feat_dir)
        if audio_feat is None:
            fail_count += 1
            continue

        # 确定动作标签
        if args.action_text:
            action_text = args.action_text
        else:
            action_text = name_to_action.get(name, "动作：说话")

        # 运行流水线
        result = run_pipeline_single(
            vllm_client,
            mask_model,
            action_text=action_text,
            audio_tokens=audio_tokens,
            audio_features=audio_feat,
            name=name,
            offset=args.offset,
            temperature=args.temperature,
            top_p=args.top_p,
            generate_steps=args.generate_steps,
        )

        if result is not None:
            result["gt_tokens"] = motion_tokens
            result["num_frames_gt"] = len(motion_tokens)
            results.append(result)
            success_count += 1
            total_llm_time += result["timing"]["llm_time"]
            total_interp_time += result["timing"]["interp_time"]
        else:
            fail_count += 1

    # 保存结果
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"  批量推理完成")
    print(f"  成功: {success_count}, 失败: {fail_count}")
    if success_count > 0:
        print(f"  平均耗时 — LLM: {total_llm_time / success_count:.3f}s, "
              f"插帧: {total_interp_time / success_count:.3f}s")
    print(f"  结果已保存到: {args.output_path}")
    print(f"{'=' * 70}")


# ======================================================================
#  Part 6: 主函数
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="完整推理流水线: LLM Motion Token Plan + Mask Transformer 插帧 (Hubert Features)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # Demo 模式（单样本）
  python pipeline_infer.py \\
      --mask_ckpt ./outputs_audio_motion_interp/checkpoint-20000 \\
      --vllm_port 8095 \\
      --mode demo --sample_idx 0

  # Batch 模式（验证集）
  python pipeline_infer.py \\
      --mask_ckpt ./outputs_audio_motion_interp/checkpoint-20000 \\
      --vllm_port 8095 \\
      --mode batch --output_path ./pipeline_batch_results.json

  # 自定义动作标签
  python pipeline_infer.py \\
      --mask_ckpt ./outputs_audio_motion_interp/checkpoint-20000 \\
      --mode demo --sample_idx 5 --action_text "动作：抬头仰望"
        """,
    )

    # vLLM 服务配置
    parser.add_argument("--vllm_port", type=int, default=8095,
                        help="vLLM 服务端口 (默认: 8095)")
    parser.add_argument("--vllm_url", type=str, default=None,
                        help="vLLM 服务完整 URL (覆盖 --vllm_port)")

    # Mask Transformer 配置
    parser.add_argument("--mask_ckpt", type=str, required=True,
                        help="Mask Transformer checkpoint 路径")
    parser.add_argument("--device", type=str, default="cuda",
                        help="推理设备 (默认: cuda)")
    parser.add_argument("--generate_steps", type=int, default=6,
                        help="Mask Transformer 生成步数 (默认: 6)")

    # 数据路径
    parser.add_argument("--motion_token_dir", type=str, default=None,
                        help="motion token JSON 文件目录")
    parser.add_argument("--audio_token_dir", type=str, default=None,
                        help="hubert audio token JSON 文件目录")
    parser.add_argument("--audio_feat_dir", type=str, default=None,
                        help="hubert 音频特征 npy 文件目录")
    parser.add_argument("--val_split_file", type=str, default=None,
                        help="验证集 split 文件路径")
    parser.add_argument("--motion2text_json", type=str, default=None,
                        help="motion2text.json 路径 (用于获取动作标签)")

    # 推理参数
    parser.add_argument("--mode", type=str, default="demo",
                        choices=["demo", "batch"],
                        help="推理模式: demo (单样本) 或 batch (批量)")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Demo 模式的样本索引 (默认: 0)")
    parser.add_argument("--action_text", type=str, default=None,
                        help="动作标签文本 (不指定则从数据中自动获取)")
    parser.add_argument("--offset", type=int, default=0,
                        help="采样偏移 0-3 (默认: 0)")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="LLM 采样温度 (默认: 0.5)")
    parser.add_argument("--top_p", type=float, default=0.4,
                        help="LLM top_p (默认: 0.4)")
    parser.add_argument("--output_path", type=str, default="./pipeline_demo_result.json",
                        help="输出结果 JSON 路径")

    args = parser.parse_args()

    # ---- 默认路径 ----
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, "data")

    if args.motion_token_dir is None:
        args.motion_token_dir = os.path.join(data_dir, "motion_token_data")
    if args.audio_token_dir is None:
        args.audio_token_dir = os.path.join(data_dir, "audio_tokens_hubert_layer9_fps10")
    if args.audio_feat_dir is None:
        args.audio_feat_dir = os.path.join(data_dir, "audio_features_hubert_layer9_fps10")
    if args.val_split_file is None:
        args.val_split_file = os.path.join(data_dir, "split/test_file_list.txt")
    if args.motion2text_json is None:
        args.motion2text_json = os.path.join(data_dir, "text_data/motion2text.json")

    vllm_url = args.vllm_url or f"http://localhost:{args.vllm_port}"

    # ---- 检查 vLLM 服务 ----
    print(f"[vLLM] 检查服务: {vllm_url}")
    vllm_client = VLLMClient(vllm_url)
    if not vllm_client.health_check():
        print(f"❌ vLLM 服务不可用: {vllm_url}")
        print("   请先启动 vLLM 服务:")
        print("   cd gen_vllm_infer && bash scripts/at2m_v4_vllm_sft.sh")
        sys.exit(1)
    print("[vLLM] ✅ 服务正常\n")

    # ---- 加载 Mask Transformer ----
    mask_model = load_mask_transformer(args.mask_ckpt, device=args.device)
    print()

    # ---- 构建动作标签映射（从 motion2text.json）----
    print(f"[数据] 加载 motion2text: {args.motion2text_json}")
    name_to_action = build_action_text_from_motion2text(args.motion2text_json)

    print()

    # ---- 运行推理 ----
    if args.mode == "demo":
        demo_mode(args, vllm_client, mask_model, name_to_action,
                  args.motion_token_dir, args.audio_token_dir, args.audio_feat_dir)
    elif args.mode == "batch":
        batch_mode(args, vllm_client, mask_model, name_to_action,
                   args.motion_token_dir, args.audio_token_dir, args.audio_feat_dir)


if __name__ == "__main__":
    main()
