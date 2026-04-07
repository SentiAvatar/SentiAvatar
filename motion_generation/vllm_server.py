#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   t2m_vllm_service.py
@Time    :   2026/01/04 18:27:11
@Author  :   Chuhao Jin 
@Contact :   jinchuhao@ruc.edu.cn
'''

import os
import torch
import json, time 
import numpy as np
from vllm import LLM, SamplingParams
from flask import Flask, jsonify, request
from transformers import AutoTokenizer
import argparse

app = Flask(__name__)
app.json.sort_keys = False

# 解析参数
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8081)
parser.add_argument('--model_path', type=str)
args = parser.parse_args()

port = args.port
model_path = args.model_path
print("model_path:", model_path)

# gpu_id = max(0, port - 8081)
# os.environ['CUDA_VISIBLE_DEVICES'] = f"{gpu_id}"

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    padding_side="left"
)

# 🚀 关键优化：配置vLLM降低显存占用
stop_token = "<|im_end|>"
llm = LLM(
    model=model_path,
    # 显存优化参数
    gpu_memory_utilization=0.2,  # 降低显存利用率，避免OOM [1,4](@ref)
    max_model_len=1600,  # 限制最大上下文长度，减少KV缓存 [1](@ref)
    max_num_batched_tokens=1600,  # 限制批次token数，控制峰值显存 [1](@ref)
    max_num_seqs=1,  # 限制并发序列数 [1](@ref)
    enable_chunked_prefill=True,  # 长提示词分块处理 [1](@ref)
    # enforce_eager=True,  # 关闭CUDA图，减少内存池开销 [1](@ref)
    # 可选量化（如果模型支持）
    # quantization="awq",  # 使用AWQ量化减少75%显存 [1,3](@ref)
    # dtype="bfloat16",  # 混合精度推理 [1](@ref)
    # 系统优化
    # swap_space=4,  # 开启CPU内存交换 [2](@ref)
)

# 🆕 默认采样参数（当请求未提供时使用）
DEFAULT_SAMPLING_PARAMS = {
    "temperature": 0.3,
    "top_p": 0.4,
    "max_tokens": 512,
    "stop": "<|im_end|>"
}

def validate_sampling_params(data):
    """验证和提取采样参数[5](@ref)"""
    params = DEFAULT_SAMPLING_PARAMS.copy()
    
    # 从请求中获取参数，如未提供则使用默认值
    if 'temperature' in data:
        temp = float(data.get('temperature', params['temperature']))
        params['temperature'] = max(0.05, min(2.0, temp))  # 限制在合理范围内
    
    if 'top_p' in data:
        top_p = float(data.get('top_p', params['top_p']))
        params['top_p'] = max(0.05, min(1.0, top_p))  # 限制在0.1-1.0范围内
    
    if 'max_tokens' in data:
        max_tokens = int(data.get('max_tokens', params['max_tokens']))
        params['max_tokens'] = max(64, min(2048, max_tokens))  # 限制token数量
    
    if 'stop' in data:
        stop = data.get('stop')
        # 支持字符串或字符串列表[1](@ref)
        if isinstance(stop, list):
            params['stop'] = stop
        elif isinstance(stop, str) and stop.strip():
            params['stop'] = stop.strip()
        else:
            params['stop'] = "<|im_end|>"  # 如果没有提供有效的stop token，设为"<|im_end|>"
    
    return params

@app.route('/text_to_motion', methods=['POST', 'GET'])  # 🆕 改为POST请求，更适合传递复杂参数
def t2m_api():
    try:
        # 🆕 支持JSON格式的POST请求
        if request.is_json:
            data = request.get_json()
        else:
            # 也支持表单格式
            data = request.form.to_dict()
        
        # 🆕 验证必需的text_list参数
        if 'text_list' not in data:
            return jsonify({"error": "缺少必需的参数: text_list"}), 400
        

        # 解析文本列表
        if isinstance(data['text_list'], str):
            try:
                text_list = json.loads(data['text_list'])
            except json.JSONDecodeError:
                text_list = [data['text_list']]  # 如果是单个字符串，转为列表
        else:
            text_list = data['text_list']

        if not text_list or not isinstance(text_list, list):
            return jsonify({"error": "text_list必须是包含文本的列表"}), 400

        len_list = None 
        if 'len_list' in data:
            if isinstance(data['len_list'], str):
                try:
                    len_list = json.loads(data['len_list'])
                except json.JSONDecodeError:
                    len_list = [data['len_list']]  # 如果是单个字符串，转为列表
            else:
                len_list = data['text_list']

        
        # 🆕 动态获取采样参数
        sampling_config = validate_sampling_params(data)
        
        # 🆕 创建采样参数对象
        sampling_params = SamplingParams(
            temperature=sampling_config['temperature'],
            top_p=sampling_config['top_p'],
            max_tokens=sampling_config['max_tokens'],
            stop=sampling_config['stop'],
            skip_special_tokens=False  # 跳过特殊token[3](@ref)
        )
        
        if "Human: " in text_list[0] and "Assistant:" in text_list[0]:
            prompts = [text for text in text_list]
        else:
            prompts = [f'Human: {text}\nAssistant:' for text in text_list]

        
        # 执行推理
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        inference_time = time.time() - start_time
        
        # 处理结果
        results = []
        for i, output in enumerate(outputs):
            if output.outputs:
                response_text = output.outputs[0].text
                # 清理文本
                cleaned_text = response_text.strip()
                if sampling_config['stop']:
                    if isinstance(sampling_config['stop'], list):
                        for stop_token in sampling_config['stop']:
                            cleaned_text = cleaned_text.split(stop_token)[0]
                    else:
                        cleaned_text = cleaned_text.split(sampling_config['stop'])[0]
                cleaned_text = cleaned_text.replace("<unk>", "").replace(" ", "")
                results.append(cleaned_text)
            else:
                results.append("")  # 处理空响应
        
        motion_sequence_list = [(text, result) for text, result in zip(text_list, results)]
        
        # 🆕 返回详细的响应信息，包含使用的参数
        return jsonify({
            "motion_sequence_list": motion_sequence_list,
            "model_name": model_path.split("/")[-1],
            "used_parameters": sampling_config,  # 🆕 显示实际使用的参数
            "inference_time": round(inference_time, 3),
            "batch_size": len(text_list)
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"处理请求时出错: {str(e)}"}), 500

# 🆕 添加参数说明端点
@app.route('/parameters', methods=['GET'])
def get_parameters_info():
    """返回支持的参数信息[2](@ref)"""
    return jsonify({
        "supported_parameters": {
            "text_list": {
                "type": "list of strings or string",
                "required": True,
                "description": "要处理的文本列表"
            },
            "temperature": {
                "type": "float",
                "default": DEFAULT_SAMPLING_PARAMS['temperature'],
                "range": "0.1-2.0",
                "description": "控制生成随机性，值越高越随机"
            },
            "top_p": {
                "type": "float", 
                "default": DEFAULT_SAMPLING_PARAMS['top_p'],
                "range": "0.1-1.0",
                "description": "核采样参数，控制词汇选择的多样性"
            },
            "max_tokens": {
                "type": "integer",
                "default": DEFAULT_SAMPLING_PARAMS['max_tokens'],
                "range": "64-2048", 
                "description": "最大生成token数量"
            },
            "stop": {
                "type": "string or list of strings",
                "default": DEFAULT_SAMPLING_PARAMS['stop'],
                "description": "停止生成的token或token列表"
            }
        }
    })

# 🆕 健康检查端点[2](@ref)
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "gpu_memory_allocated": f"{torch.cuda.memory_allocated() // 1024**2}MB" if torch.cuda.is_available() else "N/A"
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port, debug=False)