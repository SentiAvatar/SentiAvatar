#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   params.py
@Time    :   2025/12/17 00:33:12
@Author  :   Chuhao Jin 
@Contact :   jinchuhao@ruc.edu.cn
'''

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Any


@dataclass
class TTSConfig:
    voice_name: str = "susu"
    api_url: str = f"http://127.0.0.1:7803/tts_url"
    output_path: str = "./happy.wav"
    speaking_rate: float = 1.0
    pitch: float = 0.0
    style: str = "chat"          # "chat" / "sing" 等风格标签
    

@dataclass
class SamplingParams:
    temperature: float = 0.3
    top_p: float = 0.4
    max_tokens: int = 1024
    stop: Optional[List[str]] = None
    base_token_start: int = 0


@dataclass
class ComplParams:
    infill_ckpt: str = "checkpoints/mask_transformer"
    tmr_model: str = "checkpoints/eval_model"

@dataclass
class BaseConfig:
    do_vqvae: bool = True # True for server and False for server_v2
    return_bvh_anim: bool = True

@dataclass
class ServiceURL:
    action_vllm_api: str = "http://localhost:8086"
    face_vllm_url = "http://127.0.0.1:8096"

@dataclass
class CkptParams:
    base_path: str = "checkpoints"
    motion_infll_model_path: str= "mask_transformer"
    text_model_path: str = "eval_model"
    face_vq_ckpt_file: str = "face_vqvae/pytorch_model_face_fad2cl_1209_codesize2048_codelength512.bin" # 0feats token = 
    face_infill_model_path: str = "face_infill_transformer"
    audo_emb_model_path :str = "chinese-hubert-base"
    motion_vq_name: str = "VQVAE_nb_code2048_down_t1_qdp0.8_num_quantizers1_add_2CNN"
    motion_vq_dataset_name: str = "mocap_pyramid_1016_fix_dict_bug"
    motion_vq_ckpt: str = "epoch_15_loss_0.07886.tar"
