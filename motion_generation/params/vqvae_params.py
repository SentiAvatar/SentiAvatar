#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   vqvae_params.py
@Time    :   2025/12/16 21:21:33
@Author  :   Chuhao Jin 
@Contact :   jinchuhao@ruc.edu.cn
'''

# models/vqvae/vqvae_params.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class VQVAEParams:
    # ==== 你 init 传入的核心参数 ====
    vq_name: str
    base_opt_path: str
    dataset_name: str
    ckpt_file: str
    device: str = "cuda:0"

    # ==== 你原 arg_parse 默认值（保留） ====
    batch_size: int = 256
    body_part: str = "whole"
    window_size: int = 64
    gpu_id: int = 0
    debug: bool = False
    local_rank: int = 0

    # optimization（虽然推理不用，但保留，避免 opt 依赖缺字段）
    max_epoch: int = 50
    warm_up_iter: int = 2000
    lr: float = 2e-4
    milestones: List[int] = field(default_factory=lambda: [20000, 30000])
    gamma: float = 0.1
    weight_decay: float = 0.0
    commit: float = 0.02
    loss_vel: float = 0.5
    recons_loss: str = "l1_smooth"

    # vqvae arch（保留默认）
    code_dim: int = 512
    nb_code: int = 512
    mu: float = 0.99
    down_t: int = 2
    stride_t: int = 2
    width: int = 512
    depth: int = 3
    dilation_growth_rate: int = 3
    output_emb_width: int = 512
    vq_act: str = "relu"
    vq_norm: str | None = None
    num_quantizers: int = 3
    shared_codebook: bool = False
    quantize_dropout_prob: float = 0.2
    ext: str = "default"

    # other
    name: str = "test"
    is_continue: bool = False
    checkpoints_dir: str = "./ckpt"
    log_every: int = 10
    save_latest: int = 500
    save_every_e: int = 2
    eval_every_e: int = 1
    feat_bias: float = 5.0
    which_epoch: str = "all"
    seed: int = 3407

    # ==== 你 load_model 里写死的 runtime 字段 ====
    vq_cnn_depth: int = 3
    unit_length: int = 2

    # ==== 你 load_configuration 里写死的维度/关节/链 ====
    dim_pose: int = 755
    body_dim_pose: int = 296
    left_dim_pose: int = 240
    right_dim_pose: int = 240

    body_joints_num: int = 24
    left_joints_num: int = 20
    right_joints_num: int = 20

    # joints id（照你原始写法）
    body_joints_id: List[int] = field(default_factory=lambda: list(range(0, 25)))
    left_hand_joints_id: List[int] = field(default_factory=lambda: list(range(25, 44)))
    right_hand_joints_id: List[int] = field(default_factory=lambda: list(range(44, 63)))

    # 你原来 joints_ids = body + left + right
    @property
    def joints_ids(self) -> List[int]:
        return self.body_joints_id + self.left_hand_joints_id + self.right_hand_joints_id

    # 关节配置（你原来的 joint_config）
    @property
    def joint_config(self) -> Dict[str, int]:
        return {"body": self.body_joints_num, "left": self.left_joints_num, "right": self.right_joints_num}

    # 运动学链（你原来的 t2m_body_hand_kinematic_chain）
    @property
    def t2m_body_hand_kinematic_chain(self) -> List[List[int]]:
        return [
            [0, 9, 10, 11, 12, 13, 14, 15, 16],
            [0, 5, 6, 7, 8],
            [0, 1, 2, 3, 4],
            [13, 17, 18, 19, 23],
            [13, 20, 21, 22, 24],
            [23, 25, 26, 27, 28],
            [23, 29, 30, 31, 32],
            [23, 33, 34, 35, 36],
            [23, 37, 38, 39, 40],
            [23, 41, 42, 43],
            [24, 44, 45, 46, 47],
            [24, 48, 49, 50, 51],
            [24, 52, 53, 54, 55],
            [24, 56, 57, 58, 59],
            [24, 60, 61, 62],
        ]

    # ==== 路径 ====
    @property
    def vq_path(self) -> str:
        return f"{self.base_opt_path}/{self.dataset_name}/{self.vq_name}"

    @property
    def opt_path(self) -> str:
        return f"{self.vq_path}/opt.txt"

    @property
    def mean_std_folder(self) -> str:
        return f"{self.vq_path}/meta"
