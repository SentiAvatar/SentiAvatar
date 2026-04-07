#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   mta25joints_params.py
@Time    :   2025/09/02 20:41:59
@Author  :   Chuhao Jin 
@Contact :   jinchuhao@ruc.edu.cn
'''
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Optional

import os
import json
import numpy as np
import torch

from utils.visualization_torch import BVH_mod as BVH


# -----------------------------
# 纯函数：工具
# -----------------------------

def get_offsets_joints(src_offset: torch.Tensor, raw_offset: torch.Tensor) -> torch.Tensor:
    """
    根据 template offset 的骨骼长度，把 unit raw_offset 缩放成目标 offsets
    src_offset: (J, 3)  template offsets（每个关节的真实骨长方向向量）
    raw_offset: (J, 3)  归一化的方向 offset（你 t2m_raw_offsets）
    """
    assert len(src_offset.shape) == 2
    offsets = raw_offset.clone()
    for i in range(1, raw_offset.shape[0]):
        offsets[i] = torch.norm(src_offset[i], p=2, dim=0) * offsets[i]
    return offsets


def build_parents(joints_num: int, kinematic_chain: List[List[int]]) -> List[int]:
    parents = [0] * joints_num
    parents[0] = -1
    for chain in kinematic_chain:
        for j in range(1, len(chain)):
            parents[chain[j]] = chain[j - 1]
    return parents


def load_from_json(file):
    with open(file, "r") as f:
        data = json.load(f)
    return data

# -----------------------------
# dataclass：骨架配置
# -----------------------------

@dataclass
class SkeletonSpec:
    # ---- meta file paths ----
    template_bvh: str
    mta_joint_sorted_file: str
    skeleton_file: str
    demo_quats_file: Optional[str]

    # ---- basic skeleton meta ----
    joints_num: int = 63
    end_points: List[str] = field(default_factory=lambda: ["head", "ball_l", "ball_r", "hand_l", "hand_r"])

    # MetaHuman mocap indices
    r_hip: int = 1
    l_hip: int = 5
    l_idx1: int = 6
    l_idx2: int = 7
    fid_r: List[int] = field(default_factory=lambda: [3, 4])
    fid_l: List[int] = field(default_factory=lambda: [7, 8])
    face_joint_indx: List[int] = field(default_factory=lambda: [1, 5, 21, 18])

    # ---- kinematic structure ----
    kinematic_chain: List[List[int]] = field(default_factory=list)

    # ---- offsets & joint map ----
    t2m_raw_offsets: np.ndarray = field(default_factory=lambda: np.zeros((63, 3), dtype=np.float32))
    src_joint_dict: Dict[str, int] = field(default_factory=dict)

    # ---- joint list/order ----
    SKELETON: List[str] = field(default_factory=list)
    skel_dict: Dict[str, int] = field(default_factory=dict)
    origin_whole_joints_id: List[int] = field(init=False)
    mta_joint_idx_dict: Dict[int, int] = field(init=False)

    # ---- face default values ----
    static_face: List[float] = field(default_factory=list)

    # ---- computed tensors ----
    n_raw_offsets: torch.Tensor = field(init=False)   # (J,3)
    parents: List[int] = field(init=False)

    # from meta files
    skeleton_json: Dict = field(init=False)           # mta_skeleton.json 内容
    joint_list: List[str] = field(init=False)         # 过滤后的关节排序列表（只保留 src_joint_dict 有的）
    template_offset: torch.Tensor = field(init=False) # (J,3) 映射到 src_joint_dict index 上
    tgt_offsets: torch.Tensor = field(init=False)     # (J,3) 缩放后的 offsets

    def __post_init__(self):
        # 计算 raw offsets tensor
        self.n_raw_offsets = torch.from_numpy(self.t2m_raw_offsets.astype(np.float32))
        # parents 由 kinematic_chain 推导
        self.parents = build_parents(self.joints_num, self.kinematic_chain)

        # skel_dict
        if self.SKELETON and not self.skel_dict:
            self.skel_dict = {j: i for i, j in enumerate(self.SKELETON)}

    # -----------------------------
    # 工厂：从 meta 文件构建
    # -----------------------------
    
    @classmethod
    def from_meta_files(
        cls,
        *,
        template_bvh: str,
        mta_joint_sorted_file: str,
        skeleton_file: str,
        end_points: Optional[List[str]] = None,
        joints_num: int = 63,
        kinematic_chain_file: Optional[str] = None,
        t2m_raw_offsets: Optional[List[List[int]]] = None,
        src_joint_dict_file: Optional[str] = None,
        joint_nodes_file: Optional[str] = None,
        static_face_file: Optional[str] = None,
        demo_quats_file: Optional[str] = None,
        # mocap indices 可覆盖
        r_hip: int = 1,
        l_hip: int = 5,
        l_idx1: int = 6,
        l_idx2: int = 7,
        fid_r: Optional[List[int]] = None,
        fid_l: Optional[List[int]] = None,
        face_joint_indx: Optional[List[int]] = None,
    ) -> "SkeletonSpec":
        if end_points is None:
            end_points = ["head", "ball_l", "ball_r", "hand_l", "hand_r"]
        if fid_r is None:
            fid_r = [3, 4]
        if fid_l is None:
            fid_l = [7, 8]
        if face_joint_indx is None:
            face_joint_indx = [1, 5, 21, 18]

        if kinematic_chain_file is None:
            raise ValueError("kinematic_chain is required")
        if t2m_raw_offsets is None:
            raise ValueError("t2m_raw_offsets is required")
        if src_joint_dict_file is None:
            raise ValueError("src_joint_dict is required")
        if joint_nodes_file is None:
            raise ValueError("SKELETON is required")
        if static_face_file is None:
            static_face = []
        
        kinematic_chain = load_from_json(kinematic_chain_file)
        src_joint_dict = load_from_json(src_joint_dict_file)
        SKELETON = load_from_json(joint_nodes_file)
        static_face = load_from_json(static_face_file)
        
        t2m_raw_offsets = np.array(t2m_raw_offsets, dtype=np.float32)
        
        spec = cls(
            template_bvh=template_bvh,
            mta_joint_sorted_file=mta_joint_sorted_file,
            skeleton_file=skeleton_file,
            demo_quats_file=demo_quats_file,
            joints_num=joints_num,
            end_points=end_points,
            r_hip=r_hip,
            l_hip=l_hip,
            l_idx1=l_idx1,
            l_idx2=l_idx2,
            fid_r=fid_r,
            fid_l=fid_l,
            face_joint_indx=face_joint_indx,
            kinematic_chain=kinematic_chain,
            t2m_raw_offsets=t2m_raw_offsets,
            src_joint_dict=src_joint_dict,
            SKELETON=SKELETON,
            static_face=static_face,
        )

        # 1) load jsons
        spec.skeleton_json = json.load(open(skeleton_file, "r"))
        joint_list = json.load(open(mta_joint_sorted_file, "r"))
        # 只保留 src_joint_dict 里存在的 joint（你现在做的过滤）
        spec.joint_list = [item for item in joint_list if item in spec.src_joint_dict]
        # 2) load template bvh
        spec.bvh_template = BVH.load(template_bvh, need_quater=True)

        # 3) 构建 template_offset: 映射到 src_joint_dict index 上
        template_offset = np.zeros_like(spec.bvh_template.offsets)
        # 注意：你原逻辑是从 1 开始，用 joint_list[i] 对应 template.offsets[i]
        for i in range(1, len(spec.bvh_template.offsets)):
            if i >= len(spec.joint_list):
                break
            joint_name = spec.joint_list[i]
            joint_id = spec.src_joint_dict[joint_name]
            template_offset[joint_id] = spec.bvh_template.offsets[i]

        spec.template_offset = torch.from_numpy(template_offset.astype(np.float32))

        # 4) tgt_offsets = get_offsets_joints(template_offset, n_raw_offsets)
        spec.tgt_offsets = get_offsets_joints(spec.template_offset, spec.n_raw_offsets)
        spec.origin_whole_joints_id = []
        for joint in spec.src_joint_dict.keys():
            spec.origin_whole_joints_id.append(spec.skel_dict[joint])
        print("spec.skel_dict:", spec.skel_dict)
        spec.mta_joint_idx_dict = {}
        for idx, joint_id in enumerate(spec.origin_whole_joints_id):
            spec.mta_joint_idx_dict[joint_id] = idx 
        return spec
