#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   mta63joints_constants.py
@Time    :   2025/12/19 15:37:32
@Author  :   Chuhao Jin 
@Contact :   jinchuhao@ruc.edu.cn
'''
import json 
import os as _os
_BASE_DIR = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "meta")

template_bvh = _os.path.join(_BASE_DIR, "mta63joints/template_susu_retarget_63nodes.bvh")
mta_joint_sorted_file = _os.path.join(_BASE_DIR, "mta63joints/mta_joint_sorted_list.json")
skeleton_file = _os.path.join(_BASE_DIR, "mta63joints/mta_skeleton.json")
kinematic_chain_file = _os.path.join(_BASE_DIR, "mta63joints/kinematic_chain.json")
src_joint_dict_file = _os.path.join(_BASE_DIR, "mta63joints/src_joint_dict.json")
joint_nodes_file = _os.path.join(_BASE_DIR, "mta63joints/joint_nodes.json")
static_face_file = _os.path.join(_BASE_DIR, "mta63joints/static_face.json")
demo_quats_file = _os.path.join(_BASE_DIR, "demo_quat.npy")

t2m_raw_offsets = [
    [0, 0, 0], # 0 pelvis
    # 右腿
    [-1, 0, 0], # 1 thigh_r 
    [0, -1, 0], # 2 calf_r
    [0, -1, 0], # 3 foot_r
    [0, 0, 1], # 4 ball_r
    
    # 左腿
    [1, 0, 0], # 5 thigh_l
    [0, -1, 0], # 6 calf_l
    [0, -1, 0], # 7 foot_l
    [0, 0, 1], # 8 ball_l
    
    # 骨盆区域
    [0, 1, 0], # 9 spine_01
    [0, 1, 0], # 10 spine_02
    [0, 1, 0], # 11 spine_03
    [0, 1, 0], # 12 spine_04
    [0, 1, 0], # 13 spine_05
    [0, 1, 0], # 14 neck_01
    [0, 1, 0], # 15 neck_02
    [0, 0, 1], # 16 head
    
    # 左臂
    [1, 0, 0], # 17 clavicle_l
    [1, 0, 0], # 18 upperarm_l
    [1, 0, 0], # 19 lowerarm_l
    # "hand_l": "lowerarm_l", # 23: 19
    
    # 右臂
    [-1, 0, 0], # 20 clavicle_r
    [-1, 0, 0], # 21 upperarm_r
    [-1, 0, 0], # 22 lowerarm_r
    # "hand_r": "lowerarm_r", # 45: 22
    
    [1, 0, 0], # 23 hand_l
    [-1, 0, 0], # 24 hand_r
    
    # 左手
    [1, 0, 0], # 'index_metacarpal_l': 25, 
    [1, 0, 0], #'index_01_l': 26, 
    [1, 0, 0], #'index_02_l': 27, 
    [1, 0, 0], #'index_03_l': 28, 
    
    [1, 0, 0], #'middle_metacarpal_l': 29,
    [1, 0, 0], #'middle_01_l': 30, 
    [1, 0, 0], #'middle_02_l': 31, 
    [1, 0, 0], #'middle_03_l': 32, 
    
    [1, 0, 0], #'ring_metacarpal_l': 33,
    [1, 0, 0], #'ring_01_l': 34,
    [1, 0, 0], #'ring_02_l': 35, 
    [1, 0, 0], #'ring_03_l': 36, 
    
    [1, 0, 0], #'pinky_metacarpal_l': 37, 
    [1, 0, 0], #'pinky_01_l': 38, 
    [1, 0, 0], #'pinky_02_l': 39, 
    [1, 0, 0], #'pinky_03_l': 40, 
    
    [1, 0, 0], #'thumb_01_l': 41, 
    [1, 0, 0], #'thumb_02_l': 42, 
    [1, 0, 0], #'thumb_03_l': 43, 
    
    # 右手
    [-1, 0, 0], # 'index_metacarpal_r': 44,
    [-1, 0, 0], #'index_01_r': 45, 
    [-1, 0, 0], #'index_02_r': 46, 
    [-1, 0, 0], #'index_03_r': 47, 
    
    [-1, 0, 0], #'middle_metacarpal_r': 48,
    [-1, 0, 0], #'middle_01_r': 49, 
    [-1, 0, 0], #'middle_02_r': 50, 
    [-1, 0, 0], #'middle_03_r': 51, 
    
    [-1, 0, 0], #'ring_metacarpal_r': 52,
    [-1, 0, 0], #'ring_01_r': 53, 
    [-1, 0, 0], #'ring_02_r': 54,
    [-1, 0, 0], #'ring_03_r': 55, 
    
    [-1, 0, 0], #'pinky_metacarpal_r': 56, 
    [-1, 0, 0], #'pinky_01_r': 57, 
    [-1, 0, 0], #'pinky_02_r': 58,
    [-1, 0, 0], #'pinky_03_r': 59, 
    
    [-1, 0, 0], #'thumb_01_r': 60, 
    [-1, 0, 0], #'thumb_02_r': 61, 
    [-1, 0, 0], #'thumb_03_r': 62,
    ]