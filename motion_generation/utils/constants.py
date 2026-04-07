#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   constants.py
@Time    :   2026/01/18 16:00:00
@Author  :   Chuhao Jin 
@Contact :   jinchuhao@ruc.edu.cn

@Description:
    MTA 63 节点骨骼常量定义
'''

import numpy as np

# 关节数量
JOINTS_NUM = 63

# 关节 ID 定义
BODY_JOINTS_ID = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
LEFT_HAND_JOINTS_ID = [23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]
RIGHT_HAND_JOINTS_ID = [24, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]

# 左右关节对应
LEFT_JOINTS = [5, 6, 7, 8, 17, 18, 19, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]
RIGHT_JOINTS = [1, 2, 3, 4, 20, 21, 22, 24, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]
CENTER_JOINTS = [0, 9, 10, 11, 12, 13, 14, 15, 16]

# 脚部关节 ID (用于 foot position loss)
# right foot: 3 (foot_r), 4 (ball_r)
# left foot: 7 (foot_l), 8 (ball_l)
FOOT_JOINTS_ID = [3, 4, 7, 8]

# 运动链
KINEMATIC_CHAIN = [
    [0, 9, 10, 11, 12, 13, 14, 15, 16],  # 脊柱
    [0, 5, 6, 7, 8],  # 左腿
    [0, 1, 2, 3, 4],  # 右腿
    [13, 17, 18, 19, 23],  # 左臂
    [13, 20, 21, 22, 24],  # 右臂
    [23, 25, 26, 27, 28],  # 左手食指
    [23, 29, 30, 31, 32],  # 左手中指
    [23, 33, 34, 35, 36],  # 左手无名指
    [23, 37, 38, 39, 40],  # 左手小指
    [23, 41, 42, 43],  # 左手拇指
    [24, 44, 45, 46, 47],  # 右手食指
    [24, 48, 49, 50, 51],  # 右手中指
    [24, 52, 53, 54, 55],  # 右手无名指
    [24, 56, 57, 58, 59],  # 右手小指
    [24, 60, 61, 62],  # 右手拇指
]

# 关节名称到索引的映射
SRC_JOINT_DICT = {
    'pelvis': 0, 
    
    # 右腿
    'thigh_r': 1, 
    'calf_r': 2,
    'foot_r': 3, 
    'ball_r': 4, 
        
    # 左腿
    'thigh_l': 5, 
    'calf_l': 6,
    'foot_l': 7, 
    'ball_l': 8, 
    
    # 骨盆区域
    'spine_01': 9, 
    'spine_02': 10,
    'spine_03': 11, 
    'spine_04': 12, 
    'spine_05': 13, 
    'neck_01': 14, 
    'neck_02': 15, 
    'head': 16, 
    
    # 左臂
    'clavicle_l': 17,
    'upperarm_l': 18, 
    'lowerarm_l': 19, 
    'hand_l': 23, 
    
    # 右臂
    'clavicle_r': 20, 
    'upperarm_r': 21, 
    'lowerarm_r': 22, 
    'hand_r': 24,
    
    # 左手
    'index_metacarpal_l': 25, 
    'index_01_l': 26, 
    'index_02_l': 27, 
    'index_03_l': 28, 
    'middle_metacarpal_l': 29,
    'middle_01_l': 30, 
    'middle_02_l': 31, 
    'middle_03_l': 32, 
    'ring_metacarpal_l': 33,
    'ring_01_l': 34,
    'ring_02_l': 35, 
    'ring_03_l': 36, 
    'pinky_metacarpal_l': 37, 
    'pinky_01_l': 38, 
    'pinky_02_l': 39, 
    'pinky_03_l': 40, 
    'thumb_01_l': 41, 
    'thumb_02_l': 42, 
    'thumb_03_l': 43, 
    
    # 右手
    'index_metacarpal_r': 44,
    'index_01_r': 45, 
    'index_02_r': 46, 
    'index_03_r': 47, 
    'middle_metacarpal_r': 48,
    'middle_01_r': 49, 
    'middle_02_r': 50, 
    'middle_03_r': 51, 
    'ring_metacarpal_r': 52,
    'ring_01_r': 53, 
    'ring_02_r': 54,
    'ring_03_r': 55, 
    'pinky_metacarpal_r': 56, 
    'pinky_01_r': 57, 
    'pinky_02_r': 58,
    'pinky_03_r': 59, 
    'thumb_01_r': 60, 
    'thumb_02_r': 61, 
    'thumb_03_r': 62,
}
