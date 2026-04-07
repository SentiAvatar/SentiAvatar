#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   mta_63nodes.py
@Time    :   2025/09/03 14:28:58
@Author  :   Chuhao Jin 
@Contact :   jinchuhao@ruc.edu.cn
'''
import numpy as np 


############################ metahuman mocap ###################
# thigh_r thigh_l 
r_hip, l_hip = 1, 5 
joints_num = 63

# Lower legs
l_idx1, l_idx2 = 6, 7  # calf_l foot_r 

# Right/Left foot
fid_r, fid_l = [3, 4], [7, 8]
# Face direction, thigh_r thigh_l upperarm_r, upperarm_l
face_joint_indx = [1, 5, 21, 18]
##########################################################

body_joints_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
left_hand_joints_id = [23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]
right_hand_joints_id = [24, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]
assert len(body_joints_id) + len(left_hand_joints_id) + len(right_hand_joints_id) - 2 == joints_num

left_joints = [5, 6, 7, 8, 17, 18, 19, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]
right_joints = [1, 2, 3, 4, 20, 21, 22, 24, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]
center_joints = [0, 9, 10, 11, 12, 13, 14, 15, 16]

    
kinematic_chain = [
    [0, 9, 10, 11, 12, 13, 14, 15, 16], # 脊柱
    [0, 5, 6, 7, 8], # 左腿
    [0, 1, 2, 3, 4], # 右腿
    [13, 17, 18, 19, 23], # 左臂
    [13, 20, 21, 22, 24], # 右臂
    [23, 25, 26, 27, 28],
    [23, 29, 30, 31, 32],
    [23, 33, 34, 35, 36],
    [23, 37, 38, 39, 40],
    [23, 41, 42, 43],
    [24, 44, 45, 46, 47],
    [24, 48, 49, 50, 51],
    [24, 52, 53, 54, 55],
    [24, 56, 57, 58, 59],
    [24, 60, 61, 62]
]


HIERARCHY = {
    "spine_01": "pelvis",
    "spine_02": "spine_01",
    "spine_03": "spine_02",
    "spine_04": "spine_03",
    "spine_05": "spine_04",
    "neck_01": "spine_05",
    "neck_02": "neck_01",
    "head": "neck_02",
    "thigh_l": "pelvis",
    "calf_l": "thigh_l",
    "foot_l": "calf_l",
    "ball_l": "foot_l",
    "thigh_r": "pelvis",
    "calf_r": "thigh_r",
    "foot_r": "calf_r",
    "ball_r": "foot_r",
    "clavicle_l": "spine_05",
    "upperarm_l": "clavicle_l",
    "lowerarm_l": "upperarm_l",
    "hand_l": "lowerarm_l",
    "thumb_01_l": "hand_l",
    "thumb_02_l": "thumb_01_l",
    "thumb_03_l": "thumb_02_l",
    "index_metacarpal_l": "hand_l",
    "index_01_l": "index_metacarpal_l",
    "index_02_l": "index_01_l",
    "index_03_l": "index_02_l",
    "middle_metacarpal_l": "hand_l",
    "middle_01_l": "middle_metacarpal_l",
    "middle_02_l": "middle_01_l",
    "middle_03_l": "middle_02_l",
    "ring_metacarpal_l": "hand_l",
    "ring_01_l": "ring_metacarpal_l",
    "ring_02_l": "ring_01_l",
    "ring_03_l": "ring_02_l",
    "pinky_metacarpal_l": "hand_l",
    "pinky_01_l": "pinky_metacarpal_l",
    "pinky_02_l": "pinky_01_l",
    "pinky_03_l": "pinky_02_l",
    "clavicle_r": "spine_05",
    "upperarm_r": "clavicle_r",
    "lowerarm_r": "upperarm_r",
    "hand_r": "lowerarm_r",
    "thumb_01_r": "hand_r",
    "thumb_02_r": "thumb_01_r",
    "thumb_03_r": "thumb_02_r",
    "index_metacarpal_r": "hand_r",
    "index_01_r": "index_metacarpal_r",
    "index_02_r": "index_01_r",
    "index_03_r": "index_02_r",
    "middle_metacarpal_r": "hand_r",
    "middle_01_r": "middle_metacarpal_r",
    "middle_02_r": "middle_01_r",
    "middle_03_r": "middle_02_r",
    "ring_metacarpal_r": "hand_r",
    "ring_01_r": "ring_metacarpal_r",
    "ring_02_r": "ring_01_r",
    "ring_03_r": "ring_02_r",
    "pinky_metacarpal_r": "hand_r",
    "pinky_01_r": "pinky_metacarpal_r",
    "pinky_02_r": "pinky_01_r",
    "pinky_03_r": "pinky_02_r"
}

t2m_raw_offsets = np.array([
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
    ], dtype=np.float32) 


src_joint_dict = {
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
    
    # 左手 [23, 25, 26, 27, 28]
    'index_metacarpal_l': 25, 
    'index_01_l': 26, 
    'index_02_l': 27, 
    'index_03_l': 28, 
    
    # [23, 29, 90, 31, 32]
    'middle_metacarpal_l': 29,
    'middle_01_l': 30, 
    'middle_02_l': 31, 
    'middle_03_l': 32, 
    
    # [23, 33, 34, 35, 36]
    'ring_metacarpal_l': 33,
    'ring_01_l': 34,
    'ring_02_l': 35, 
    'ring_03_l': 36, 
    
    # [23, 37, 38, 39, 40]
    'pinky_metacarpal_l': 37, 
    'pinky_01_l': 38, 
    'pinky_02_l': 39, 
    'pinky_03_l': 40, 
    
    # [23, 41, 42, 43]
    'thumb_01_l': 41, 
    'thumb_02_l': 42, 
    'thumb_03_l': 43, 
    
    # 右手 [24, 44, 45, 46, 47]
    'index_metacarpal_r': 44,
    'index_01_r': 45, 
    'index_02_r': 46, 
    'index_03_r': 47, 
    
    # [24, 48, 49, 50, 51]
    'middle_metacarpal_r': 48,
    'middle_01_r': 49, 
    'middle_02_r': 50, 
    'middle_03_r': 51, 
    
    # [24, 52, 53, 54, 55]
    'ring_metacarpal_r': 52,
    'ring_01_r': 53, 
    'ring_02_r': 54,
    'ring_03_r': 55, 
    
    # [24, 56, 57, 58, 59]
    'pinky_metacarpal_r': 56, 
    'pinky_01_r': 57, 
    'pinky_02_r': 58,
    'pinky_03_r': 59, 
    
    # [24, 60, 61, 62]
    'thumb_01_r': 60, 
    'thumb_02_r': 61, 
    'thumb_03_r': 62,
}

src_joint_dict = dict(sorted(src_joint_dict.items(), key=lambda x: x[1]))

SKELETON = [
    "root", "pelvis", "thigh_r", "calf_r", "foot_r", "ball_r", "thigh_l", "calf_l", 
    "foot_l", "ball_l", "spine_01", "spine_02", "spine_03", "spine_04", "spine_05", 
    "neck_01", "neck_02", "head", "clavicle_l", "upperarm_l", "lowerarm_l", "clavicle_r", 
    "upperarm_r", "lowerarm_r", "hand_l", "lowerarm_twist_01_l", "lowerarm_twist_02_l", 
    "index_metacarpal_l", "index_01_l", "index_02_l", "index_03_l", "middle_metacarpal_l", 
    "middle_01_l", "middle_02_l", "middle_03_l", "ring_metacarpal_l", "ring_01_l", "ring_02_l", "ring_03_l",
    "pinky_metacarpal_l", "pinky_01_l", "pinky_02_l", "pinky_03_l", "thumb_01_l", "thumb_02_l", "thumb_03_l", "hand_r",
    "lowerarm_twist_01_r", "lowerarm_twist_02_r", "index_metacarpal_r", "index_01_r", "index_02_r", "index_03_r",
    "middle_metacarpal_r", "middle_01_r", "middle_02_r", "middle_03_r", "ring_metacarpal_r", "ring_01_r", "ring_02_r", "ring_03_r",
    "pinky_metacarpal_r", "pinky_01_r", "pinky_02_r", "pinky_03_r", "thumb_01_r", "thumb_02_r", "thumb_03_r"
]



skel_dict = {joint: i for i, joint in enumerate(SKELETON)}

origin_whole_joints_id = []
for joint in src_joint_dict.keys():
    origin_whole_joints_id.append(skel_dict[joint])
print(origin_whole_joints_id)
print(len(origin_whole_joints_id))


# print(body_joints_id, len(body_joints_id))

# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 45] # 25 joint 
