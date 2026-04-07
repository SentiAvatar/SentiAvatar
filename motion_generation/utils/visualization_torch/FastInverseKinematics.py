#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   FastInverseKinematics.py
@Time    :   2025/10/09 10:06:24
@Author  :   Chuhao Jin 
@Contact :   jinchuhao@ruc.edu.cn
'''
import numpy as np
import scipy.linalg as linalg

from utils.visualization_torch import Animation
from utils.visualization_torch import AnimationStructure
from tqdm import tqdm 

from utils.visualization_torch.Quaternions import Quaternions

import torch
import numpy as np

# local transformation matrices
def transforms_local(anim, ):    
    transforms = anim.rotations.transforms()
    # print("transforms:", transforms.shape)
    transforms[:, :, 0:3, 3] = anim.positions
    transforms[:, :, 3:4, 3] = 1.0
    return transforms

def transforms_blank(anim, device, dtype):
    ts = torch.zeros(anim.shape + (4, 4), device = device, dtype = dtype)
    ts[:, :, 0, 0] = 1.0
    ts[:, :, 1, 1] = 1.0
    ts[:, :, 2, 2] = 1.0
    ts[:, :, 3, 3] = 1.0
    return ts


def transforms_global(anim):
    locals = transforms_local(anim)
    globals = transforms_blank(anim, device = locals.device, dtype = locals.dtype)
    globals[:, 0] = locals[:, 0]
    
    for i in range(1, anim.shape[1]):
        globals[:, i] = torch.matmul(globals[:, anim.parents[i]], locals[:, i])
    return globals


def positions_global(anim):
    # get the last column -- corresponding to the coordinates
    positions = transforms_global(anim)[:, :, :, 3]
    return positions[:, :, :3] / positions[:, :, 3, None]


def update_subtree_globals(globals, locals, parents, children, root_j):
    # children: list-of-lists
    # parents: numpy array or tensor giving parent index
    # root_j: int
    # 修改 globals in-place (torch tensors)
    queue = [root_j]
    head = 0
    while head < len(queue):
        p = queue[head]; head += 1
        for c in children[p]:
            # globals[:, c] = globals[:, parents[c]] @ locals[:, c]
            globals[:, c] = torch.matmul(globals[:, parents[c]], locals[:, c])
            queue.append(c)
    return globals

# helper: 构造单个 joint 的 local transform（4x4），必须根据你的 Quaternions 内部表示来写
# 下面我只放一个 placeholder：你需要把 body 内部的 quaternion->matrix 的细节替换成你们库的实现
def local_transform_for_joint(animation, j):
    # placeholder: 尝试使用 animation.rotations.transforms() 如果可索引单个 joint 比较高效
    # 或者使用一个你们已有的 quaternion->rotation-matrix 的函数（更优）
    # fallback（最安全但最慢）：调用 transforms_local(animation) 并取第 j 个条目
    all_locals = transforms_local(animation)   # 如果这里太慢，可替换为单个 joint 的转换实现
    return all_locals[:, j]  # shape: (batch, 4, 4)


def inner_from_transforms(ts):
    # with torch.cuda.amp.autocast():
    d0, d1, d2 = ts[..., 0, 0], ts[..., 1, 1], ts[..., 2, 2]

    q0 = (d0 + d1 + d2 + 1.0) / 4.0
    q1 = (d0 - d1 - d2 + 1.0) / 4.0
    q2 = (-d0 + d1 - d2 + 1.0) / 4.0
    q3 = (-d0 - d1 + d2 + 1.0) / 4.0

    q0 = torch.sqrt(torch.clamp(q0, min=0))
    q1 = torch.sqrt(torch.clamp(q1, min=0))
    q2 = torch.sqrt(torch.clamp(q2, min=0))
    q3 = torch.sqrt(torch.clamp(q3, min=0))
    
    # 优化后的条件逻辑
    q_stack = torch.stack([q0, q1, q2, q3], dim=-1)
    max_idx = torch.argmax(q_stack, dim=-1)
    
    sign01 = torch.sign(ts[..., 2, 1] - ts[..., 1, 2])
    sign02 = torch.sign(ts[..., 0, 2] - ts[..., 2, 0])
    sign03 = torch.sign(ts[..., 1, 0] - ts[..., 0, 1])
    sign12 = torch.sign(ts[..., 1, 0] + ts[..., 0, 1])
    sign13 = torch.sign(ts[..., 0, 2] + ts[..., 2, 0])
    sign23 = torch.sign(ts[..., 2, 1] + ts[..., 1, 2])
    
    # 向量化调整
    q1 = torch.where(max_idx == 0, q1 * sign01, q1)
    q2 = torch.where(max_idx == 0, q2 * sign02, q2)
    q3 = torch.where(max_idx == 0, q3 * sign03, q3)
    
    q0 = torch.where(max_idx == 1, q0 * sign01, q0)
    q2 = torch.where(max_idx == 1, q2 * sign12, q2)
    q3 = torch.where(max_idx == 1, q3 * sign13, q3)
    
    q0 = torch.where(max_idx == 2, q0 * sign02, q0)
    q1 = torch.where(max_idx == 2, q1 * sign12, q1)
    q3 = torch.where(max_idx == 2, q3 * sign23, q3)
    
    q0 = torch.where(max_idx == 3, q0 * sign03, q0)
    q1 = torch.where(max_idx == 3, q1 * sign13, q1)
    q2 = torch.where(max_idx == 3, q2 * sign23, q2)
    return q0, q1, q2, q3 

def from_transforms(ts):
    # print("cls:", cls.shape, "ts:", ts.shape)
    q0, q1, q2, q3 = inner_from_transforms(ts)
    qs = torch.empty(ts.shape[:-2] + (4,), device=ts.device, dtype=ts.dtype)
    qs[..., 0] = q0
    qs[..., 1] = q1
    qs[..., 2] = q2
    qs[..., 3] = q3

    return Quaternions(qs)

# @torch.compile
# @torch.inference_mode()
def cal_global(c_list, anim_transforms, parents, locals):
    for i in c_list:
        anim_transforms[:, i] = torch.matmul(anim_transforms[:, parents[i]], locals[:, i])
    return anim_transforms

class FastInverseKinematics:

    def __init__(self, animation, positions, iterations=1, silent=True):

        self.animation = animation  # 当前动画数据
        self.positions = positions  # 目标世界坐标 (T, J, 3)
        self.iterations = iterations
        self.silent = silent

        self.body_joints = [
            "root", "pelvis", "spine_01", "spine_02", "spine_03", "spine_04", "spine_05",
            "neck_01", "neck_02", "head", "clavicle_l", "upperarm_l", "lowerarm_l", "hand_l",
            "clavicle_r", "upperarm_r", "lowerarm_r", "hand_r",
            "thigh_l", "calf_l", "foot_l", "ball_l",
            "thigh_r", "calf_r", "foot_r", "ball_r"
        ]
        self.hand_joints = [
            "hand_l", "hand_r",
            "thumb_01_l", "thumb_02_l", "thumb_03_l",
            "index_metacarpal_l", "index_01_l", "index_02_l", "index_03_l",
            "middle_metacarpal_l", "middle_01_l", "middle_02_l", "middle_03_l",
            "ring_metacarpal_l", "ring_01_l", "ring_02_l", "ring_03_l",
            "pinky_metacarpal_l", "pinky_01_l", "pinky_02_l", "pinky_03_l",
            "thumb_01_r", "thumb_02_r", "thumb_03_r",
            "index_metacarpal_r", "index_01_r", "index_02_r", "index_03_r",
            "middle_metacarpal_r", "middle_01_r", "middle_02_r", "middle_03_r",
            "ring_metacarpal_r", "ring_01_r", "ring_02_r", "ring_03_r",
            "pinky_metacarpal_r", "pinky_01_r", "pinky_02_r", "pinky_03_r",
        ]
        self.joints_list = [self.body_joints, self.hand_joints]

        # ==== 预计算 children & descendants（只算一次） ====
        parents = animation.parents
        self.children = AnimationStructure.children_list(parents)  # list[list[int]]
        num_joints = len(parents)

        # 每个关节 -> 所有子孙节点列表（Python list）
        self.descendants = [None] * num_joints
        for j in range(num_joints):
            self.descendants[j] = self._get_all_subchild_list(j, self.children)

    def _get_all_subchild_list(self, joint, children):
        """
        递归获得 joint 的所有子孙节点，只在 __init__ 里调用。
        注意：children[joint] 可能是 np.ndarray，这里统一转成 Python list，
        防止后面 now_child + subset_child 触发 NumPy 的逐元素加法。
        """
        now_child_arr = children[joint]
        if len(now_child_arr) == 0:
            return []

        # 统一转成 Python list
        if hasattr(now_child_arr, "tolist"):
            now_child = now_child_arr.tolist()
        else:
            now_child = list(now_child_arr)

        subset_child = []
        for child in now_child:
            subset_child += self._get_all_subchild_list(child, children)

        # 这里是 list + list，做的是拼接，不是数值加法
        return now_child + subset_child

    def __call__(self):

        device = self.positions.device
        dtype = self.positions.dtype

        parents = self.animation.parents
        num_joints = len(parents)

        # ================== 初始 FK ==================
        locals = self.animation.rotations.transforms()        # (T, J, 4, 4)
        locals[:, :, 0:3, 3] = self.animation.positions       # 局部平移
        locals[:, :, 3:4, 3] = 1.0

        anim_transforms = transforms_blank(
            self.animation, device=locals.device, dtype=locals.dtype
        )  # (T, J, 4, 4)
        anim_transforms[:, 0] = locals[:, 0]

        # 计算每个节点的世界坐标（一次 FK）
        for j in range(1, num_joints):
            anim_transforms[:, j] = torch.matmul(
                anim_transforms[:, parents[j]], locals[:, j]
            )

        # 预先拿出 view，避免循环内部重复切片
        anim_positions = anim_transforms[:, :, :3, 3]  # (T, J, 3) view

        # ================== 迭代求解 IK ==================
        for it in range(self.iterations):

            for j in AnimationStructure.joints(parents):
                name = self.animation.names[j]

                children_j = self.children[j]
                if len(children_j) == 0:
                    continue

                # 直接使用预先缓存好的所有子孙节点列表
                c_list = self.descendants[j]
                if not c_list:
                    c = torch.tensor(children_j, dtype=torch.long, device=device)
                else:
                    # 这里保持和你原先一样：只用直接孩子做方向，
                    # c_list 用于局部重算 FK
                    c = torch.tensor(children_j, dtype=torch.long, device=device)

                # 注意：anim_positions 是 view，随着 anim_transforms 更新会自动变
                # anim_rotations 需要世界旋转（从当前 anim_transforms 提取）
                anim_rotations = from_transforms(anim_transforms[:, j:j+1])  # (T,1,quat)

                # ---------------- 方向向量及归一化 ----------------
                # 当前世界位置到其直接子节点（当前姿态）
                jdirs = anim_positions[:, c] - anim_positions[:, None, j]     # (T, C, 3)
                # 当前世界位置到目标位置（希望的姿态）
                ddirs = self.positions[:, c] - anim_positions[:, None, j]     # (T, C, 3)

                jsums = torch.linalg.norm(jdirs, dim=-1) + 1e-10              # (T, C)
                dsums = torch.linalg.norm(ddirs, dim=-1) + 1e-10              # (T, C)

                jdirs = jdirs / jsums.unsqueeze(-1)
                ddirs = ddirs / dsums.unsqueeze(-1)

                # ---------------- 夹角 ----------------
                dot_product = torch.sum(jdirs * ddirs, dim=2)                  # (T, C)
                clamped_dot = torch.clamp(dot_product, -1.0, 1.0)
                angles = torch.acos(clamped_dot)                               # (T, C)

                # ---------------- 旋转轴 ----------------
                axises = torch.cross(jdirs, ddirs, dim=2)                      # (T, C, 3)
                axises = -anim_rotations * axises                             # 旋转到父局部/世界?

                # ---------------- 聚合旋转 ----------------
                rotations = Quaternions.from_angle_axis(angles, axises)       # (T, C)
                if rotations.shape[1] == 1:
                    averages = rotations[:, 0]                                 # (T,)
                else:
                    # log/exp 平均
                    averages = Quaternions.exp(
                        rotations.log().mean(dim=-2)
                    ).to(device)

                # 更新该关节局部旋转
                self.animation.rotations[:, j] = self.animation.rotations[:, j] * averages

                # clavicle_r 特殊翻转（保持原逻辑）
                if name == "clavicle_r":
                    self.animation.rotations.qs[:, j, 0] = -self.animation.rotations.qs[:, j, 0]
                    self.animation.rotations.qs[:, j, 1] = -self.animation.rotations.qs[:, j, 1]
                    self.animation.rotations.qs[:, j, 3] = -self.animation.rotations.qs[:, j, 3]

                # ---------------- 局部 / 全局重算 ----------------
                # 只对关节 j 相关的局部变换进行更新
                locals = self.animation.rotations.transforms_by_joint(j)      # (T, J, 4, 4)
                locals[:, :, 0:3, 3] = self.animation.positions
                locals[:, :, 3:4, 3] = 1.0

                if j == 0:
                    anim_transforms[:, 0] = locals[:, 0]
                else:
                    anim_transforms[:, j] = torch.matmul(
                        anim_transforms[:, parents[j]], locals[:, j]
                    )

                # 更新 j 的子树世界变换
                if c_list:
                    anim_transforms = cal_global(
                        c_list, anim_transforms, parents, locals
                    )

                # anim_positions 是 view，无需重新赋值

            anim_positions_full = Animation.positions_global(self.animation)
            error = torch.mean(
                torch.sum((anim_positions_full - self.positions) ** 2.0, dim=-1) ** 0.5
            )
            if not self.silent:
                print(f'[FastInverseKinematics] Iteration {it + 1} Loss: {error.item():.6f}')
            if error < 0.05:
                break
        return self.animation
    

class FastInverseKinematicsLayered:

    def __init__(self, animation, positions, iterations=1, silent=True):

        self.animation = animation          # 动画数据
        self.positions = positions          # 目标世界坐标 (T, J, 3)
        self.iterations = iterations        # “大迭代”次数
        self.silent = silent

        # ---------------- 关节分组（名字） ----------------
        self.body_joints = [
            "root", "pelvis",
            "spine_01", "spine_02", "spine_03", "spine_04", "spine_05",
            "neck_01", "neck_02", "head",
            "clavicle_l", "upperarm_l", "lowerarm_l", "hand_l",
            "clavicle_r", "upperarm_r", "lowerarm_r", "hand_r",
            "thigh_l", "calf_l", "foot_l", "ball_l",
            "thigh_r", "calf_r", "foot_r", "ball_r",
        ]

        self.hand_joints = [
            "hand_l", "hand_r",
            "thumb_01_l", "thumb_02_l", "thumb_03_l",
            "index_metacarpal_l", "index_01_l", "index_02_l", "index_03_l",
            "middle_metacarpal_l", "middle_01_l", "middle_02_l", "middle_03_l",
            "ring_metacarpal_l", "ring_01_l", "ring_02_l", "ring_03_l",
            "pinky_metacarpal_l", "pinky_01_l", "pinky_02_l", "pinky_03_l",
            "thumb_01_r", "thumb_02_r", "thumb_03_r",
            "index_metacarpal_r", "index_01_r", "index_02_r", "index_03_r",
            "middle_metacarpal_r", "middle_01_r", "middle_02_r", "middle_03_r",
            "ring_metacarpal_r", "ring_01_r", "ring_02_r", "ring_03_r",
            "pinky_metacarpal_r", "pinky_01_r", "pinky_02_r", "pinky_03_r",
        ]

        # ---------------- name -> index 映射 ----------------
        self.name2idx = {name: i for i, name in enumerate(animation.names)}
        parents = animation.parents
        num_joints = len(parents)

        # 按拓扑顺序的全关节序列
        self.full_joint_order = list(AnimationStructure.joints(parents))

        # 身体 / 手 的索引（保持拓扑顺序）
        self.body_joint_indices = [
            j for j in self.full_joint_order
            if animation.names[j] in self.body_joints
        ]
        self.hand_joint_indices = [
            j for j in self.full_joint_order
            if animation.names[j] in self.hand_joints
        ]

        # ---------------- children & descendants 预计算 ----------------
        # children: 每个关节的直接孩子列表（list[list[int]]）
        self.children = AnimationStructure.children_list(parents)

        # descendants: 每个关节的整棵子树索引（含所有子孙）
        self.descendants = [None] * num_joints
        for j in range(num_joints):
            self.descendants[j] = self._get_all_subchild_list(j, self.children)

        # （可选）缓存 children 的 tensor 形式，避免每次转 tensor
        self.device = positions.device
        self.children_tensor = []
        for c in self.children:
            if len(c) == 0:
                self.children_tensor.append(None)
            else:
                self.children_tensor.append(
                    torch.tensor(c, dtype=torch.long, device=self.device)
                )

    def _get_all_subchild_list(self, joint, children):
        """
        递归获得 joint 的所有子孙节点，只在 __init__ 里调用。
        注意：children[joint] 可能是 np.ndarray，这里统一转成 Python list，
        防止后面 now_child + subset_child 触发 NumPy 的逐元素加法。
        """
        now_child_arr = children[joint]
        if len(now_child_arr) == 0:
            return []

        # 统一转成 Python list
        if hasattr(now_child_arr, "tolist"):
            now_child = now_child_arr.tolist()
        else:
            now_child = list(now_child_arr)

        subset_child = []
        for child in now_child:
            subset_child += self._get_all_subchild_list(child, children)

        # 这里是 list + list，做的是拼接，不是数值加法
        return now_child + subset_child

    # ---------------------- 核心 IK pass（对一组关节做一遍） ----------------------

    def _ik_pass(self, joint_order, locals, anim_transforms, anim_positions):
        """
        对 joint_order 中的关节做一遍 IK 迭代。

        参数：
            joint_order: 要处理的关节索引列表（比如 body 或 hand）
            locals:      当前局部变换 (T, J, 4, 4)
            anim_transforms: 当前全局变换 (T, J, 4, 4)
            anim_positions:  anim_transforms 的 view: (T, J, 3)
        """
        parents = self.animation.parents
        device = self.device

        for j in joint_order:
            name = self.animation.names[j]

            children_j = self.children[j]
            if len(children_j) == 0:
                continue

            # 用预先缓存的 children tensor
            c = self.children_tensor[j]
            if c is None:
                continue

            # 完整子树，用于 cal_global 更新
            c_list = self.descendants[j]

            # 当前世界旋转（从 4x4 变换矩阵取出）
            anim_rotations = from_transforms(anim_transforms[:, j:j+1])  # (T,1,quat)

            # ---------- 方向向量 ----------
            # 关节 j -> 直接孩子（当前姿态）
            jdirs = anim_positions[:, c] - anim_positions[:, None, j]          # (T, C, 3)
            # 关节 j -> 目标位置（目标）
            ddirs = self.positions[:, c] - anim_positions[:, None, j]          # (T, C, 3)

            jsums = torch.linalg.norm(jdirs, dim=-1) + 1e-10                   # (T, C)
            dsums = torch.linalg.norm(ddirs, dim=-1) + 1e-10                   # (T, C)

            jdirs = jdirs / jsums.unsqueeze(-1)
            ddirs = ddirs / dsums.unsqueeze(-1)

            # ---------- 夹角 ----------
            dot_product = torch.sum(jdirs * ddirs, dim=2)                       # (T, C)
            clamped_dot = torch.clamp(dot_product, -1.0, 1.0)
            angles = torch.acos(clamped_dot)                                    # (T, C)

            # ---------- 旋转轴 ----------
            axises = torch.cross(jdirs, ddirs, dim=2)                           # (T, C, 3)
            axises = -anim_rotations * axises                                   # rotate?

            # ---------- 聚合旋转 ----------
            rotations = Quaternions.from_angle_axis(angles, axises)            # (T, C)
            if rotations.shape[1] == 1:
                averages = rotations[:, 0]                                      # (T,)
            else:
                averages = Quaternions.exp(
                    rotations.log().mean(dim=-2)
                ).to(device)                                                   # (T,)

            # 更新该关节局部旋转
            self.animation.rotations[:, j] = self.animation.rotations[:, j] * averages

            # clavicle_r 特殊翻转（保持你原先逻辑）
            if name == "clavicle_r":
                qs = self.animation.rotations.qs
                qs[:, j, 0] = -qs[:, j, 0]
                qs[:, j, 1] = -qs[:, j, 1]
                qs[:, j, 3] = -qs[:, j, 3]

            # ---------- 局部 / 全局重算 ----------
            # 只针对关节 j 和它的子树重算 FK
            locals = self.animation.rotations.transforms_by_joint(j)           # (T, J, 4, 4)
            locals[:, :, 0:3, 3] = self.animation.positions
            locals[:, :, 3:4, 3] = 1.0

            if j == 0:
                anim_transforms[:, 0] = locals[:, 0]
            else:
                anim_transforms[:, j] = torch.matmul(
                    anim_transforms[:, parents[j]], locals[:, j]
                )

            if c_list:
                anim_transforms = cal_global(
                    c_list, anim_transforms, parents, locals
                )

            # anim_positions 是 anim_transforms 的 view，无需重新赋值

        return locals, anim_transforms, anim_positions

    # ---------------------- 对外调用接口 ----------------------

    def __call__(self, solve_body=True, solve_hands=True):
        """
        分层 IK 调用：
            - solve_body=True: 迭代时对身体关节做 IK
            - solve_hands=True: 迭代时对手指关节做 IK
        每一轮 iteration 内部顺序：
            1. 身体 IK pass
            2. 手指 IK pass
        """
        device = self.device
        dtype = self.positions.dtype
        parents = self.animation.parents
        num_joints = len(parents)

        # ---------- 初始 FK ----------
        locals = self.animation.rotations.transforms()                  # (T, J, 4, 4)
        locals[:, :, 0:3, 3] = self.animation.positions
        locals[:, :, 3:4, 3] = 1.0

        anim_transforms = transforms_blank(
            self.animation, device=locals.device, dtype=locals.dtype
        )
        anim_transforms[:, 0] = locals[:, 0]

        for j in range(1, num_joints):
            anim_transforms[:, j] = torch.matmul(
                anim_transforms[:, parents[j]], locals[:, j]
            )

        # anim_positions 是 anim_transforms 的 view
        anim_positions = anim_transforms[:, :, :3, 3]

        # ---------- 分层迭代 ----------
        for it in range(self.iterations):

            # 1. 身体 IK
            if solve_body and self.body_joint_indices:
                locals, anim_transforms, anim_positions = self._ik_pass(
                    self.body_joint_indices, locals, anim_transforms, anim_positions
                )

            # 2. 手指 IK
            if solve_hands and self.hand_joint_indices:
                locals, anim_transforms, anim_positions = self._ik_pass(
                    self.hand_joint_indices, locals, anim_transforms, anim_positions
                )

            if not self.silent:
                anim_positions_full = Animation.positions_global(self.animation)
                error = torch.mean(
                    torch.sum((anim_positions_full - self.positions) ** 2.0, dim=-1) ** 0.5
                )
                print(f'[FastInverseKinematicsLayered] Iteration {it + 1} Loss: {error.item():.6f}')

        return self.animation




def quat_to_matrix(q):
    """
    Convert unit quaternion to rotation matrix.
    q : (...,4)
    return (...,3,3)
    """
    w, x, y, z = q.unbind(-1)
    B = q.shape[:-1]
    R = torch.empty(*B, 3, 3, device=q.device, dtype=q.dtype)

    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R[..., 0, 0] = ww + xx - yy - zz
    R[..., 0, 1] = 2*(xy - wz)
    R[..., 0, 2] = 2*(xz + wy)

    R[..., 1, 0] = 2*(xy + wz)
    R[..., 1, 1] = ww - xx + yy - zz
    R[..., 1, 2] = 2*(yz - wx)

    R[..., 2, 0] = 2*(xz - wy)
    R[..., 2, 1] = 2*(yz + wx)
    R[..., 2, 2] = ww - xx - yy + zz
    return R


def quat_normalize(q, eps=1e-8):
    return q / (q.norm(p=2, dim=-1, keepdim=True) + eps)


def quat_mul(q, r):
    """
    Hamilton product of two quaternions.
    q, r : (...,4)
    """
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)
    return torch.stack((
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2), dim=-1)
    

def forward_kinematics(quat, root_pos, offsets, parents):
    """
    quat      : (F,J,4) local quats   (unit)
    root_pos  : (F,3)   root translation
    offsets   : (J,3)
    parents   : (J,)
    return global joint positions (F,J,3)
    """
    F, J = quat.shape[:2]
    device = quat.device

    # global rotations & translations
    g_rot = torch.zeros(F, J, 3, 3, device=device)
    g_pos = torch.zeros(F, J, 3, device=device)

    for j in range(J):
        rot_j = quat_to_matrix(quat[:, j])           # (F,3,3)
        if parents[j] == -1:
            g_rot[:, j] = rot_j
            g_pos[:, j] = root_pos                   # + 0
        else:
            p = parents[j]
            g_rot[:, j] = g_rot[:, p].bmm(rot_j)
            g_pos[:, j] = g_pos[:, p] + g_rot[:, p].matmul(offsets[j])

    # 最后再把本地 offset 乘当前关节旋转加回来
    g_pos += torch.einsum('fjk,kb->fjb', g_rot, offsets)

    return g_pos