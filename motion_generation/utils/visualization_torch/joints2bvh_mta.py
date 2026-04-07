import torch
import json 
from torch import nn
import numpy as np 
# from params.videocap_params import SKELETON, skel_dict, static_face
from tqdm import tqdm
import time 

from utils.visualization_torch import AnimationStructure
import utils.visualization_torch.Animation as Animation
from utils.visualization_torch.FastInverseKinematics import FastInverseKinematics, FastInverseKinematicsLayered
from utils.visualization_torch.Quaternions import Quaternions
import utils.visualization_torch.BVH_mod as BVH
from utils.visualization_torch.remove_fs import remove_fs

def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def qmul_np(q, r):
    q = torch.from_numpy(q).contiguous().float()
    r = torch.from_numpy(r).contiguous().float()
    return qmul(q, r).numpy()
 
class Joint2BVHConvertor:
    def __init__(self, template_bvh, re_order, re_order_inv, end_points, parents, fid_r, fid_l, SKELETON, skel_dict, static_face, demo_quats):
        ### mta setting.         
        # origin t2m setting.
        self.template = BVH.load(template_bvh, need_quater=True)
        self.re_order = re_order
        self.re_order_inv = re_order_inv
        self.end_points = end_points
        self.parents = parents
        self.fid_r, self.fid_l = fid_r, fid_l
        self.template_offset = self.template.offsets.clone()
        self.demo_quats = torch.tensor(demo_quats)[10:11]
        self.SKELETON = SKELETON
        self.skel_dict = skel_dict
        self.static_face = static_face
        
    def convert(self, positions, iterations=10, device = "cuda:0", silent = True): 
        '''
        Convert the SMPL joint positions to Mocap BVH
        :param positions: (N, 22, 3)
        :param iterations: iterations for optimizing rotations, 10 is usually enough
        :param foot_ik: whether to enfore foot inverse kinematics, removing foot slide issue.
        :return:
        '''
            
        positions = torch.tensor(positions[:, self.re_order]).to(device) # (frame, joint, 3)
        new_anim = self.template.clone().to(device)
        
        new_anim.rotations.qs = self.demo_quats.repeat(positions.shape[0], 1, 1).to(device) # Quaternions.id(positions.shape[:-1]).to(device) # (frame, joint, 4)
        new_anim.rotations.m = torch.zeros(new_anim.rotations.shape + (4, 4), device=new_anim.rotations.qs.device, dtype=new_anim.rotations.qs.dtype)
        
        # new_anim.rotations = Quaternions.id(positions.shape[:-1]).to(device)
        new_anim.positions = new_anim.positions[0:1].repeat(positions.shape[0], 1, 1) # (0 frame, joint, 3)
        new_anim.positions[:, 0] = positions[:, 0] # (frame, root)
        
        
        time_a = time.time()
        ik_solver = FastInverseKinematics(new_anim, positions, iterations=iterations, silent=silent) # JacobianInverseKinematics BasicInverseKinematics BasicJacobianIK
        new_anim = ik_solver()
        # total_iterations = iterations
        # fik = FastInverseKinematicsLayered(new_anim, positions, iterations=total_iterations, silent=False)

        # 1）正常：身体 + 手指分层迭代
        # new_anim = fik(solve_body=True, solve_hands=True)

        # 2）只求身体（不动手指）
        # fik.iterations = total_iterations // 2
        # new_anim = fik(solve_body=True, solve_hands=False)

        # 3）只求手指（比如身体已经定好，只微调手势）
        # new_anim = fik(solve_body=False, solve_hands=True)

        glb = Animation.positions_global(new_anim)[:, self.re_order_inv]

        # ---------------------------
        # 1. 在 Torch 里一次性处理所有四元数
        # ---------------------------
        # (F, J, 4), 在 CPU 上做就好
        quats_t = new_anim.rotations.qs.detach().cpu()  # float32 by default

        # 假设原始顺序是 [w, x, y, z]
        w = quats_t[..., 0]
        x = quats_t[..., 1]
        y = quats_t[..., 2]
        z = quats_t[..., 3]

        # 对所有关节统一做：q = [-q[1], q[2], -q[3], q[0]]
        quats_reordered = torch.stack([-x, y, -z, w], dim=-1)  # (F, J, 4)

        # 索引准备（建议你在 __init__ 里就算好，这里简单写）
        names = new_anim.names
        pelvis_idx = names.index("pelvis")
        head_idx   = names.index("head")

        # head 直接设为单位四元数 [0, 0, 0, 1]
        quats_reordered[:, head_idx] = torch.tensor([0, 0, 0, 1], dtype=quats_reordered.dtype)

        # pelvis 需要再和 diff 相乘：q = qmul(q, diff)
        diff = torch.tensor([0.7071, 0.0, 0.0, 0.7071], dtype=quats_reordered.dtype)
        pelvis_q = quats_reordered[:, pelvis_idx]                  # (F, 4)
        diff_batch = diff.view(1, 4).expand(pelvis_q.size(0), 4)   # (F, 4)
        quats_reordered[:, pelvis_idx] = qmul(pelvis_q, diff_batch)

        # 一次性 round + 转 numpy，后面循环只做取值
        quats_np = quats_reordered.numpy().astype(np.float32)  # (F, J, 4)

        # ---------------------------
        # 2. 位置同样一次性取出来
        # ---------------------------
        poss_root = new_anim.positions[:, 0].detach().cpu().numpy()  # (F, 3)

        # ---------------------------
        # 3. 构造 quat_anim：只做轻量级 Python 循环
        # ---------------------------
        time_b = time.time()
        print("quat infer time:", time_b - time_a)

        quat_anim = {
            "timestamp": 1754016912754,
            "fps": 30,
            "frames": []
        }

        body_len = len(self.SKELETON)
        # 映射动画关节到 skeleton 索引，避免在内层循环反复查
        joint_indices = [self.skel_dict[joint_name] for joint_name in names]

        for frame_idx in range(quats_np.shape[0]):
            pos = poss_root[frame_idx]  # (3,)
            # 注意：不要用 [[...]] * body_len，会共用同一子列表
            body = [[0, 0, 0, 1] for _ in range(body_len)]

            frame = {
                "offset": [pos[0], pos[2], pos[1]],
                "body": body,
                "face": self.static_face
            }

            q_frame = quats_np[frame_idx]  # (J, 4)

            # 这里不再做任何数值运算，只是赋值
            for j, joint_name in enumerate(names):
                joint_index = joint_indices[j]
                q = q_frame[j].tolist()
                q = [round(v, 4) for v in q]
                frame["body"][joint_index] = q
            # ball_l / ball_r 固定写死
            frame["body"][self.skel_dict["ball_l"]] = [0, 0, 0.8509, 0.5253]
            frame["body"][self.skel_dict["ball_r"]] = [0, 0, 0.8509, 0.5253]

            quat_anim["frames"].append(frame)
            
        return new_anim, quat_anim, glb