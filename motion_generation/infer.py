#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   infer.py
@Time    :   2026/02/05 12:00:00
@Description:
    Motion VQ-VAE 推理脚本 (简化版)
    
    使用训练好的 RVQVAE 模型进行动作序列的编码和解码：
    1. 加载预训练的 RVQVAE 模型
    2. 读取输入的动作特征数据（.npy 格式）
    3. 编码 -> 解码 -> 后处理 -> 输出 JSON 和 BVH
'''

import os
import json
import random
import re
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional

from models.rvqvae import RVQVAE
from configs.default_config import Config, DataConfig, ModelConfig, TrainConfig
from actions.postprocess import MotionPostprocesser
from actions.schema import MotionTokens
from utils.rotation_utils import sixd_to_quaternion
from utils.constants import BODY_JOINTS_ID, LEFT_HAND_JOINTS_ID, RIGHT_HAND_JOINTS_ID


# ==================== 平滑和重采样工具函数 ====================

def _gaussian_kernel1d(kernel_size: int, sigma: float, device, dtype):
    """生成一维高斯核"""
    assert kernel_size % 2 == 1, "kernel_size 必须是奇数"
    half = kernel_size // 2
    x = torch.arange(-half, half + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / (k.sum() + 1e-12)
    return k


def smooth_motion_gaussian(x: torch.Tensor, kernel_size: int = 7, sigma: float = 2.0, pad_mode: str = "reflect"):
    """对动作序列在时间维度上进行高斯平滑"""
    if x.dim() != 4:
        raise ValueError(f"期望 x 为 4 维 (B,T,J,D)，但得到 {x.shape}")

    B, T, J, D = x.shape
    device, dtype = x.device, x.dtype

    y = x.permute(0, 2, 3, 1).contiguous().view(B, J * D, T)
    k = _gaussian_kernel1d(kernel_size, sigma, device, dtype)
    weight = k.view(1, 1, kernel_size).repeat(J * D, 1, 1)

    pad = kernel_size // 2
    y_pad = F.pad(y, (pad, pad), mode=pad_mode)
    y_smooth = F.conv1d(y_pad, weight=weight, bias=None, stride=1, padding=0, groups=J * D)

    out = y_smooth.view(B, J, D, T).permute(0, 3, 1, 2).contiguous()
    return out


def resample_fps(x: torch.Tensor, src_fps: float = 20.0, tgt_fps: float = 30.0, mode: str = "linear", align_corners: bool = False):
    """对动作序列进行帧率重采样"""
    if x.dim() != 4:
        raise ValueError(f"期望 x 为 4 维 (B,T,J,D)，但得到 {x.shape}")

    B, T, J, D = x.shape
    new_T = max(1, int(round(T * (tgt_fps / src_fps))))

    y = x.permute(0, 2, 3, 1).contiguous().view(B, J * D, T)
    y2 = F.interpolate(y, size=new_T, mode=mode, align_corners=align_corners if mode in ("linear", "bilinear", "bicubic", "trilinear") else None)

    out = y2.view(B, J, D, new_T).permute(0, 3, 1, 2).contiguous()
    return out


def smooth_then_resample(x: torch.Tensor, src_fps=20.0, tgt_fps=30.0, kernel_size=7, sigma=2.0):
    """先平滑后重采样"""
    x = smooth_motion_gaussian(x, kernel_size=kernel_size, sigma=sigma)
    x = resample_fps(x, src_fps=src_fps, tgt_fps=tgt_fps)
    return x


def fixseed(seed: int):
    """设置随机种子"""
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def find_config_path_from_checkpoint(checkpoint_path: str) -> str:
    """
    根据 checkpoint 路径找到对应的 config 文件路径
    
    例如:
    checkpoint: /disk0/home/gaoqz/gqzwork/aTrain_VQVAE/checkpoints/quat63nodes_v2_0120/gqzV4/model/latest.pth
    config: /disk0/home/gaoqz/gqzwork/aTrain_VQVAE/checkpoints/quat63nodes_v2_0120/gqzV4/opt.txt
    
    Args:
        checkpoint_path: checkpoint 文件的完整路径
        
    Returns:
        config 文件的完整路径
    """
    # 获取 checkpoint 所在目录 (model 目录)
    model_dir = os.path.dirname(checkpoint_path)
    # 获取 model 目录的父目录 (实验目录)
    experiment_dir = os.path.dirname(model_dir)
    # config 文件在实验目录下
    config_path = os.path.join(experiment_dir, "opt.txt")
    
    return config_path


def parse_opt_txt(opt_path: str) -> Dict:
    """
    解析 opt.txt 文件，返回配置字典
    
    opt.txt 格式:
    ------------ Options -------------
    batch_size: 256
    body_dim: 153
    ...
    -------------- End ----------------
    
    Args:
        opt_path: opt.txt 文件路径
        
    Returns:
        配置字典
    """
    config_dict = {}
    
    with open(opt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        # 跳过分隔线
        if line.startswith('---') or not line:
            continue
        
        # 解析 key: value 格式
        if ':' in line:
            # 只在第一个冒号处分割，因为值可能包含冒号（如路径）
            idx = line.index(':')
            key = line[:idx].strip()
            value_str = line[idx+1:].strip()
            
            # 尝试解析值的类型
            value = parse_value(value_str)
            config_dict[key] = value
    
    return config_dict


def parse_value(value_str: str):
    """
    解析配置值字符串，转换为适当的 Python 类型
    
    Args:
        value_str: 值的字符串表示
        
    Returns:
        解析后的值
    """
    value_str = value_str.strip()
    
    # 布尔值
    if value_str == 'True':
        return True
    if value_str == 'False':
        return False
    
    # None
    if value_str == 'None':
        return None
    
    # 列表 (如 ['body', 'left', 'right', 'positions'] 或 [50000, 1000000])
    if value_str.startswith('[') and value_str.endswith(']'):
        try:
            # 使用 eval 解析列表（注意：仅用于可信的配置文件）
            return eval(value_str)
        except:
            return value_str
    
    # 整数
    try:
        return int(value_str)
    except ValueError:
        pass
    
    # 浮点数
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # 字符串
    return value_str


def load_config_from_checkpoint(checkpoint_path: str) -> Config:
    """
    根据 checkpoint 路径加载对应的配置
    
    Args:
        checkpoint_path: checkpoint 文件路径
        
    Returns:
        Config 对象
    """
    # 找到 config 文件路径
    config_path = find_config_path_from_checkpoint(checkpoint_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"Loading config from {config_path}...")
    
    # 解析 opt.txt
    opt_dict = parse_opt_txt(config_path)
    
    # 创建 Config 对象
    config = Config(
        name=opt_dict.get('name', 'VQVAE_v2'),
        dataset_name=opt_dict.get('dataset_name', 'quat63nodes_v2'),
        checkpoints_dir=opt_dict.get('checkpoints_dir', './checkpoints'),
        log_dir=opt_dict.get('log_dir', './log/vq'),
        gpu_id=opt_dict.get('gpu_id', 0),
        local_rank=opt_dict.get('local_rank', 0),
        seed=opt_dict.get('seed', 3407),
        debug=opt_dict.get('debug', False),
    )
    
    # 数据配置
    config.data.data_root = opt_dict.get('data_root', '')
    config.data.body_parts = opt_dict.get('body_parts', ['body', 'left', 'right', 'positions'])
    config.data.body_joints_num = opt_dict.get('body_joints_num', 24)
    config.data.left_joints_num = opt_dict.get('left_joints_num', 20)
    config.data.right_joints_num = opt_dict.get('right_joints_num', 20)
    config.data.total_joints_num = opt_dict.get('total_joints_num', 63)
    config.data.body_dim = opt_dict.get('body_dim', 153)
    config.data.left_dim = opt_dict.get('left_dim', 120)
    config.data.right_dim = opt_dict.get('right_dim', 120)
    config.data.whole_dim = opt_dict.get('whole_dim', 393)
    config.data.window_size = opt_dict.get('window_size', 64)
    config.data.batch_size = opt_dict.get('batch_size', 128)
    config.data.num_workers = opt_dict.get('num_workers', 4)
    config.data.fps = opt_dict.get('fps', 20)
    
    # 模型配置
    config.model.nb_code = opt_dict.get('nb_code', 512)
    config.model.code_dim = opt_dict.get('code_dim', 512)
    config.model.down_t = opt_dict.get('down_t', 1)
    config.model.stride_t = opt_dict.get('stride_t', 2)
    config.model.width = opt_dict.get('width', 512)
    config.model.depth = opt_dict.get('depth', 3)
    config.model.dilation_growth_rate = opt_dict.get('dilation_growth_rate', 3)
    config.model.vq_act = opt_dict.get('vq_act', 'relu')
    config.model.vq_norm = opt_dict.get('vq_norm', None)
    config.model.vq_cnn_depth = opt_dict.get('vq_cnn_depth', 3)
    config.model.num_quantizers = opt_dict.get('num_quantizers', 4)
    config.model.shared_codebook = opt_dict.get('shared_codebook', False)
    config.model.quantize_dropout_prob = opt_dict.get('quantize_dropout_prob', 0.8)
    config.model.quantize_dropout_cutoff_index = opt_dict.get('quantize_dropout_cutoff_index', 1)
    config.model.use_whole_encoder = opt_dict.get('use_whole_encoder', False)
    config.model.mu = opt_dict.get('mu', 0.99)
    
    # 训练配置
    config.train.max_epoch = opt_dict.get('max_epoch', 100)
    config.train.lr = opt_dict.get('lr', 0.0001)
    config.train.weight_decay = opt_dict.get('weight_decay', 0.0)
    config.train.warm_up_iter = opt_dict.get('warm_up_iter', 2000)
    config.train.milestones = opt_dict.get('milestones', [50000, 1000000])
    config.train.gamma = opt_dict.get('gamma', 0.05)
    config.train.commit = opt_dict.get('commit', 0.02)
    config.train.loss_vel = opt_dict.get('loss_vel', 50.0)
    config.train.weight_rec = opt_dict.get('weight_rec', 5.0)
    config.train.recons_loss = opt_dict.get('recons_loss', 'l1_smooth')
    config.train.start_positions_epoch = opt_dict.get('start_positions_epoch', 0)
    config.train.feat_bias = opt_dict.get('feat_bias', 5)
    config.train.log_every = opt_dict.get('log_every', 10)
    config.train.save_latest = opt_dict.get('save_latest', 500)
    config.train.save_every_e = opt_dict.get('save_every_e', 2)
    config.train.eval_every_e = opt_dict.get('eval_every_e', 1)
    config.train.is_continue = opt_dict.get('is_continue', False)
    config.train.which_epoch = opt_dict.get('which_epoch', 'all')
    
    # 重新计算派生属性
    config.unit_length = config.model.down_t * 2
    config.save_root = os.path.join(config.checkpoints_dir, config.dataset_name, config.name)
    config.model_dir = os.path.join(config.save_root, 'model')
    config.meta_dir = os.path.join(config.save_root, 'meta')
    config.eval_dir = os.path.join(config.save_root, 'animation')
    config.log_path = os.path.join(config.log_dir, config.dataset_name, config.name)
    
    print(f"Config loaded: name={config.name}, dataset={config.dataset_name}")
    print(f"  Model: nb_code={config.model.nb_code}, num_quantizers={config.model.num_quantizers}, use_whole_encoder={config.model.use_whole_encoder}")
    print(f"  Data: whole_dim={config.data.whole_dim}, body_dim={config.data.body_dim}, fps={config.data.fps}")
    
    return config


def load_model(checkpoint_path: str, config: Config, device: torch.device) -> RVQVAE:
    """加载 RVQVAE 模型"""
    print(f"Loading model from {checkpoint_path}...")
    
    model = RVQVAE(
        config=config,
        input_dim=config.data.whole_dim,
        nb_code=config.model.nb_code,
        code_dim=config.model.code_dim,
        output_dim=config.model.code_dim,
        down_t=config.model.down_t,
        stride_t=config.model.stride_t,
        width=config.model.width,
        depth=config.model.depth,
        dilation_growth_rate=config.model.dilation_growth_rate,
        activation=config.model.vq_act,
        norm=config.model.vq_norm,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}, step {checkpoint.get('global_step', 'unknown')}")
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    return model


def encode_motion(model: RVQVAE, motion_dict: Dict[str, np.ndarray], device: torch.device) -> MotionTokens:
    
    _module_dir = os.path.dirname(os.path.abspath(__file__))
    mean = np.load(os.path.join(_module_dir, "meta/mta_gen_demo/mean.npy")) 
    std = np.load(os.path.join(_module_dir, "meta/mta_gen_demo/std.npy")) 
        
    mean = torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)


    """编码动作为 tokens"""
    body_motion = torch.tensor(motion_dict["body"], dtype=torch.float32).to(device)
    left_motion = torch.tensor(motion_dict["left"], dtype=torch.float32).unsqueeze(0).to(device)
    right_motion = torch.tensor(motion_dict["right"], dtype=torch.float32).unsqueeze(0).to(device)
    
    body_motion[:, 2] = body_motion[:, 2] - body_motion[0, 2]
    body_motion[1:, :3] = body_motion[1:, :3] - body_motion[:-1, :3]
    body_motion = (body_motion - mean) / std 
    body_motion = body_motion.unsqueeze(0)
    print("body_motion:", body_motion.shape) 
    with torch.no_grad():
        output = model.encode(body_motion)
    
    # code_idx shape: (b, n, q) 其中 q 是 num_quantizers
    motion_tokens = MotionTokens(
        body=output["code_idx"]["body"].squeeze(0).cpu().numpy().tolist(),
        # left_arm=output["code_idx"]["left"].squeeze(0).cpu().numpy().tolist(),
        # right_arm=output["code_idx"]["right"].squeeze(0).cpu().numpy().tolist(),
    )
    print("motion_tokens:", motion_tokens)
    # MotionTokens(whole=[], body=[[283, 411, 167, 41], [317, 34, 467, 19], [377, 64, 167, 493], [13, 64, 0, 265], [236, 160, 246, 404], [253, 492, 477, 454], [377, 198, 100, 404], [13, 34, 28, 506], [13, 198, 139, 319], [237, 198, 100, 311], [359, 198, 400, 190], [333, 278, 493, 259], [38, 377, 310, 41], [35, 387, 294, 29], [169, 17, 117, 269], [372, 384, 300, 90], [65, 393, 362, 252], [449, 227, 340, 57], [85, 474, 276, 384], [110, 311, 108, 308], [251, 358, 106, 150], [129, 308, 261, 173], [184, 313, 389, 9], [184, 163, 389, 9], [284, 405, 161, 11], [344, 64, 87, 11], [344, 506, 63, 163], [185, 501, 63, 239], [41, 64, 329, 228], [243, 64, 389, 463], [41, 64, 172, 282], [471, 64, 191, 499], [21, 118, 226, 81], [477, 500, 97, 229], [421, 64, 226, 75], [453, 362, 63, 75], [294, 362, 63, 504], [476, 362, 343, 504], [489, 362, 63, 78], [332, 481, 472, 336], [375, 481, 335, 305], [19, 34, 63, 188], [375, 64, 178, 358], [332, 511, 63, 197], [294, 294, 9, 509], [476, 426, 145, 509], [303, 426, 145, 509], [217, 216, 145, 509], [96, 64, 275, 336], [433, 34, 63, 449], [268, 362, 94, 449], [178, 232, 472, 449], [393, 232, 472, 210], [483, 369, 117, 358], [374, 29, 429, 449], [19, 29, 429, 449], [374, 232, 472, 298], [19, 481, 472, 200], [374, 29, 472, 200], [393, 34, 286, 163]], left_arm=[], right_arm=[])
    return motion_tokens

def decode_tokens(model: RVQVAE, motion_tokens: MotionTokens, motion_dict: Dict[str, np.ndarray], config: Config, device: torch.device,
                  src_fps: float = 20.0, tgt_fps: float = 30.0) -> Dict[str, np.ndarray]:
    """解码 tokens 为动作数据"""
    _module_dir = os.path.dirname(os.path.abspath(__file__))
    mean = np.load(os.path.join(_module_dir, "meta/mta_gen_demo/mean.npy")) 
    std = np.load(os.path.join(_module_dir, "meta/mta_gen_demo/std.npy")) 
        
    mean = torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)

    with torch.no_grad():
        # 将 tokens 转换为模型输入格式
        # 需要 shape: (b, n, q)
        body_tokens = torch.tensor(np.array(motion_tokens.body), dtype=torch.long).unsqueeze(0).to(device)
        # left_tokens = torch.tensor(np.array(motion_tokens.left_arm), dtype=torch.long).unsqueeze(0).to(device)
        # right_tokens = torch.tensor(np.array(motion_tokens.right_arm), dtype=torch.long).unsqueeze(0).to(device)
        
        # 如果 tokens 是 1D 的 (n,)，需要添加 quantizer 维度
        # if body_tokens.dim() == 2:
        #     body_tokens = body_tokens.unsqueeze(-1)  # (b, n) -> (b, n, 1)
            # left_tokens = left_tokens.unsqueeze(-1)
            # right_tokens = right_tokens.unsqueeze(-1)
        
        code_idx_dict = {
            "body": body_tokens,
            # "left": left_tokens,
            # "right": right_tokens,
        }
        print("body_tokens:", body_tokens.shape)
        left_motion = torch.tensor(motion_dict["left"], dtype=torch.float32).unsqueeze(0).to(device)
        right_motion = torch.tensor(motion_dict["right"], dtype=torch.float32).unsqueeze(0).to(device)
        body_motion = torch.tensor(motion_dict["body"], dtype=torch.float32).to(device)
        
        # 解码
        x_out = model.forward_decoder(code_idx_dict)  # (B, T, D)
        pred_whole_motion = x_out.sum(0)  # 取第一个 batch
        
        body_motion[:, 2] = body_motion[:, 2] - body_motion[0, 2]
        body_motion[1:, :3] = body_motion[1:, :3] - body_motion[:-1, :3]
        body_motion = (body_motion - mean) / std 
        # body_motion = body_motion.unsqueeze(0)
        pred_whole_motion = body_motion
        
        pred_whole_motion = pred_whole_motion * std + mean 
        # pred_whole_motion = body_motion.squeeze(0)
        # 分离不同部位的特征
        body_dim = config.data.body_dim  # 153
                
        print("pred_whole_motion:", pred_whole_motion.shape) 
        
        pred_body_motion = pred_whole_motion[:, :body_dim]
        
        print("left_motion:", left_motion.shape)
        print("right_motion:", right_motion.shape)

        frames = pred_body_motion.shape[0]
        
        # 分离 offset 和 6D 旋转
        offset_frame0 = torch.tensor([0.0, 0.0, 102.0]).to(device)
        pred_offset_vel = pred_body_motion[:, :3]  # 根节点速度，需要累加起来转为位移
        
        for i in range(1, pred_offset_vel.shape[0]):
            pred_offset_vel[i] = pred_offset_vel[i] + pred_offset_vel[i - 1]
        pred_offset = pred_offset_vel + offset_frame0
        
        pred_body_6d = pred_body_motion[:, 3:]  # 身体关节的 6D 旋转
        
        # body_motion = body_motion[:, :, 3:]
        # 重塑为 (B, T, J, 6) 格式
        print("body_motion:", body_motion.shape)
        
        pred_body_6d = pred_body_6d.reshape(1, frames, 25, 6)
        pred_left_motion = left_motion[:, :frames].reshape(1, frames, 20, 6)
        pred_right_motion = right_motion[:, :frames].reshape(1, frames, 20, 6)
        
        print("pred_left_motion:", pred_left_motion.shape)
        print("pred_right_motion:", pred_right_motion.shape)
        # 平滑并重采样
        pred_body_6d = smooth_then_resample(pred_body_6d, src_fps=src_fps, tgt_fps=tgt_fps)
        pred_left_motion = smooth_then_resample(pred_left_motion, src_fps=src_fps, tgt_fps=tgt_fps)
        pred_right_motion = smooth_then_resample(pred_right_motion, src_fps=src_fps, tgt_fps=tgt_fps)
        
        
        # offset 也需要重采样
        pred_offset = pred_offset.reshape(1, frames, 1, 3)
        pred_offset = smooth_then_resample(pred_offset, src_fps=src_fps, tgt_fps=tgt_fps)
        new_frames = pred_right_motion.shape[1]
        
        # 将 6D 旋转转换为四元数
        pred_body_quat = sixd_to_quaternion(pred_body_6d.reshape(-1, 6)).reshape(1, new_frames, 25, 4)
        pred_left_quat = sixd_to_quaternion(pred_left_motion.reshape(-1, 6)).reshape(1, new_frames, 20, 4)
        pred_right_quat = sixd_to_quaternion(pred_right_motion.reshape(-1, 6)).reshape(1, new_frames, 20, 4)
        
        # 处理 offset
        pred_offset = pred_offset.reshape(new_frames, 3)
        # pred_offset[:, 2] = pred_offset[:, 2] + 1
        # pred_offset = pred_offset * 100
        pred_offset = pred_offset.detach().cpu().numpy()
        
        # 合并所有关节的四元数
        # BODY_JOINTS_ID: [0, 1, ..., 24] (25 个)
        # LEFT_HAND_JOINTS_ID: [23, 25, 26, ..., 43] (20 个，第一个是 hand_l)
        # RIGHT_HAND_JOINTS_ID: [24, 44, 45, ..., 62] (20 个，第一个是 hand_r)
        merge_pred_quat = torch.zeros(new_frames, 63, 4).to(device)
        
        # 身体关节
        merge_pred_quat[:, BODY_JOINTS_ID] = pred_body_quat[0]
        # 左手关节（跳过手腕 hand_l，因为已经在 body 中）
        merge_pred_quat[:, LEFT_HAND_JOINTS_ID[1:]] = pred_left_quat[0, :, 1:]
        # 右手关节（跳过手腕 hand_r，因为已经在 body 中）
        merge_pred_quat[:, RIGHT_HAND_JOINTS_ID[1:]] = pred_right_quat[0, :, 1:]
        
        pred_quat = merge_pred_quat.detach().cpu().numpy()
        
        motion = {"offset": pred_offset, "quat": pred_quat}
        return motion


def main():
    """主函数"""
    
    # ==================== 配置参数（在这里修改） ====================
    
    # 模型检查点路径
    checkpoint_path = "/data/home/jinch/tech_report/susu_avatar_training_gen_demo/VQ_V0205/checkpoints/quat63nodes_v2_0120/gqzV4/model/epoch_30.pth"
    # 输入 npy 文件路径
    # input_npy_path = "/disk1/chuhao/dataset/mocap/mocap_susu_gen_demo/quat63nodes_v4_fix_pos/joint_quat_vecs/fbx_to_json_data_susu_retarget_maya/20251011/Human_0916_180_2_6_01.npy"
    # input_npy_path = "/disk1/chuhao/dataset/mocap/mocap_susu_gen_demo/quat63nodes_v4_fix_pos/joint_quat_vecs/fbx_to_json_data_susu_retarget_maya/20251104/Human_0916_162_3_7_02_XG.npy"
    # input_npy_path = "/disk1/chuhao/dataset/mocap/mocap_susu_gen_demo/quat63nodes_v4_fix_pos/joint_quat_vecs/fbx_to_json_data_susu_retarget_maya/20251027/Human_0916_112_4_6_03_XC.npy"
    # input_npy_path = "/disk1/chuhao/dataset/mocap/mocap_susu_gen_demo/quat63nodes_v4_fix_pos/joint_quat_vecs/fbx_to_json_data_susu_retarget_maya/20251125/Human_1119_86_1_5_02_XG.npy"
    # input_npy_path = "/disk1/chuhao/dataset/mocap/mocap_susu_gen_demo/quat63nodes_v4_fix_pos/joint_quat_vecs/fbx_to_json_data_susu_retarget_maya/20251031/Human_1021_95_0_2_01_XC.npy"
    # input_npy_path = "/disk1/chuhao/dataset/mocap/mocap_susu_gen_demo/quat63nodes_v4_fix_pos/joint_quat_vecs/fbx_to_json_data_susu_retarget_maya/20251112/Human_1021_300_2_6_01_XG.npy"
    # input_npy_path = "/disk1/chuhao/dataset/mocap/mocap_susu_gen_demo/quat63nodes_v4_fix_pos/joint_quat_vecs/fbx_to_json_data_susu_retarget_maya/20250915/Human_0912_145-6_01.npy"
    # input_npy_path = "/disk1/chuhao/dataset/mocap/mocap_susu_gen_demo/quat63nodes_v4_fix_pos/joint_quat_vecs/fbx_to_json_data_susu_chonglu/20260115/Human_155_229_01_A.npy"
    # input_npy_path = "/disk1/chuhao/dataset/mocap/mocap_susu_gen_demo/quat63nodes_v4_fix_pos/joint_quat_vecs/fbx_to_json_data_susu_retarget_maya/20251107/Human_0916_188_3_4_01_XC.npy"
    # input_npy_path = "/disk1/chuhao/dataset/mocap/mocap_susu_gen_demo/quat63nodes_v4_fix_pos/joint_quat_vecs/fbx_to_json_data_susu_retarget_maya/20250903/Human_0901_203-4_01.npy"
    # input_npy_path = "/disk1/chuhao/dataset/mocap/mocap_susu_gen_demo/quat63nodes_v4_fix_pos/joint_quat_vecs/fbx_to_json_data_susu_retarget_maya/20250903/Human_0901_208-4_01.npy"
    # input_npy_path = "/disk0/home/chuhao/susu_avatar_training_clean_mix_version/dataset/quat63nodes_v2/joint_quat_vecs/fbx_to_json_data_susu_chonglu/260114/Human_THINKING1_02_A.npy"
    
    input_npy_path = "/data/public/chuhao_share/susu_avatar_training_gen_demo/merged_processed_all_data/motion_data/fbx_to_json_data_susu_retarget_maya/20250821/Human_0819_100-8_01.npy"
    
    # input_npy_path = "/disk1/chuhao/dataset/mocap/mocap_susu_gen_demo/hunyuan_distill/joint_quat_vecs/00000160_000.npy"
    # 输出目录和文件名
    output_dir = "./debug"
    
    output_name = input_npy_path.split("/")[-1].split(".")[0] + "_pred"
    
    # 设备
    device_str = "cuda:0"
    
    # 帧率配置（如果为 None，则从 config 中读取 src_fps）
    src_fps = 20.0  # 模型输出帧率，None 表示从 config 读取
    tgt_fps = 30.0  # 目标输出帧率
    
    # 随机种子（如果为 None，则从 config 中读取）
    seed = None
    
    # ==================== 执行推理 ====================
    
    # 根据 checkpoint 路径自动加载配置
    config = load_config_from_checkpoint(checkpoint_path)
    
    # 使用 config 中的值作为默认值
    if seed is None:
        seed = config.seed
    if src_fps is None:
        src_fps = float(config.data.fps)
    
    fixseed(seed)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = load_model(checkpoint_path, config, device)
    
    # 加载输入数据
    print(f"Loading input data from {input_npy_path}...")
    motion_dict = np.load(input_npy_path, allow_pickle=True)
    if isinstance(motion_dict, np.ndarray) and motion_dict.dtype == object:
        motion_dict = motion_dict.item()
    
    # 编码
    print("Encoding motion to tokens...")
    motion_tokens = encode_motion(model, motion_dict, device)
    print(f"Motion tokens: body={len(motion_tokens.body)}, left={len(motion_tokens.left_arm)}, right={len(motion_tokens.right_arm)}")
    
    # 解码
    print("Decoding tokens to motion...")
    motion = decode_tokens(model, motion_tokens, motion_dict, config, device, src_fps=src_fps, tgt_fps=tgt_fps)
    print(f"Decoded motion: offset shape={motion['offset'].shape}, quat shape={motion['quat'].shape}")
    
    # 后处理和保存
    os.makedirs(output_dir, exist_ok=True)
    postprocesser = MotionPostprocesser()
    
    # 保存 JSON
    print("Converting to UE format...")
    quat_anim = postprocesser.convert_quat_motion_to_ue_from_bvh(motion=motion)
    json_path = os.path.join(output_dir, f"{output_name}.json")
    with open(json_path, "w") as f:
        json.dump(quat_anim, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON to {json_path}")
    
    # 保存 BVH
    print("Saving BVH...")
    bvh_path = os.path.join(output_dir, f"{output_name}.bvh")
    postprocesser.save_quat_motion_to_bvh(motion=motion, save_path=bvh_path)
    print(f"Saved BVH to {bvh_path}")
    
    print("\nInference completed!")


if __name__ == "__main__":
    main()
