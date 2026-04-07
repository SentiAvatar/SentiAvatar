#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   default_config.py
@Time    :   2026/01/18 16:00:00
@Author  :   Chuhao Jin 
@Contact :   jinchuhao@ruc.edu.cn

@Description:
    默认配置文件，集中管理所有配置参数
'''

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import os


@dataclass
class DataConfig:
    """数据相关配置"""
    # 数据路径
    data_root: str = ""
    
    # 身体部位配置
    body_parts: List[str] = field(default_factory=lambda: ["body", "left", "right", "positions"])
    
    # 关节数量配置 (63 节点骨架)
    body_joints_num: int = 24  # body 部分关节数
    left_joints_num: int = 20  # 左手关节数
    right_joints_num: int = 20  # 右手关节数
    total_joints_num: int = 63  # 总关节数
    
    # 特征维度配置 (6D 旋转表示)
    # body: offset(3) + body_joints(24) * 6D(6) = 3 + 144 = 147 -> 实际是 153 (包含额外信息)
    # 根据 quat_v2 数据格式:
    # body: offset(3) + body_6d(24*6=144) = 147 (但代码中是153，可能有额外3维)
    body_dim: int = 153  # body 特征维度 (offset + body_6d)
    left_dim: int = 120  # left 特征维度 (20*6)
    right_dim: int = 120  # right 特征维度 (20*6)
    whole_dim: int = 393  # 总特征维度 (153 + 120 + 120)
    
    # 数据加载配置
    window_size: int = 64  # 训练窗口大小
    batch_size: int = 128
    num_workers: int = 4
    
    # 帧率
    fps: int = 20


@dataclass
class ModelConfig:
    """模型相关配置"""
    # VQ-VAE 配置
    nb_code: int = 1024  # 码本大小
    code_dim: int = 512  # 码本维度
    
    # 编码器/解码器配置
    down_t: int = 1  # 时间维度下采样率
    stride_t: int = 2  # 时间维度步长
    width: int = 512  # 网络宽度
    depth: int = 3  # 网络深度
    dilation_growth_rate: int = 3  # 扩张增长率
    vq_act: str = "relu"  # 激活函数
    vq_norm: Optional[str] = None  # 归一化方式
    vq_cnn_depth: int = 3  # VQ-CNN 深度
    
    # RVQ 配置
    num_quantizers: int = 1  # 量化器数量
    shared_codebook: bool = False  # 是否共享码本
    quantize_dropout_prob: float = 0.8  # 量化 dropout 概率
    quantize_dropout_cutoff_index: int = 1
    use_whole_encoder: bool = False
    # EMA 更新
    mu: float = 0.99  # 指数移动平均系数


@dataclass
class TrainConfig:
    """训练相关配置"""
    # 基本训练配置
    max_epoch: int = 30
    lr: float = 2e-4
    weight_decay: float = 0.0
    warm_up_iter: int = 2000
    
    # 学习率调度
    milestones: List[int] = field(default_factory=lambda: [50000, 1000000])
    gamma: float = 0.05
    
    # 损失权重
    commit: float = 0.02  # commitment loss 权重
    loss_vel: float = 50  # 速度损失权重
    weight_rec: float = 5  # 重建损失权重
    recons_loss: str = "l1_smooth"  # 重建损失类型
    
    # 位置损失
    start_positions_epoch: int = 0  # 开始计算位置损失的 epoch
    
    # 特征偏置
    feat_bias: float = 5
    
    # 日志和保存
    log_every: int = 10
    save_latest: int = 500
    save_every_e: int = 2
    eval_every_e: int = 1
    
    # 继续训练
    is_continue: bool = False
    which_epoch: str = "all"


@dataclass
class Config:
    """总配置"""
    # 实验名称
    name: str = "VQVAE_v2"
    dataset_name: str = "quat63nodes_v2"
    
    # 路径配置
    checkpoints_dir: str = "./checkpoints"
    log_dir: str = "./log/vq"
    
    # GPU 配置
    gpu_id: int = 0
    local_rank: int = 0
    
    # 随机种子
    seed: int = 3407
    
    # 子配置
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    
    # 调试模式
    debug: bool = False
    def save_opt(self):
        """将当前配置保存为 opt.txt (扁平化输出)"""
        # 确保保存目录存在
        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root)
            
        file_name = os.path.join(self.save_root, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            
            # 1. 获取所有配置
            config_dict = asdict(self)
            flat_dict = {}
            
            # 2. 展平字典 (Flatten)
            for k, v in config_dict.items():
                if isinstance(v, dict):
                    # 如果是嵌套的配置(data, model, train)，把里面的键值对提取出来
                    for sub_k, sub_v in v.items():
                        flat_dict[str(sub_k)] = sub_v
                else:
                    # 如果是顶层配置，直接放入
                    flat_dict[str(k)] = v
            
            # 3. 排序并写入
            for k, v in sorted(flat_dict.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
                
            opt_file.write('-------------- End ----------------\n')
    def __post_init__(self):
        """初始化后处理"""
        # 计算 unit_length
        self.unit_length = self.model.down_t * 2
        
        # 设置保存路径
        self.save_root = os.path.join(self.checkpoints_dir, self.dataset_name, self.name)
        self.model_dir = os.path.join(self.save_root, 'model')
        self.meta_dir = os.path.join(self.save_root, 'meta')
        self.eval_dir = os.path.join(self.save_root, 'animation')
        self.log_path = os.path.join(self.log_dir, self.dataset_name, self.name)


def get_default_config() -> Config:
    """获取默认配置"""
    return Config()


def create_config_from_args(args) -> Config:
    """从命令行参数创建配置"""
    config = Config(
        name=args.name,
        dataset_name=args.dataset_name,
        checkpoints_dir=args.checkpoints_dir,
        gpu_id=args.gpu_id,
        local_rank=getattr(args, 'local_rank', 0),
        seed=args.seed,
        debug=getattr(args, 'debug', False),
    )
    
    # 数据配置
    config.data.data_root = args.data_root
    config.data.window_size = args.window_size
    config.data.batch_size = args.batch_size
    
    # 模型配置
    config.model.nb_code = args.nb_code
    config.model.code_dim = args.code_dim
    config.model.down_t = args.down_t
    config.model.stride_t = args.stride_t
    config.model.width = args.width
    config.model.depth = args.depth
    config.model.dilation_growth_rate = args.dilation_growth_rate
    config.model.vq_act = args.vq_act
    config.model.vq_norm = args.vq_norm
    config.model.num_quantizers = args.num_quantizers
    config.model.shared_codebook = args.shared_codebook
    config.model.quantize_dropout_prob = args.quantize_dropout_prob
    config.model.use_whole_encoder = args.use_whole_encoder
    
    # 训练配置
    config.train.max_epoch = args.max_epoch
    config.train.lr = args.lr
    config.train.weight_decay = args.weight_decay
    config.train.warm_up_iter = args.warm_up_iter
    config.train.milestones = args.milestones
    config.train.gamma = args.gamma
    config.train.commit = args.commit
    config.train.loss_vel = args.loss_vel
    config.train.loss_offset = args.loss_offset
    config.train.weight_rec = args.weight_rec
    config.train.recons_loss = args.recons_loss
    config.train.start_positions_epoch = args.start_positions_epoch
    config.train.is_continue = args.is_continue
    config.train.which_epoch = args.which_epoch
    config.train.log_every = args.log_every
    config.train.save_latest = args.save_latest
    config.train.save_every_e = args.save_every_e
    config.train.eval_every_e = args.eval_every_e
    config.train.loss_foot = args.loss_foot
    config.train.loss_pos = args.loss_pos
    config.train.loss_slide = args.loss_slide
    # 重新计算派生属性
    config.__post_init__()
    
    config.save_opt()
    return config
