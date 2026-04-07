#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   schema.py
@Time    :   2025/12/14 17:26:34
@Author  :   Chuhao Jin 
@Contact :   jinchuhao@ruc.edu.cn

@Description:
    数据结构定义模块
    
    本模块定义了 Motion VQ-VAE 系统中使用的核心数据结构，
    包括动作 tokens、面部 tokens、音频特征和动作序列等。
    
    这些数据类使用 Python dataclass 实现，提供了类型安全和
    清晰的数据组织方式，便于在系统各模块之间传递数据。
'''

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import torch 


@dataclass
class MotionTokens:
    """
    动作离散 Token 数据结构
    
    存储 VQ-VAE 编码后的离散动作 tokens，按身体部位分组存储。
    每个部位的 tokens 是一个整数列表，表示该部位在各时间步的
    codebook 索引。
    
    Attributes:
        whole (List[int]): 全身动作 tokens
            - 编码整体动作的协调性和全局特征
            - 长度为 T/downsample_rate，其中 T 为原始帧数
        body (List[int]): 身体动作 tokens
            - 编码躯干和四肢的动作
            - 包含 25 个关节的运动信息
        left_arm (List[int]): 左手动作 tokens
            - 编码左手和手指的精细动作
            - 包含 20 个关节的运动信息
        right_arm (List[int]): 右手动作 tokens
            - 编码右手和手指的精细动作
            - 包含 20 个关节的运动信息
    
    Example:
        >>> tokens = MotionTokens(
        ...     whole=[1, 5, 3, 8, 2],
        ...     body=[10, 20, 15, 30, 25],
        ...     left_arm=[100, 200, 150, 300, 250],
        ...     right_arm=[50, 60, 55, 70, 65]
        ... )
        >>> len(tokens.whole)
        5
    
    Note:
        - 各部位的 tokens 长度应该相同
        - Token 值的范围取决于 codebook 大小（如 2048）
    """
    whole: List[int] = field(default_factory=list)
    body: List[int] = field(default_factory=list)
    left_arm: List[int] = field(default_factory=list)
    right_arm: List[int] = field(default_factory=list)

    def to_sequence(self) -> List[int]:
        """
        将所有部位的 tokens 拼接为单一序列
        
        用于需要统一处理所有 tokens 的场景，如序列化存储或
        作为语言模型的输入。
        
        Returns:
            List[int]: 拼接后的 token 序列
                顺序为: whole + body + left_arm + right_arm
        
        Example:
            >>> tokens = MotionTokens(whole=[1,2], body=[3,4], left_arm=[5,6], right_arm=[7,8])
            >>> tokens.to_sequence()
            [1, 2, 3, 4, 5, 6, 7, 8]
        """
        return self.whole + self.body + self.left_arm + self.right_arm


@dataclass
class FaceTokens:
    """
    面部表情离散 Token 数据结构
    
    存储面部表情 VQ-VAE 编码后的离散 tokens。
    
    Attributes:
        face (List[int]): 面部表情 tokens 列表
            - 编码面部 blendshape 或关键点的运动
            - 长度为 T/downsample_rate
    
    Note:
        - 当前版本主要关注身体动作，面部功能待扩展
    """
    face: List[int] = field(default_factory=list)


@dataclass
class AudioFeatures:
    """
    音频特征数据结构
    
    存储从音频中提取的特征，用于音频驱动的动作生成。
    
    Attributes:
        audio (torch.Tensor): 音频特征张量
            - 通常来自预训练的音频编码器（如 HuBERT、Wav2Vec2）
            - shape 通常为 (T, D)，T 为时间步，D 为特征维度
        audio_9layer (Optional[torch.Tensor]): 第 9 层的音频特征
            - 某些模型需要特定层的特征
            - 用于 motion 生成模型的输入
    
    Note:
        - 音频特征的采样率需要与动作数据对齐
    """
    audio: torch.Tensor
    audio_9layer: Optional[torch.Tensor] = None


@dataclass
class MotionSemantic:
    """
    动作语义标签数据结构
    
    存储动作的高层语义描述，用于条件生成或动作检索。
    
    Attributes:
        expression (str): 表情描述
            - 例如: "调皮", "开心", "严肃"
        body_action (str): 身体动作描述
            - 例如: "玩手指", "挥手", "点头"
        extra (Dict[str, Any]): 额外的语义信息
            - 可扩展字段，存储其他元数据
            - 例如: {"intensity": 0.8, "style": "casual"}
    
    Example:
        >>> semantic = MotionSemantic(
        ...     expression="开心",
        ...     body_action="挥手打招呼",
        ...     extra={"duration": 3.0}
        ... )
    """
    expression: str = ""
    body_action: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MotionSequence:
    """
    完整动作序列数据结构
    
    整合动作生成流程中的所有数据，包括语义标签、离散 tokens、
    连续动作数据和各种输出格式。
    
    Attributes:
        semantic (MotionSemantic): 动作的语义描述
            - 包含表情、动作类型等高层信息
        motion_tokens (MotionTokens): 离散的动作 tokens
            - VQ-VAE 编码后的离散表示
        face_tokens (FaceTokens): 离散的面部 tokens
            - 面部表情的离散表示
        continuous_face (Optional[Any]): 连续的面部表情数据
            - VQ-VAE 解码后的面部 blendshape 值
            - 通常为 np.ndarray，shape 为 (T, num_blendshapes)
        continuous_motion (Optional[Any]): 连续的动作数据
            - VQ-VAE 解码后的关节数据
            - 可以是 np.ndarray 或 torch.Tensor
            - shape 通常为 (T, J, D)
        quat_anim (Optional[Any]): 四元数格式的动画数据
            - 符合 UE 引擎驱动格式的 JSON 结构
            - 包含 offset、body、face 等字段
        bvh_anim (Optional[Any]): BVH 格式的动画数据
            - 标准的 BVH 动画对象
            - 可直接保存为 .bvh 文件
    
    Example:
        >>> sequence = MotionSequence(
        ...     semantic=MotionSemantic(expression="开心", body_action="挥手"),
        ...     motion_tokens=MotionTokens(...),
        ...     face_tokens=FaceTokens(...)
        ... )
    
    Note:
        - 该数据结构贯穿整个动作生成流程
        - 不同阶段会填充不同的字段
    """
    semantic: MotionSemantic
    motion_tokens: MotionTokens
    face_tokens: FaceTokens
    continuous_face: Optional[Any] = None
    continuous_motion: Optional[Any] = None
    quat_anim: Optional[Any] = None
    bvh_anim: Optional[Any] = None
