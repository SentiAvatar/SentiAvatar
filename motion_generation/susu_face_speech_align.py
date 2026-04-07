import os, io
import torch
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, HubertModel
import argparse
import importlib
import soundfile as sf
from scipy.interpolate import CubicSpline, interp1d
from face_model_vq import Af2FaceVQVAEConvZeroStrideV3
import logging

logger = logging.getLogger("susu_face_speech_align")

client_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("[susu_face_speech_align] 使用设备: %s", client_device)

def resample_motion(data, original_fps, target_fps=20):
    """
    data: 输入序列，shape=(frames, keypoints, 3)
    original_fps: 原始帧率（如24, 30）
    target_fps: 目标帧率（默认20）
    """
    frames, keypoints, dim = data.shape
    t_original = np.arange(frames) / original_fps  # 原始时间轴
    max_time = t_original[-1]  # 总时长
    # target_frames = int(max_time * target_fps)  # 目标帧数
    target_frames = int(len(t_original) * target_fps // original_fps) # 新目标帧数
    t_target = np.linspace(0, max_time, target_frames)  # 目标时间轴
    resampled_data = np.zeros((target_frames, keypoints, dim))
    
    # 对每个关键点和坐标维度独立插值
    for k in range(keypoints):
        for d in range(dim):  # 处理X/Y/Z三个维度
            # 提取原始数据（一维时间序列）
            y_original = data[:, k, d]
            # 创建插值函数（线性）
            f_interp = interp1d(
                t_original, y_original, 
                kind='cubic', # linear or cubic
                bounds_error=False, 
                fill_value="extrapolate"
            )
            # 生成新数据
            resampled_data[:, k, d] = f_interp(t_target)
    
    return resampled_data
# load chinese hubert
def load_chinese_hubert():
    _module_dir = os.path.dirname(os.path.abspath(__file__))
    chinese_hubert_model_path = os.path.join(_module_dir, "..", "checkpoints", "chinese-hubert-base")
    logger.info("[load_chinese_hubert] 开始加载Chinese Hubert模型, path=%s", chinese_hubert_model_path)
    
    ###HF transformers
    audio_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(chinese_hubert_model_path)
    audio_encoder = HubertModel.from_pretrained(chinese_hubert_model_path)
    audio_encoder = audio_encoder.to(client_device)
    audio_encoder.eval()
    logger.info("[load_chinese_hubert] Chinese Hubert模型加载完成, device=%s", client_device)
    return audio_encoder, audio_feature_extractor

# 具体加载脸部的vq模型
def load_face_vqvae():
    logger.info("[load_face_vqvae] 开始加载Face VQVAE模型")
    face_vq_parser = argparse.ArgumentParser()
    face_vq_parser.add_argument("--vae_test_dim", default=51, type=int) # arkit infer
    face_vq_parser.add_argument("--vae_length", default=512, type=int)
    face_vq_parser.add_argument("--vae_codebook_size", default=2048, type=int)
    face_vq_parser.add_argument("--vae_layer", default=2, type=int)
    face_vq_parser.add_argument("--vae_stride", default=2, type=int)
    face_vq_parser.add_argument("--pose_dims", default=102, type=int) # arkit infer
    face_vq_parser.add_argument("--audio_feat_dims", default=768, type=int)
    face_vq_parser.add_argument("--vae_quantizer_lambda", default=1.0, type=float)
    face_vq_parser.add_argument("--facial_norm", default=False, type=bool)
    face_vq_args = face_vq_parser.parse_args(args=[])

    _module_dir = os.path.dirname(os.path.abspath(__file__))
    base_model_folder = os.path.join(_module_dir, "..", "checkpoints", "face_vqvae")
    ckpt_path = os.path.join(base_model_folder, "pytorch_model_face_fad2cl_260116_codesize2048_codelength512.bin")
    
    # spec = importlib.util.spec_from_file_location("face_vq", f'{base_model_folder}/face_model_af.py')
    # module = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(module)

    face_model = Af2FaceVQVAEConvZeroStrideV3(face_vq_args)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    face_model.load_state_dict(ckpt)

    # load VQVAE
    face_model.eval()
    face_model.to(client_device)
    weight_matrix = np.load(os.path.join(base_model_folder, 'mat_final.npy'))
    weight_matrix_R_I = np.load(os.path.join(base_model_folder, 'mat_final_R_I.npy'))

    logger.info("[load_face_vqvae] Face VQVAE模型加载完成, device=%s", client_device)
    return face_model, weight_matrix, weight_matrix_R_I

def infer_face_vqvae(input_data: dict, audio_feature_extractor, audio_encoder, face_vq_model):
    '''
    输入：
    {
        "name": "【表情：平静】",
        "id": 0,
        "frame": 60,
        "expect_frame": 90,
        "fps": 30,
        "motion":[
            { // 每个dict是一帧数据
                "face": [0.01, 0.02, ..., 0.52]  // 面部arkit值，共51个
            }
        ]
        "tts": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA..."，
        "tts_frames": 72000, // 音频包长度
        "tts_hz": 24000, // 音频赫兹
    }

    输出：
    {
        "name": "【表情：平静】",
        "id": 0,
        "frame": 90,
        "fps": 30,
        "motion":[
            { // 每个dict是一帧数据
                "face": [0.01, 0.02, ..., 0.52]  // 面部mta值，共52个
            }
        ]
        "tts": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA..."，
        "tts_frames": 72000, // 音频包长度
        "tts_hz": 24000, // 音频赫兹
    }
    '''

    arkit_standard_names =  [
        'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
        'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight',
        'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft',
        'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight',
        'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 
        'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft',
        'mouthFrownRight', 'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight',
        'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower',
        'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight',
        'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight',
    ]
    arkit_standard_names_lower = [name.lower() for name in arkit_standard_names]
    mta_names = [
        'EyeBlinkLeft', 'EyeLookDownLeft', 'EyeLookInLeft', 'EyeLookOutLeft', 'EyeLookUpLeft', 'EyeSquintLeft', 
        'EyeWideLeft', 'EyeBlinkRight', 'EyeLookDownRight', 'EyeLookInRight', 'EyeLookOutRight', 'EyeLookUpRight',
        'EyeSquintRight', 'EyeWideRight', 'JawForward', 'JawLeft', 'JawRight', 'JawOpen', 'MouthClose', 'MouthFunnel',
        'MouthPucker', 'MouthLeft', 'MouthRight', 'MouthSmileLeft', 'MouthSmileRight', 'MouthFrownLeft', 'MouthFrownRight',
        'MouthDimpleLeft', 'MouthDimpleRight', 'MouthStretchLeft', 'MouthStretchRight', 'MouthRollLower', 'MouthRollUpper',
        'MouthShrugLower', 'MouthShrugUpper', 'MouthPressLeft', 'MouthPressRight', 'MouthLowerDownLeft', 'MouthLowerDownRight',
        'MouthUpperUpLeft', 'MouthUpperUpRight', 'BrowDownLeft', 'BrowDownRight', 'BrowInnerUp', 'BrowOuterUpLeft', 
        'BrowOuterUpRight', 'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight', 'NoseSneerLeft', 'NoseSneerRight', 'TongueOut'
    ]

    # 读取脸数据库数据，并预处理
    record_facial = np.array([frame_data['face'] for frame_data in input_data['motion']])

    ### 4.0的lip在3.0基础上增加【jawOpen 24】,
    ### 这里的lip以及not_lip中的index应该与训练保持一致，保证每个通道的数据可以正确获取且拼接正确
    lip_col_index_list = [24, 26, 31, 33, 34, 42, 47, 48] + [37, 39, 40, 41] # 【jawOpen 24】【mouthClose 26】【mouthFunnel 31】【mouthLowerDownLeft 33】【mouthLowerDownRight 34】【mouthShrugUpper 42】【mouthUpperUpLeft 47】 【mouthUpperUpRight 48】
    # [37, 39, 40, 41] 模棱两可的lip 【mouthPucker 37】【mouthRollLower 39】【mouthRollUpper 40】【mouthShrugLower 41】
    not_lip_col_index_list = list(range(0,24)) + list(range(27,31)) + [25, 32, 35, 36, 38, 43, 44, 45, 46, 49, 50]
    ### 这里用于数据后处理，保证只清除/获取想要通道的数据，可与训练不一致
    not_lip_col_index_list_for_postprocess = list(range(0,24)) + list(range(27,31)) + [25, 32, 35, 36, 37, 38, 43, 44, 45, 46, 49, 50]

    # 读取tts音频
    if isinstance(input_data['tts'], str):
        wav, sr = sf.read(input_data['tts'])
    elif isinstance(input_data['tts'], bytes):
        with io.BytesIO(input_data['tts']) as f:
            wav, sr = sf.read(f)

    # Chinese Hubert推理
    input_values = audio_feature_extractor(wav, return_tensors="pt", sampling_rate=16000).input_values
    input_values = input_values.to(client_device)
    with torch.no_grad():
        outputs = audio_encoder(input_values)
        audio_feature = outputs.last_hidden_state

    # 新的arkit原生vq推理
    need_token_len = int(np.ceil(input_data["expect_frame"] / 3))
    zero_feat_token = [1009] * need_token_len #1009 是0 feat token，基于20260116的模型
    zero_feat_token[0] = 878
    zero_feat_token[1] = 878
    zero_feat_token[-1] = 878
    mtokens = torch.tensor(zero_feat_token, dtype=int, device=client_device).unsqueeze(0)
    net_out = face_vq_model.decode(mtokens, af_inputs=audio_feature)
    bs, n, j  = net_out.shape[0], net_out.shape[1], 1
    rec_contour_a = net_out[:, :, :int(net_out.shape[-1]//2)].reshape(bs*n, -1).detach().cpu().numpy()
    rec_contour_a[:, lip_col_index_list] = 0.0
    rec_lip_a = net_out[:, :, int(net_out.shape[-1]//2):].reshape(bs*n, -1).detach().cpu().numpy()
    rec_lip_a[:, not_lip_col_index_list] = 0.0
    # print(f'msg from face_infer_vq: {rec_contour_at.shape}, {rec_lip_a.shape}')
    face_data = rec_contour_a + rec_lip_a

    # resample音频驱动生成的唇部动作，由20fps重采样至30fps
    face_data = face_data[:, :, None]
    face_data = resample_motion(face_data, 20, input_data['fps'])
    rec_lip_a_resample = face_data[:input_data["expect_frame"], :, 0]
    rec_lip_a_resample[:, not_lip_col_index_list_for_postprocess] = 0.0

    # 数据库脸部表情再填充
    need_len = rec_lip_a_resample.shape[0] #目标长度为重采样至30fps的唇部表情长度
    orig_n = record_facial.shape[0]
    if orig_n == need_len:
        record_facial_padded = record_facial
    elif orig_n < 2:
        # Not enough points for spline: repeat rows
        record_facial_padded = np.repeat(record_facial[:1, :], need_len, axis=0)
    elif orig_n > need_len:
        record_facial_padded = record_facial[:need_len]
    elif orig_n < need_len:
        x_old = np.arange(orig_n)
        x_new = np.linspace(0, orig_n - 1, need_len)
        # CubicSpline supports vectorized y with axis=0
        cs = CubicSpline(x_old, record_facial, axis=0)
        record_facial_padded = cs(x_new)
    record_facial_padded[:, lip_col_index_list] = 0.0

    # 录制好的脸部与音频驱动生成的唇部，两个部分长度对齐后，进行融合拼接
    generate_arkit_face_data = rec_lip_a_resample + record_facial_padded
    
    #由arkit定义的顺序重排序为mta定义的顺序 n*51变为n*52
    num_frames = generate_arkit_face_data.shape[0]
    num_nodes = len(mta_names)
    generate_mta_face_data = np.array([[0.0 for _ in range(num_nodes)] for _ in range(num_frames)])
    for m, node_name in enumerate(mta_names):
        if node_name.lower() not in arkit_standard_names_lower:
            continue
        else:
            idx = arkit_standard_names_lower.index(node_name.lower())
            generate_mta_face_data[:, m] = generate_arkit_face_data[:, idx]

    # 生成输出字典数据
    output_data = input_data
    output_data['motion'] = [{'face': temp.tolist()} for temp in generate_mta_face_data]
    expect_frame = output_data['expect_frame']
    # while len(output_data['motion']) < expect_frame:
    #     output_data['motion'].append(output_data['motion'][-1])
    # if len(output_data['motion']) > expect_frame:
    #     output_data['motion'] = output_data['motion'][:]
    output_data['frame'] = expect_frame
    del output_data['expect_frame']

    return output_data