import re
import numpy as np
import torch 
from utils.visualization_torch.Animation import Animation
from utils.visualization_torch.Quaternions import Quaternions

channelmap = {
    'Xrotation': 'x',
    'Yrotation': 'y',
    'Zrotation': 'z'
}

channelmap_inv = {
    'x': 'Xrotation',
    'y': 'Yrotation',
    'z': 'Zrotation',
}

ordermap = {
    'x': 0,
    'y': 1,
    'z': 2,
}

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

def smooth_euler_animation(animation_data, window_length=11, polyorder=2, sigma=1.5, mode='savgol'):
    """
    平滑欧拉角动画数据，处理角度周期性特性
    
    参数:
    animation_data: numpy数组，形状为(frames, joint_num, 3)，包含欧拉角数据
    window_length: Savitzky-Golay滤波器窗口长度（必须是奇数）
    polyorder: Savitzky-Golay滤波器多项式阶数
    sigma: 高斯滤波的标准差（当mode='gaussian'时使用）
    mode: 平滑模式，'savgol'或'gaussian'
    
    返回:
    平滑后的动画数据，形状与输入相同
    """
    # 参数校验
    if window_length % 2 == 0:
        window_length += 1  # 确保窗口长度为奇数
        print(f"警告: 窗口长度调整为奇数: {window_length}")
    
    if polyorder >= window_length:
        polyorder = window_length - 1
        print(f"警告: 多项式阶数调整为: {polyorder}")
    
    frames, joint_num, _ = animation_data.shape
    smoothed_data = np.zeros_like(animation_data)
    
    # 处理欧拉角的周期性（0-360度范围）
    # 先将角度规范化到0-360度范围内，避免360度跳变问题
    animation_data = animation_data + 180
    normalized_data = animation_data % 360
    
    for joint in range(joint_num):
        for dim in range(3):  # 对每个欧拉角维度单独处理
            angle_series = normalized_data[:, joint, dim]
            
            # 处理角度跳变问题（如从359度到1度实际上是2度的变化）
            # 通过添加/减去360度来消除跳变
            unwrapped_series = np.unwrap(angle_series, period=360)
            
            if mode == 'savgol':
                # 使用Savitzky-Golay滤波器进行平滑
                smoothed_series = savgol_filter(unwrapped_series, 
                                               window_length, 
                                               polyorder,
                                               mode='mirror')  # 镜像边界处理
            elif mode == 'gaussian':
                # 使用高斯滤波
                smoothed_series = gaussian_filter1d(unwrapped_series, 
                                                   sigma=sigma, 
                                                   mode='mirror')
            else:
                raise ValueError("模式必须是'savgol'或'gaussian'")
            
            # 将角度重新映射回0-360度范围
            smoothed_data[:, joint, dim] = smoothed_series % 360
    smoothed_data = smoothed_data - 180
    return smoothed_data


def load(filename, start=None, end=None, order=None, world=False, need_quater=True):
    """
    Reads a BVH file and constructs an animation

    Parameters
    ----------
    filename: str
        File to be opened

    start : int
        Optional Starting Frame

    end : int
        Optional Ending Frame

    order : str
        Optional Specifier for joint order.
        Given as string E.G 'xyz', 'zxy'

    world : bool
        If set to true euler angles are applied
        together in world space rather than local
        space

    Returns
    -------

    (animation, joint_names, frametime)
        Tuple of loaded animation and joint names
    """

    f = open(filename, "r")

    i = 0
    active = -1
    end_site = False

    names = []
    orients = Quaternions.id(0)
    offsets = np.array([]).reshape((0, 3))
    parents = np.array([], dtype=int)

    for line in f:

        if "HIERARCHY" in line: continue
        if "MOTION" in line: continue

        """ Modified line read to handle mixamo data """
        #        rmatch = re.match(r"ROOT (\w+)", line)
        rmatch = re.match(r"ROOT (\w+:?\w+)", line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients.qs = np.append(orients.qs, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            continue

        if "{" in line: continue

        if "}" in line:
            if end_site:
                end_site = False
            else:
                active = parents[active]
            continue

        offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
        if offmatch:
            if not end_site:
                offsets[active] = np.array([list(map(float, offmatch.groups()))])
            continue

        chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chanmatch:
            channels = int(chanmatch.group(1))
            if order is None:
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2 + channelis:2 + channelie]
                if any([p not in channelmap for p in parts]):
                    continue
                order = "".join([channelmap[p] for p in parts])
            continue

        """ Modified line read to handle mixamo data """
        #        jmatch = re.match("\s*JOINT\s+(\w+)", line)
        jmatch = re.match("\s*JOINT\s+(\w+:?\w+)", line)
        if jmatch:
            names.append(jmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients.qs = np.append(orients.qs, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            continue
        
        if "End Site" in line:
            end_site = True
            continue
        
        fmatch = re.match("\s*Frames:\s+(\d+)", line)
        if fmatch:
            if start and end:
                fnum = (end - start) - 1
            else:
                fnum = int(fmatch.group(1))
            jnum = len(parents)
            positions = offsets[np.newaxis].repeat(fnum, axis=0)
            rotations = np.zeros((fnum, len(orients), 3))
            continue

        fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
        if fmatch:
            frametime = float(fmatch.group(1))
            continue

        if (start and end) and (i < start or i >= end - 1):
            i += 1
            continue

        # dmatch = line.strip().split(' ')
        dmatch = line.strip().split()
        if dmatch:
            data_block = np.array(list(map(float, dmatch)))
            N = len(parents)
            fi = i - start if start else i
            if channels == 3:
                positions[fi, 0:1] = data_block[0:3]
                rotations[fi, :] = data_block[3:].reshape(N, 3)
            elif channels == 6:
                data_block = data_block.reshape(N, 6)
                positions[fi, :] = data_block[:, 0:3]
                rotations[fi, :] = data_block[:, 3:6]
            elif channels == 9:
                positions[fi, 0] = data_block[0:3]
                data_block = data_block[3:].reshape(N - 1, 9)
                rotations[fi, 1:] = data_block[:, 3:6]
                positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
            else:
                raise Exception("Too many channels! %i" % channels)

            i += 1

    f.close()

    if need_quater:
        rotations = Quaternions.from_euler(torch.tensor(np.radians(rotations)), order=order, world=world)
    elif order != 'xyz':
        rotations = Quaternions.from_euler(torch.tensor(np.radians(rotations)), order=order, world=world)
        rotations = torch.rad2deg(rotations.euler())
    
    offsets = torch.tensor(offsets)
    parents = torch.tensor(parents)
    positions = torch.tensor(positions)
    orients.qs = torch.tensor(orients.qs) 
    return Animation(rotations, positions, orients, offsets, parents, names, frametime)


def save(filename, anim, names=None, frametime=1.0 / 24.0, order='zyx', positions=False, mask=None, quater=False):
    """
    Saves an Animation to file as BVH

    Parameters
    ----------
    filename: str
        File to be saved to

    anim : Animation
        Animation to save

    names : [str]
        List of joint names

    order : str
        Optional Specifier for joint order.
        Given as string E.G 'xyz', 'zxy'

    frametime : float
        Optional Animation Frame time

    positions : bool
        Optional specfier to save bone
        positions for each frame

    orients : bool
        Multiply joint orients to the rotations
        before saving.

    """

    if names is None:
        names = ["joint_" + str(i) for i in range(len(anim.parents))]

    with open(filename, 'w') as f:

        t = ""
        f.write("%sHIERARCHY\n" % t)
        f.write("%sROOT %s\n" % (t, names[0]))
        f.write("%s{\n" % t)
        t += '\t'

        f.write("%sOFFSET %f %f %f\n" % (t, anim.offsets[0, 0], anim.offsets[0, 1], anim.offsets[0, 2]))
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" %
                (t, channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))

        for i in range(anim.shape[1]):
            if anim.parents[i] == 0:
                t = save_joint(f, anim, names, t, i, order=order, positions=positions)

        t = t[:-1]
        f.write("%s}\n" % t)

        f.write("MOTION\n")
        f.write("Frames: %i\n" % anim.shape[0]);
        f.write("Frame Time: %f\n" % frametime);

        # if orients:
        #    rots = np.degrees((-anim.orients[np.newaxis] * anim.rotations).euler(order=order[::-1]))
        # else:
        #    rots = np.degrees(anim.rotations.euler(order=order[::-1]))
        # rots = np.degrees(anim.rotations.euler(order=order[::-1]))
        if quater:
            rots = np.degrees(anim.rotations.euler(order=order[::-1]))
        else:
            rots = anim.rotations
        rots = smooth_euler_animation(rots)
        poss = anim.positions
        for i in range(anim.shape[0]):
            for j in range(anim.shape[1]):
                name = anim.names[j]
                
                if positions or j == 0:

                    f.write("%f %f %f %f %f %f " % (
                        poss[i, j, 0], poss[i, j, 1], poss[i, j, 2],
                        rots[i, j, ordermap[order[0]]], rots[i, j, ordermap[order[1]]], rots[i, j, ordermap[order[2]]]))
                    
                    # f.write("%f %f %f %f %f %f " % (0, 0, 0, 0, 0, 0))
                else:
                    if mask == None or mask[j] == 1:
                        if name == "ball_r" or name == "ball_l":
                            f.write("%f %f %f " % (-89.999996, 0, 0))
                        # elif name == "clavicle_r":
                        #     if i == 0:
                        #         print(rots[i, j, ordermap[order[0]]], rots[i, j, ordermap[order[1]]], rots[i, j, ordermap[order[2]]])
                        #     f.write("%f %f %f " % (
                        #         -rots[i, j, ordermap[order[0]]], rots[i, j, ordermap[order[1]]],
                        #         -rots[i, j, ordermap[order[2]]]))
                        else:
                            f.write("%f %f %f " % (
                                rots[i, j, ordermap[order[0]]], rots[i, j, ordermap[order[1]]],
                                rots[i, j, ordermap[order[2]]]))
                    else:
                        f.write("%f %f %f " % (0, 0, 0))

            f.write("\n")


def save_joint(f, anim, names, t, i, order='zyx', positions=False):
    f.write("%sJOINT %s\n" % (t, names[i]))
    f.write("%s{\n" % t)
    t += '\t'

    f.write("%sOFFSET %f %f %f\n" % (t, anim.offsets[i, 0], anim.offsets[i, 1], anim.offsets[i, 2]))

    if positions:
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" % (t,
                                                                            channelmap_inv[order[0]],
                                                                            channelmap_inv[order[1]],
                                                                            channelmap_inv[order[2]]))
    else:
        f.write("%sCHANNELS 3 %s %s %s\n" % (t,
                                             channelmap_inv[order[0]], channelmap_inv[order[1]],
                                             channelmap_inv[order[2]]))

    end_site = True

    for j in range(anim.shape[1]):
        if anim.parents[j] == i:
            t = save_joint(f, anim, names, t, j, order=order, positions=positions)
            end_site = False

    if end_site:
        f.write("%sEnd Site\n" % t)
        f.write("%s{\n" % t)
        t += '\t'
        f.write("%sOFFSET %f %f %f\n" % (t, 0.0, 0.0, 0.0))
        t = t[:-1]
        f.write("%s}\n" % t)

    t = t[:-1]
    f.write("%s}\n" % t)

    return t