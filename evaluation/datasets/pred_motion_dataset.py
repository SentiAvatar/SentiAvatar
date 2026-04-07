"""
Dataset for evaluating pred (or gt) generated motion against text.
Loads *_pred.npy (or *_gt.npy) files from a directory,
looks up the corresponding text from motion2text.json,
and provides (text, motion, length, ...) samples.
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


class PredMotionTextDataset(Dataset):
    """
    Dataset that reads *_pred.npy or *_gt.npy files from a directory,
    extracts the 'name' field inside each npy to look up corresponding
    text in motion2text.json.

    Args:
        cfg: Hydra / OmegaConf config object.
        motion_dir (str): Directory containing the pred/gt npy files.
        motion2text_path (str): Path to motion2text.json.
        stats_dir (str): Directory containing mean.pt and std.pt.
        motion_type (str): "pred" or "gt" — controls which npy files are loaded.
    """

    def __init__(
        self,
        cfg,
        motion_dir: str,
        motion2text_path: str,
        stats_dir: str,
        motion_type: str = "pred",
    ):
        self.cfg = cfg
        self.motion_dir = motion_dir
        self.motion_type = motion_type  # "pred" or "gt"
        self.max_motion_length = cfg.dataset.max_motion_length
        self.eps = 1e-12

        # ── 1. 加载归一化统计量 ──────────────────────────────────────────────
        try:
            self.mean = torch.load(os.path.join(stats_dir, "mean.pt")).float()
            self.std = torch.load(os.path.join(stats_dir, "std.pt")).float()
            print(f"[*] Loaded stats from {stats_dir}, mean.shape={self.mean.shape}")
        except Exception as e:
            print(f"[!] Warning: Failed to load stats: {e}. Normalization will be skipped.")
            self.mean = None
            self.std = None

        # ── 2. 加载 motion2text 映射 ─────────────────────────────────────────
        with open(motion2text_path, "r", encoding="utf-8") as f:
            self.motion2text = json.load(f)
        print(f"[*] Loaded motion2text.json with {len(self.motion2text)} entries.")

        # ── 3. 扫描目录，收集所有 *_{motion_type}.npy 文件 ───────────────────
        suffix = f"_{motion_type}.npy"
        all_files = sorted(
            f for f in os.listdir(motion_dir) if f.endswith(suffix)
        )
        # print("all_files:", all_files)
        print(f"[*] Found {len(all_files)} '{suffix}' files in {motion_dir}")

        # ── 4. 过滤：只保留能在 motion2text.json 里找到 text 的样本 ──────────
        self.samples = []  # list of (npy_path, text, motion_name)
        missing_text = 0
        for fname in all_files:
            npy_path = os.path.join(motion_dir, fname)
            try:
                raw = np.load(npy_path, allow_pickle=True)
                if raw.shape == ():
                    raw = raw.item()
                    name = raw.get("name", None)
                else:
                    raw = {"body": raw, "left": raw, "right": raw} 
                    name = fname.replace("_2026", "/2026").replace("_Human", "/Human").replace("_2025", "/2025").replace("_pred.npy", "")
            except Exception as e:
                print(f"[!] Cannot read {fname}: {e}")
                continue
            text = self.motion2text.get(name, None)
            if text is None:
                missing_text += 1
                continue

            self.samples.append((npy_path, text, name))

        print(
            f"[*] Valid samples: {len(self.samples)} "
            f"(skipped {missing_text} without text in motion2text.json)"
        )

    # ─────────────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def load_keyid(self, index):
        """
        Returns: (text, motion_tensor, m_length, event, shuffled_event_text, name)
        Compatible with the same signature used in evaluate_gen_pred.py.
        """
        npy_path, text, name = self.samples[index]

        try:
            raw = np.load(npy_path, allow_pickle=True)
            if raw.shape == ():
                raw = raw.item()
                body = np.array(raw["body"])           # (T, 153)
                left = np.array(raw["left"])            # (1, T, 120) or (T, 120)
                right = np.array(raw["right"])          # (1, T, 120) or (T, 120)
            else:
                body = raw 
                left = raw 
                right = raw 

            # squeeze leading batch dim if present (pred/gt npy format)
            if left.ndim == 3:
                left = left.squeeze(0)              # → (T, 120)
            if right.ndim == 3:
                right = right.squeeze(0)            # → (T, 120)

            min_len = min(len(body), len(left), len(right))
            # motion_np = np.concatenate(
            #     [body[:min_len], left[:min_len], right[:min_len]], axis=1
            # )  # (T, 393)
            motion_np = body[:min_len]
            motion = torch.from_numpy(motion_np).float()

        except Exception as e:
            print(f"[!] Error loading {npy_path}: {e}. Falling back to next sample.")
            return self.load_keyid((index + 1) % len(self.samples))

        # ── 归一化 ────────────────────────────────────────────────────────────
        if self.mean is not None and self.std is not None:
            self.mean = self.mean.to(motion.device)
            self.std = self.std.to(motion.device)
            body_dim = motion.shape[1]
            self.mean = self.mean[:body_dim]
            self.std = self.std[:body_dim]
            motion = (motion - self.mean) / (self.std + self.eps)

        # ── 截断 / 填充 ───────────────────────────────────────────────────────
        m_length = len(motion)
        if m_length >= self.max_motion_length:
            motion = motion[: self.max_motion_length]
            m_length = self.max_motion_length
        elif self.cfg.dataset.padding:
            pad_len = self.max_motion_length - m_length
            pad = torch.zeros((pad_len, motion.shape[1]), dtype=torch.float)
            motion = torch.cat((motion, pad), dim=0)

        event = torch.tensor([0])
        shuffled_event_text = ""
        return text, motion, m_length, event, shuffled_event_text, name

    def __getitem__(self, index):
        return self.load_keyid(index)
