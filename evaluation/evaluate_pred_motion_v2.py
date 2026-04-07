"""
Copyright 2024 LINE Corporation
...
Evaluate **pred** (or gt) generated motion ↔ text retrieval.

Metrics computed:
  - Contrastive retrieval: R@1, R@2, R@3, R@5, R@10, MedR (normal & 32-sample protocols)
  - FID (Fréchet Inception Distance): both GT-normalized and raw versions
  - Diversity: mean pairwise L2 distance in latent space for GT and generated motions
  - BAS / BHR / VOC: beat-level audio-motion alignment metrics
  - ESD (Event Sync Distance, 事件同步距离): bidirectional event-level sync error
    between audio onset events and motion velocity peak events. Lower is better (unit: seconds).
    ESD = (d_a→m + d_m→a) / 2, where d_a→m is mean nearest distance from audio to motion,
    and d_m→a is the reverse. Uses dynamic threshold (mean+0.2*std) for motion peak detection.
  (FID & Diversity require both pred and gt npy files in the same motion_dir)

Usage (pred, default):
    cd ChronAccRet-master
    python evaluate_pred_motion.py

Override motion_type to gt (for future use):
    python evaluate_pred_motion.py +eval.motion_type=gt

Override any path:
    python evaluate_pred_motion.py \
        +eval.motion_type=pred \
        +eval.motion_dir=/path/to/your/output_dir \
        +eval.motion2text_path=/path/to/motion2text.json \
        +eval.model_path=/path/to/best_model.pt \
        +eval.output_dir=/path/to/save/metrics \
        +eval.diversity_times=300
"""

import os 
# GPU 选择通过 CUDA_VISIBLE_DEVICES 环境变量控制（见 scripts/run_eval.sh）
import logging
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import random, os
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from scipy import linalg
import librosa
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d

from datasets.pred_motion_dataset import PredMotionTextDataset
from models.models import ChronTMR
from models.metrics import all_contrastive_metrics, print_latex_metrics
from datasets.datasets import token_process

os.environ["TOKENIZERS_PARALLELISM"] = "true"
logger = logging.getLogger(__name__)

# ─── 默认路径配置 ─────────────────────────────────────────────────────────────
# 可通过命令行 "+eval.xxx=..." 覆盖

# ours

# DEFAULT_MOTION_DIR = (
#     "/data/home/jinch/tech_report/susu_avatar_training_gen_demo"
#     "/VQ_V0205/output_gen/pipeline_reconstruct_hubertfeats_vTag_eval"
# )

# baseline: emage
# DEFAULT_MOTION_DIR = (
#     "/data/home/jinch/tech_report/susu_avatar_training_gen_demo"
#     "/baseline_results/susu_avatar_emage_align"
# )

# # ablation: from raw
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MOTION_DIR = os.path.join(_PROJECT_DIR, "output/reconstructed")

# ablation: only_text
# DEFAULT_MOTION_DIR = (
#     "/data/home/jinch/tech_report/susu_avatar_training_gen_demo"
#     "/VQ_V0205/output_gen_ablation/only_text"
# )

# # ablation: step8
# DEFAULT_MOTION_DIR = (
#     "/data/home/jinch/tech_report/susu_avatar_training_gen_demo"
#     "/VQ_V0205/output_gen_ablation/step2"
# )


# # ablation: onlytext_noaudio
# DEFAULT_MOTION_DIR = (
#     "/data/home/jinch/tech_report/susu_avatar_training_gen_demo"
#     "/VQ_V0205/output_gen_ablation/onlytext_noaudio"
# )

# ablation: token_by_token
# DEFAULT_MOTION_DIR = (
#     "/data/home/jinch/tech_report/susu_avatar_training_gen_demo"
#     "/VQ_V0205/output_gen_ablation/token_by_token"
# )

# ablation: ours_infillnoaudio
# DEFAULT_MOTION_DIR = (
#     "/data/home/jinch/tech_report/susu_avatar_training_gen_demo"
#     "/VQ_V0205/output_gen_ablation/ours_infillnoaudio"
# )

# ablation: token_by_token_onlyaudio
# DEFAULT_MOTION_DIR = (
#     "/data/home/jinch/tech_report/susu_avatar_training_gen_demo"
#     "/VQ_V0205/output_gen_ablation/token_by_token_onlyaudio"
# )


DEFAULT_MOTION2TEXT_PATH = os.path.join(_PROJECT_DIR, "data/text_data/motion2text.json")
DEFAULT_STATS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stats/humanml3d/guoh3dfeats")
DEFAULT_MODEL_PATH = os.path.join(_PROJECT_DIR, "checkpoints/eval_model/best_model.pt")
DEFAULT_OUTPUT_DIR = os.path.join(_PROJECT_DIR, "output/eval_results")
DEFAULT_MOTION_TYPE = "pred"  # "pred" or "gt"
DEFAULT_DIVERSITY_TIMES = 300  # number of random pairs for diversity
# ─────────────────────────────────────────────────────────────────────────────


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def save_metric(path, metrics):
    strings = yaml.dump(metrics, indent=4, sort_keys=False)
    with open(path, "w") as f:
        f.write(strings)


def transpose(x):
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))


def get_sim_matrix(x, y):
    x_logits = torch.nn.functional.normalize(x, dim=-1)
    y_logits = torch.nn.functional.normalize(y, dim=-1)
    return x_logits @ transpose(y_logits)


# ─────────────────────────────────────────────────────────────────────────────
# FID / Diversity helpers (ported from chronaccret_eval_npz.py)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_activation_statistics(activations: np.ndarray):
    """Compute mean and covariance of activations (features)."""
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute the Fréchet Distance between two multivariate Gaussians."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def calculate_diversity(activation: np.ndarray, diversity_times: int):
    """Compute diversity as mean pairwise L2 distance among random pairs."""
    assert activation.ndim == 2
    num_samples = activation.shape[0]
    diversity_times = min(diversity_times, num_samples - 1)
    if diversity_times <= 1:
        return 0.0
    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return float(dist.mean())


def calculate_frechet_feature_distance(feature_list1: np.ndarray, feature_list2: np.ndarray):
    """
    Normalize both feature sets using feature_list1 (GT) mean/std, then compute FID.
    Matches the logic in chronaccret_eval_npz.py.
    """
    feature_list1 = np.asarray(feature_list1)
    feature_list2 = np.asarray(feature_list2)
    mean = np.mean(feature_list1, axis=0)
    std = np.std(feature_list1, axis=0) + 1e-10
    f1 = (feature_list1 - mean) / std
    f2 = (feature_list2 - mean) / std
    return calculate_frechet_distance(
        mu1=np.mean(f1, axis=0),
        sigma1=np.cov(f1, rowvar=False),
        mu2=np.mean(f2, axis=0),
        sigma2=np.cov(f2, rowvar=False),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Beat Alignment Score (BAS) — audio-motion temporal alignment metric
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_WAV_DIR = os.path.join(_PROJECT_DIR, "data/wav_data")


def extract_audio_beats(wav_path):
    """
    Extract audio beats using librosa beat tracking.
    Returns beat times in seconds.
    """
    y, sr = librosa.load(wav_path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    audio_beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return audio_beat_times


def extract_motion_beats(motion_np, fps=20):
    """
    Extract motion beats as velocity peaks (moments of maximum motion).
    Uses find_peaks with minimum distance = fps//3 (~3 beats/sec max).
    Returns beat times in seconds.
    """
    from scipy.signal import find_peaks
    if motion_np.shape[0] < 5:
        return np.array([])
    velocity = np.diff(motion_np, axis=0)
    vel_magnitude = np.linalg.norm(velocity, axis=1)
    peaks, _ = find_peaks(vel_magnitude, distance=max(fps // 3, 1))
    motion_beat_times = peaks / fps
    return motion_beat_times


def extract_motion_beats_for_esd(motion_np, fps=20):
    """
    Extract motion beats for ESD (Event Sync Distance) metric.
    Uses dynamic amplitude threshold to filter out micro-jitter:
        threshold = mean(vel) + 0.2 * std(vel)
    Returns beat times in seconds.
    """
    from scipy.signal import find_peaks
    if motion_np is None or len(motion_np) < 2:
        return np.array([])
    velocity = np.diff(motion_np, axis=0)
    if velocity.ndim == 3:
        vel_magnitude = np.linalg.norm(velocity, axis=(1, 2))
    else:
        vel_magnitude = np.linalg.norm(velocity, axis=1)
    # Dynamic threshold: only accept peaks above mean + 0.2*std
    height_threshold = np.mean(vel_magnitude) + 0.2 * np.std(vel_magnitude)
    peaks, _ = find_peaks(vel_magnitude, distance=1, height=height_threshold)
    return peaks / fps


def calculate_esd(audio_times, motion_times, empty_penalty=2.0):
    """
    ESD — Event Sync Distance (事件同步距离)

    Measures bidirectional temporal alignment between audio events and motion events.

    Definition:
        d_a→m = (1/n) Σ_i min_j |a_i - m_j|   (audio→motion, reflects recall)
        d_m→a = (1/k) Σ_j min_i |m_j - a_i|   (motion→audio, reflects precision)
        ESD = (d_a→m + d_m→a) / 2

    Lower is better. Unit: seconds.

    Args:
        audio_times:   array of audio event times (seconds)
        motion_times:  array of motion event times (seconds)
        empty_penalty: penalty value when one side has no events

    Returns:
        float: ESD score in seconds
    """
    audio_times = np.asarray(audio_times, dtype=np.float64)
    motion_times = np.asarray(motion_times, dtype=np.float64)

    n_audio = len(audio_times)
    n_motion = len(motion_times)

    if n_audio == 0 and n_motion == 0:
        return 0.0
    if n_audio == 0 or n_motion == 0:
        return float(empty_penalty)

    # Distance matrix: |a_i - m_j|, shape (n_audio, n_motion)
    dist = np.abs(audio_times[:, None] - motion_times[None, :])
    d_a_to_m = float(dist.min(axis=1).mean())  # audio → nearest motion
    d_m_to_a = float(dist.min(axis=0).mean())  # motion → nearest audio
    return 0.5 * (d_a_to_m + d_m_to_a)


def velocity_onset_correlation(motion_np, wav_path, fps=20, sr=16000, sigma=1.0):
    """
    Compute Pearson correlation between motion velocity and audio onset envelope.

    This is a more discriminative audio-motion alignment metric than beat-level
    BAS, especially for speech data where audio onsets are dense.

    Args:
        motion_np: (T, D) raw motion features (unnormalized).
        wav_path: path to .wav file.
        fps: motion frame rate.
        sr: audio sample rate.
        sigma: Gaussian smoothing sigma for velocity curve.

    Returns:
        float: Pearson correlation coefficient, or None if too short.
    """
    from scipy.stats import pearsonr

    if motion_np.shape[0] < 5:
        return None

    # motion velocity magnitude
    velocity = np.linalg.norm(np.diff(motion_np, axis=0), axis=-1)  # (T-1,)
    velocity = gaussian_filter1d(velocity, sigma=sigma)

    # audio onset envelope
    audio, _ = librosa.load(wav_path, sr=sr)
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=512)
    onset_times = librosa.frames_to_time(
        np.arange(len(onset_env)), sr=sr, hop_length=512
    )

    # resample onset envelope to motion frame rate
    motion_times = np.arange(len(velocity)) / fps
    onset_resampled = np.interp(motion_times, onset_times, onset_env)

    # Pearson correlation, mapped to [0, 100] scale
    # 50 = no correlation, >50 = positive, <50 = negative
    v_std = velocity.std()
    o_std = onset_resampled.std()
    if v_std < 1e-8 or o_std < 1e-8:
        return None

    corr, _ = pearsonr(velocity, onset_resampled)
    return (corr + 1.0) / 2.0 * 100.0  # map [-1,1] → [0,100]


def calculate_alignment_single(motion_beats, audio_beats, tolerance=0.1):
    """
    Compute BAS (avg distance, lower=better) and BHR (hit rate, higher=better)
    for a single sample.
    """
    if len(motion_beats) == 0 or len(audio_beats) == 0:
        return None, None

    distances = []
    hits = 0
    for mt in motion_beats:
        diffs = np.abs(audio_beats - mt)
        min_diff = np.min(diffs)
        distances.append(min_diff)
        if min_diff <= tolerance:
            hits += 1

    bas_distance = np.mean(distances)  # lower is better
    bhr_rate = hits / len(motion_beats)  # higher is better
    return bas_distance, bhr_rate


def compute_beat_alignment_metrics(dataset, wav_dir, fps=20, tolerance=0.1):
    """
    Compute audio-motion alignment metrics across all samples in the dataset.

    Reports complementary metrics:
    - BAS (Beat Alignment Score): hit rate of kinematic beats near audio onsets.
    - VOC (Velocity-Onset Correlation): Pearson correlation between motion velocity
      and audio onset envelope, resampled to motion frame rate.
    - ESD (Event Sync Distance, 事件同步距离): bidirectional average distance
      between audio events and motion events. Lower is better.

    VOC is more discriminative for speech-driven motion since audio onsets are
    dense in speech data, making BAS less sensitive to method differences.
    """
    bas_scores = []
    bhr_scores = []
    voc_scores = []
    esd_scores = []
    num_skipped = 0

    for idx in tqdm(range(len(dataset)), desc="Computing Beat Alignment"):
        npy_path, text, name = dataset.samples[idx]

        # ── Load raw motion (unnormalized) ──
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
        except Exception:
            num_skipped += 1
            continue

        # ── Find corresponding wav ──
        wav_path = os.path.join(wav_dir, name + ".wav")
        if not os.path.exists(wav_path):
            num_skipped += 1
            continue

        # ── Compute BAS (distance) + BHR (hit rate) ──
        motion_beats = extract_motion_beats(body, fps=fps)
        try:
            audio_beats = extract_audio_beats(wav_path)
        except Exception:
            num_skipped += 1
            continue

        bas_dist, bhr = calculate_alignment_single(motion_beats, audio_beats, tolerance)
        if bas_dist is not None:
            bas_scores.append(bas_dist)
        if bhr is not None:
            bhr_scores.append(bhr)

        # ── Compute ESD (Event Sync Distance, 事件同步距离) ──
        try:
            esd_motion_beats = extract_motion_beats_for_esd(body, fps=fps)
            esd_val = calculate_esd(audio_beats, esd_motion_beats)
            esd_scores.append(esd_val)
        except Exception:
            pass

        # ── Compute VOC (Velocity-Onset Correlation) ──
        try:
            voc = velocity_onset_correlation(body, wav_path, fps=fps)
            if voc is not None:
                voc_scores.append(voc)
        except Exception:
            pass

    metrics = {}
    if len(bas_scores) > 0:
        bas_arr = np.array(bas_scores)
        metrics["BAS_mean"] = round(float(bas_arr.mean()), 4)
        metrics["BAS_std"] = round(float(bas_arr.std()), 4)
    else:
        metrics["BAS_mean"] = 0.0
        metrics["BAS_std"] = 0.0

    if len(bhr_scores) > 0:
        bhr_arr = np.array(bhr_scores)
        metrics["BHR_mean"] = round(float(bhr_arr.mean()), 4)
        metrics["BHR_std"] = round(float(bhr_arr.std()), 4)
    else:
        metrics["BHR_mean"] = 0.0
        metrics["BHR_std"] = 0.0

    if len(voc_scores) > 0:
        voc_arr = np.array(voc_scores)
        metrics["VOC_mean"] = round(float(voc_arr.mean()), 6)
        metrics["VOC_std"] = round(float(voc_arr.std()), 6)
    else:
        metrics["VOC_mean"] = 0.0
        metrics["VOC_std"] = 0.0

    if len(esd_scores) > 0:
        esd_arr = np.array(esd_scores)
        metrics["ESD_mean"] = round(float(esd_arr.mean()), 4)
        metrics["ESD_std"] = round(float(esd_arr.std()), 4)
    else:
        metrics["ESD_mean"] = 0.0
        metrics["ESD_std"] = 0.0

    metrics["num_evaluated"] = max(len(bas_scores), len(voc_scores), len(esd_scores))
    metrics["num_skipped"] = num_skipped
    return metrics


def compute_beat_alignment_metrics_from_dir(
    dataset, gt_motion_dir, wav_dir, fps=20, tolerance=0.1
):
    """
    Compute beat alignment metrics using real GT motion files.
    Loads motion from gt_motion_dir/{name}.npy instead of from dataset npy files.
    Uses the dataset only to get the list of sample names.
    """
    bas_scores = []
    bhr_scores = []
    voc_scores = []
    esd_scores = []
    num_skipped = 0

    for idx in tqdm(range(len(dataset)), desc="Computing Beat Alignment (real GT)"):
        _, text, name = dataset.samples[idx]

        # ── Load real GT motion ──
        gt_path = os.path.join(gt_motion_dir, name + ".npy")
        if not os.path.exists(gt_path):
            num_skipped += 1
            continue
        try:
            raw = np.load(gt_path, allow_pickle=True)
            if raw.shape == ():
                raw = raw.item()
            body = np.array(raw["body"])  # (T, D)
        except Exception:
            num_skipped += 1
            continue

        # ── Find corresponding wav ──
        wav_path = os.path.join(wav_dir, name + ".wav")
        if not os.path.exists(wav_path):
            num_skipped += 1
            continue

        # ── BAS + BHR ──
        motion_beats = extract_motion_beats(body, fps=fps)
        try:
            audio_beats = extract_audio_beats(wav_path)
        except Exception:
            num_skipped += 1
            continue

        bas_dist, bhr = calculate_alignment_single(motion_beats, audio_beats, tolerance)
        if bas_dist is not None:
            bas_scores.append(bas_dist)
        if bhr is not None:
            bhr_scores.append(bhr)

        # ── ESD (Event Sync Distance, 事件同步距离) ──
        try:
            esd_motion_beats = extract_motion_beats_for_esd(body, fps=fps)
            esd_val = calculate_esd(audio_beats, esd_motion_beats)
            esd_scores.append(esd_val)
        except Exception:
            pass

        # ── VOC ──
        try:
            voc = velocity_onset_correlation(body, wav_path, fps=fps)
            if voc is not None:
                voc_scores.append(voc)
        except Exception:
            pass

    metrics = {}
    if len(bas_scores) > 0:
        metrics["BAS_mean"] = round(float(np.mean(bas_scores)), 4)
        metrics["BAS_std"] = round(float(np.std(bas_scores)), 4)
    else:
        metrics["BAS_mean"] = 0.0
        metrics["BAS_std"] = 0.0

    if len(bhr_scores) > 0:
        metrics["BHR_mean"] = round(float(np.mean(bhr_scores)), 4)
        metrics["BHR_std"] = round(float(np.std(bhr_scores)), 4)
    else:
        metrics["BHR_mean"] = 0.0
        metrics["BHR_std"] = 0.0

    if len(voc_scores) > 0:
        metrics["VOC_mean"] = round(float(np.mean(voc_scores)), 6)
        metrics["VOC_std"] = round(float(np.std(voc_scores)), 6)
    else:
        metrics["VOC_mean"] = 0.0
        metrics["VOC_std"] = 0.0

    if len(esd_scores) > 0:
        metrics["ESD_mean"] = round(float(np.mean(esd_scores)), 4)
        metrics["ESD_std"] = round(float(np.std(esd_scores)), 4)
    else:
        metrics["ESD_mean"] = 0.0
        metrics["ESD_std"] = 0.0

    metrics["num_evaluated"] = max(len(bas_scores), len(voc_scores), len(esd_scores))
    metrics["num_skipped"] = num_skipped
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers (same logic as evaluate_gen_pred.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_sim_matrix(cfg, model, dataset, indices, device, tokenizer):
    """Compute full N×N similarity matrix (normal protocol)."""
    with torch.no_grad():
        latent_texts, latent_motions = [], []
        for index in tqdm(indices, desc="Normal protocol"):
            text, motion, length, _, _, _ = dataset.load_keyid(index)

            length = torch.Tensor([length]).to(device).int()
            motion = motion.to(device).unsqueeze(0)
            texts_token, t_length = token_process(
                cfg.model.token_num, cfg.model.text_model_name, [text], tokenizer
            )
            texts_token = texts_token.to(device)
            t_length = torch.Tensor([t_length]).to(device).int()

            texts_emb = model.text_model(texts_token, t_length).float()
            latent_text, _ = model.encode(
                texts_emb, t_length, "txt",
                sample_mean=cfg.text_encoder.vae,
                return_distribution=cfg.text_encoder.vae,
            )
            latent_motion, _ = model.encode(
                motion, length, "motion",
                sample_mean=cfg.text_encoder.vae,
                return_distribution=cfg.text_encoder.vae,
            )
            latent_texts.append(latent_text)
            latent_motions.append(latent_motion)

        latent_texts = torch.cat(latent_texts)
        latent_motions = torch.cat(latent_motions)
        sim_matrix = get_sim_matrix(latent_texts, latent_motions)

    return {"sim_matrix": sim_matrix.cpu().numpy()}


def precompute_motion_encodings(cfg, model, dataset, device):
    """Pre-encode all motions once to save compute in 128-sample protocol."""
    with torch.no_grad():
        motion_encodings = []
        for idx in tqdm(range(len(dataset)), desc="Precomputing motion encodings"):
            _, motion, length, _, _, _ = dataset.load_keyid(idx)
            length = torch.Tensor([length]).to(device).int()
            motion = motion.to(device).unsqueeze(0)
            latent, _ = model.encode(
                motion, length, "motion",
                sample_mean=cfg.text_encoder.vae,
                return_distribution=cfg.text_encoder.vae,
            )
            motion_encodings.append(latent)
        return torch.cat(motion_encodings)


def encode_all_latents(cfg, model, dataset, device, tokenizer):
    """
    Encode all samples in the dataset, returning latent text, latent motion
    tensors (on CPU). This is used for FID / Diversity computation.
    """
    with torch.no_grad():
        latent_texts, latent_motions = [], []
        for idx in tqdm(range(len(dataset)), desc="Encoding all latents"):
            text, motion, length, _, _, _ = dataset.load_keyid(idx)

            length = torch.Tensor([length]).to(device).int()
            motion = motion.to(device).unsqueeze(0)
            texts_token, t_length = token_process(
                cfg.model.token_num, cfg.model.text_model_name, [text], tokenizer
            )
            texts_token = texts_token.to(device)
            t_length = torch.Tensor([t_length]).to(device).int()

            texts_emb = model.text_model(texts_token, t_length).float()
            latent_text, _ = model.encode(
                texts_emb, t_length, "txt",
                sample_mean=cfg.text_encoder.vae,
                return_distribution=cfg.text_encoder.vae,
            )
            latent_motion, _ = model.encode(
                motion, length, "motion",
                sample_mean=cfg.text_encoder.vae,
                return_distribution=cfg.text_encoder.vae,
            )
            latent_texts.append(latent_text.cpu())
            latent_motions.append(latent_motion.cpu())

        latent_texts = torch.cat(latent_texts, dim=0)
        latent_motions = torch.cat(latent_motions, dim=0)
    return latent_texts, latent_motions


def compute_fid_diversity_metrics(
    gt_latent_motions: np.ndarray,
    pred_latent_motions: np.ndarray,
    diversity_times: int = 300,
):
    """
    Compute FID and Diversity metrics between GT and pred latent motion features.
    Returns a dict of metrics.
    """
    gt_lat = gt_latent_motions
    gen_lat = pred_latent_motions

    # ── FID (normalized by GT statistics) ─────────────────────────────────
    fid = calculate_frechet_feature_distance(gt_lat, gen_lat)

    # ── FID (raw, without per-dim normalization) ──────────────────────────
    mu_gt, sigma_gt = calculate_activation_statistics(gt_lat)
    mu_gen, sigma_gen = calculate_activation_statistics(gen_lat)
    fid_raw = calculate_frechet_distance(mu_gt, sigma_gt, mu_gen, sigma_gen)

    # ── Diversity (computed in GT-normalized latent space) ────────────────
    mean_lat = np.mean(gt_lat, axis=0)
    std_lat = np.std(gt_lat, axis=0) + 1e-10
    gt_lat_n = (gt_lat - mean_lat) / std_lat
    gen_lat_n = (gen_lat - mean_lat) / std_lat

    div_times = min(diversity_times, gt_lat_n.shape[0] - 1)
    div_gt = calculate_diversity(gt_lat_n, div_times)
    div_gen = calculate_diversity(gen_lat_n, div_times)

    metrics = {
        "FID_norm_by_GT":     round(fid, 6),
        "FID_raw":            round(fid_raw, 6),
        "Diversity_GT":       round(div_gt, 6),
        "Diversity_Gen":      round(div_gen, 6),
    }
    return metrics


def compute_128_sample_metrics(cfg, model, dataset, device, tokenizer):
    """
    32-sample protocol (1 positive + 31 negatives) — same logic as
    compute_128_sample_metrics_optimized in evaluate_gen_pred.py.
    """
    with torch.no_grad():
        motion_encodings = precompute_motion_encodings(cfg, model, dataset, device)
        total = len(dataset)
        r1 = r2 = r3 = r5 = r10 = 0
        medr = []

        batch_size = 32
        for batch_start in tqdm(range(0, total, batch_size), desc="32-sample protocol"):
            batch_end = min(batch_start + batch_size, total)
            batch_indices = list(range(batch_start, batch_end))

            batch_texts = []
            for qi in batch_indices:
                text, _, _, _, _, _ = dataset.load_keyid(qi)
                batch_texts.append(text)

            texts_token, t_lengths = token_process(
                cfg.model.token_num, cfg.model.text_model_name, batch_texts, tokenizer
            )
            texts_token = texts_token.to(device)
            t_lengths = t_lengths.to(device)

            texts_emb = model.text_model(texts_token, t_lengths).float()
            query_latents, _ = model.encode(
                texts_emb, t_lengths, "txt",
                sample_mean=cfg.text_encoder.vae,
                return_distribution=cfg.text_encoder.vae,
            )

            for i, query_idx in enumerate(batch_indices):
                neg_indices = list(range(total))
                neg_indices.remove(query_idx)
                random.shuffle(neg_indices)
                neg_indices = neg_indices[:31]

                motion_indices = [query_idx] + neg_indices
                batch_motion_enc = motion_encodings[motion_indices]

                query_latent = query_latents[i: i + 1]
                sim_scores = get_sim_matrix(query_latent, batch_motion_enc).squeeze(0).cpu().numpy()

                sorted_ids = np.argsort(-sim_scores)
                rank = np.where(sorted_ids == 0)[0][0] + 1

                if rank == 1:  r1 += 1
                if rank <= 2:  r2 += 1
                if rank <= 3:  r3 += 1
                if rank <= 5:  r5 += 1
                if rank <= 10: r10 += 1
                medr.append(rank)

        metrics = {
            "t2m/R01":  round(r1  / total * 100, 2),
            "t2m/R02":  round(r2  / total * 100, 2),
            "t2m/R03":  round(r3  / total * 100, 2),
            "t2m/R05":  round(r5  / total * 100, 2),
            "t2m/R10":  round(r10 / total * 100, 2),
            "t2m/MedR": round(float(np.median(medr)), 2),
            "t2m/len":  float(total),
        }
    return metrics


# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(config_path="config", config_name="train_bert_orig", version_base=None)
def evaluate(cfg: DictConfig):

    # ─── 读取 eval 相关路径（支持命令行 +eval.xxx= 覆盖）──────────────────
    eval_cfg = OmegaConf.select(cfg, "eval", default=OmegaConf.create({}))
    motion_dir    = OmegaConf.select(eval_cfg, "motion_dir",      default=DEFAULT_MOTION_DIR)
    motion2text   = OmegaConf.select(eval_cfg, "motion2text_path", default=DEFAULT_MOTION2TEXT_PATH)
    stats_dir     = OmegaConf.select(eval_cfg, "stats_dir",        default=DEFAULT_STATS_DIR)
    model_path    = OmegaConf.select(eval_cfg, "model_path",       default=DEFAULT_MODEL_PATH)
    output_dir    = OmegaConf.select(eval_cfg, "output_dir",       default=DEFAULT_OUTPUT_DIR)
    motion_type   = OmegaConf.select(eval_cfg, "motion_type",      default=DEFAULT_MOTION_TYPE)

    logger.info(f"motion_type : {motion_type}")
    logger.info(f"motion_dir  : {motion_dir}")
    logger.info(f"model_path  : {model_path}")

    seed_everything(cfg.train.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ─── 数据集 ───────────────────────────────────────────────────────────
    logger.info("Loading dataset …")
    dataset = PredMotionTextDataset(
        cfg=cfg,
        motion_dir=motion_dir,
        motion2text_path=motion2text,
        stats_dir=stats_dir,
        motion_type=motion_type,
    )
    logger.info(f"Dataset size: {len(dataset)}")

    # ─── 模型 ────────────────────────────────────────────────────────────
    logger.info("Loading model …")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = ChronTMR(cfg, vae=False)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # ─── Tokenizer ───────────────────────────────────────────────────────
    if cfg.model.text_model_name == "ViT-B/32":
        from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
        tokenizer = _Tokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.text_model_name, TOKENIZERS_PARALLELISM=False
        )
    logger.info(f"Language model: {cfg.model.text_model_name}")

    # ─── GT 数据集（用于 FID / Diversity 计算）────────────────────────────
    diversity_times = int(OmegaConf.select(eval_cfg, "diversity_times",
                                           default=DEFAULT_DIVERSITY_TIMES))
    gt_dataset = None
    if motion_type == "pred":
        logger.info("Loading GT dataset for FID / Diversity computation …")
        gt_dataset = PredMotionTextDataset(
            cfg=cfg,
            motion_dir="/data/home/jinch/tech_report/susu_avatar_training_gen_demo/VQ_V0205/output_gen/pipeline_reconstruct_hubertfeats_vTag_eval",
            motion2text_path=motion2text,
            stats_dir=stats_dir,
            motion_type="gt",
        )
        logger.info(f"GT Dataset size: {len(gt_dataset)}")

    # ─── 评估协议 ─────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    protocols = ["32_samples"]

    for protocol in protocols:
        logger.info(f"\n{'='*60}")
        logger.info(f"Protocol: {protocol}  |  motion_type: {motion_type}")
        logger.info(f"{'='*60}")
        print("dataset:", dataset.__len__())
        if protocol == "32_samples":
            metrics = compute_128_sample_metrics(cfg, model, dataset, device, tokenizer)
            print(metrics)
        else:
            indices = list(range(len(dataset)))
            result  = compute_sim_matrix(cfg, model, dataset, indices, device, tokenizer)
            metrics = all_contrastive_metrics(result["sim_matrix"])
            print_latex_metrics(metrics)

        # 保存指标（文件名含 motion_type 以便区分 pred / gt）
        metric_fname = f"{motion_type}_{protocol}.yaml"
        save_path = os.path.join(output_dir, metric_fname)
        save_metric(save_path, metrics)
        logger.info(f"Metrics saved → {save_path}")

    # ─── FID / Diversity 评估 ─────────────────────────────────────────────
    if gt_dataset is not None and len(gt_dataset) > 0:
        logger.info(f"\n{'='*60}")
        logger.info("Computing FID / Diversity metrics (pred vs GT) …")
        logger.info(f"{'='*60}")

        # Encode pred latents
        logger.info("Encoding pred motion latents …")
        _, pred_latent_motions = encode_all_latents(
            cfg, model, dataset, device, tokenizer
        )

        # Encode GT latents
        logger.info("Encoding GT motion latents …")
        _, gt_latent_motions = encode_all_latents(
            cfg, model, gt_dataset, device, tokenizer
        )

        pred_lat_np = pred_latent_motions.numpy()
        gt_lat_np = gt_latent_motions.numpy()

        # Compute FID & Diversity
        fid_div_metrics = compute_fid_diversity_metrics(
            gt_latent_motions=gt_lat_np,
            pred_latent_motions=pred_lat_np,
            diversity_times=diversity_times,
        )

        logger.info("─── FID / Diversity Results ───")
        for k, v in fid_div_metrics.items():
            logger.info(f"  {k}: {v}")
        print("\n=== FID / Diversity Metrics ===")
        for k, v in fid_div_metrics.items():
            print(f"  {k}: {v}")

        # 保存 FID / Diversity 指标
        fid_fname = f"{motion_type}_fid_diversity.yaml"
        fid_save_path = os.path.join(output_dir, fid_fname)
        save_metric(fid_save_path, fid_div_metrics)
        logger.info(f"FID/Diversity metrics saved → {fid_save_path}")
    else:
        if motion_type == "pred":
            logger.warning(
                "GT dataset is empty or could not be loaded. "
                "Skipping FID / Diversity computation."
            )
        else:
            logger.info(
                "motion_type='gt' — FID / Diversity not computed "
                "(requires both pred and gt)."
            )

    # ─── Beat Alignment Score (BAS) 评估 ──────────────────────────────────
    wav_dir = OmegaConf.select(eval_cfg, "wav_dir", default=DEFAULT_WAV_DIR)
    bas_tolerance = float(OmegaConf.select(eval_cfg, "bas_tolerance", default=0.1))

    logger.info(f"\n{'='*60}")
    logger.info(f"Computing Beat Alignment Score (BAS) …")
    logger.info(f"  wav_dir:   {wav_dir}")
    logger.info(f"  tolerance: {bas_tolerance}s")
    logger.info(f"{'='*60}")

    # BAS for pred (or current motion_type)
    bas_metrics = compute_beat_alignment_metrics(
        dataset, wav_dir, fps=20, tolerance=bas_tolerance
    )
    logger.info(f"─── BAS Results ({motion_type}) ───")
    for k, v in bas_metrics.items():
        logger.info(f"  {k}: {v}")
    print(f"\n=== Beat Alignment Score ({motion_type}) ===")
    for k, v in bas_metrics.items():
        print(f"  {k}: {v}")

    bas_fname = f"{motion_type}_beat_alignment.yaml"
    bas_save_path = os.path.join(output_dir, bas_fname)
    save_metric(bas_save_path, bas_metrics)
    logger.info(f"BAS metrics saved → {bas_save_path}")

    # BAS for real GT (upper bound reference — uses original motion, not VQ-reconstructed)
    real_gt_motion_dir = os.path.join(_PROJECT_DIR, "data/motion_data")
    if True:
        logger.info("Computing BAS for real GT (upper bound) …")
        gt_bas_metrics = compute_beat_alignment_metrics_from_dir(
            dataset, real_gt_motion_dir, wav_dir, fps=20, tolerance=bas_tolerance
        )
        logger.info(f"─── BAS Results (GT) ───")
        for k, v in gt_bas_metrics.items():
            logger.info(f"  {k}: {v}")
        print(f"\n=== Beat Alignment Score (GT) ===")
        for k, v in gt_bas_metrics.items():
            print(f"  {k}: {v}")

        gt_bas_fname = "gt_beat_alignment.yaml"
        gt_bas_save_path = os.path.join(output_dir, gt_bas_fname)
        save_metric(gt_bas_save_path, gt_bas_metrics)
        logger.info(f"GT BAS metrics saved → {gt_bas_save_path}")


if __name__ == "__main__":
    evaluate()
