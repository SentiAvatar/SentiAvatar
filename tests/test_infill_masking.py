import json
import random

import numpy as np
import torch

from motion_generation.training.data import AudioMotionTokenWindowDataset


def _write_fixture(tmp_path):
    token_dir = tmp_path / "motion_token_data"
    audio_dir = tmp_path / "audio_features_hubert_layer9_fps10"
    token_dir.mkdir()
    audio_dir.mkdir()
    tokens = [[0, 1], [2, 3], [4, 5], [6, 7], [1, 0]]
    (token_dir / "sample.json").write_text(json.dumps({"tokens": tokens}), encoding="utf-8")
    np.save(audio_dir / "sample.npy", np.zeros((5, 3), dtype=np.float32))
    return token_dir, audio_dir


def test_infill_random_replace_defaults_to_unmasked_scope(tmp_path):
    token_dir, audio_dir = _write_fixture(tmp_path)
    random.seed(7)
    torch.manual_seed(7)
    dataset = AudioMotionTokenWindowDataset(
        motion_token_dir=str(token_dir),
        audio_feat_dir=str(audio_dir),
        names=["sample"],
        num_frames=5,
        num_tokens_per_frame=2,
        codebook_size=8,
        mask_prob=0.0,
        random_replace_prob=1.0,
        random_replace_scope="unmasked",
    )

    item = dataset[0]
    input_ids = item["input_ids"]
    labels = item["labels"]
    mask_token_id = 16
    boundary_positions = [0, 1, 8, 9]
    interior_positions = [2, 3, 4, 5, 6, 7]

    assert all(labels[pos].item() == -100 for pos in boundary_positions)
    assert all(input_ids[pos].item() != mask_token_id for pos in boundary_positions)

    supervised = [pos for pos in interior_positions if labels[pos].item() != -100]
    unmasked = [pos for pos in interior_positions if labels[pos].item() == -100]
    assert len(supervised) == 1
    assert unmasked
    assert all(input_ids[pos].item() == mask_token_id for pos in supervised)
    assert all(input_ids[pos].item() != mask_token_id for pos in unmasked)


def test_infill_legacy_supervised_scope_corrupts_supervised_inputs(tmp_path):
    token_dir, audio_dir = _write_fixture(tmp_path)
    random.seed(11)
    torch.manual_seed(11)
    dataset = AudioMotionTokenWindowDataset(
        motion_token_dir=str(token_dir),
        audio_feat_dir=str(audio_dir),
        names=["sample"],
        num_frames=5,
        num_tokens_per_frame=2,
        codebook_size=8,
        mask_prob=1.0,
        random_replace_prob=1.0,
        random_replace_scope="legacy_supervised",
    )

    item = dataset[0]
    input_ids = item["input_ids"]
    labels = item["labels"]
    mask_token_id = 16
    interior_positions = [2, 3, 4, 5, 6, 7]

    assert all(labels[pos].item() != -100 for pos in interior_positions)
    assert all(input_ids[pos].item() != mask_token_id for pos in interior_positions)
