"""Veri modülü testleri: etiket kodlama, sentetik dataset, collate."""

import pytest
import torch
from torch.utils.data import DataLoader

from psikiyatri_ai.data.batch import SampleFeatures
from psikiyatri_ai.data.datamodule import (
    LabelEncoder,
    SyntheticDataset,
    collate_features,
)


def test_label_encoder_roundtrip():
    enc = LabelEncoder(["kontrol", "depresyon"])
    assert enc.encode("kontrol") == 0
    assert enc.decode(1) == "depresyon"
    assert len(enc) == 2


def test_label_encoder_rejects_unknown():
    enc = LabelEncoder(["kontrol", "depresyon"])
    with pytest.raises(KeyError):
        enc.encode("bilinmeyen")


def test_synthetic_dataset_shapes():
    ds = SyntheticDataset(size=4, modalities=("vision", "audio"), seed=0)
    item = ds[0]
    assert isinstance(item, SampleFeatures)
    assert item.vision.shape == (8, 468, 3)
    assert item.audio_mfcc.shape == (50, 40)
    assert item.audio_egemaps.shape == (88,)


def test_collate_pads_and_batches():
    ds = SyntheticDataset(size=6, modalities=("vision", "audio"), seed=0)
    loader = DataLoader(ds, batch_size=3, collate_fn=collate_features)
    batch, labels = next(iter(loader))
    assert labels.shape == (3,)
    assert batch.vision.shape[0] == 3
    assert batch.audio_mfcc.shape == (3, 50, 40)
    assert batch.audio_egemaps.shape == (3, 88)
    assert set(batch.present_modalities) == {"vision", "audio"}


def test_collate_rejects_mixed_modalities():
    a = SampleFeatures(label=0, vision=torch.randn(2, 468, 3))
    b = SampleFeatures(label=1)  # vision eksik
    with pytest.raises(ValueError, match="Karışık modalite"):
        collate_features([a, b])
