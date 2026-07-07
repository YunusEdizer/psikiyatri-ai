"""Füzyon modeli ve uyumsuzluk metriği testleri."""

import torch

from psikiyatri_ai.config import ModelSettings
from psikiyatri_ai.fusion.incongruence import pairwise_incongruence
from psikiyatri_ai.fusion.multimodal import MultimodalFusion


def _dummy_embeddings(batch: int = 4) -> dict[str, torch.Tensor]:
    cfg = ModelSettings()
    return {
        "vision": torch.randn(batch, cfg.vision_embed_dim),
        "audio": torch.randn(batch, cfg.audio_embed_dim),
        "text": torch.randn(batch, cfg.text_embed_dim),
    }


def test_fusion_output_shapes():
    cfg = ModelSettings()
    model = MultimodalFusion(cfg)
    out = model(_dummy_embeddings())
    assert out.logits.shape == (4, cfg.num_classes)
    assert out.incongruence.shape == (4,)
    assert out.attention.shape == (4, 3)


def test_fusion_handles_missing_modality():
    cfg = ModelSettings()
    model = MultimodalFusion(cfg)
    emb = _dummy_embeddings()
    del emb["vision"]  # yalnızca ses + metin
    out = model(emb)
    assert out.logits.shape == (4, cfg.num_classes)
    assert out.attention.shape == (4, 2)


def test_incongruence_identical_is_low():
    e = torch.randn(3, 8)
    score = pairwise_incongruence({"a": e, "b": e.clone()})
    assert torch.allclose(score, torch.zeros(3), atol=1e-5)


def test_incongruence_single_modality_is_zero():
    score = pairwise_incongruence({"a": torch.randn(3, 8)})
    assert torch.allclose(score, torch.zeros(3))
