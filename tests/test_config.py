"""Yapılandırma testleri."""

from psikiyatri_ai.config import ModelSettings, Settings


def test_settings_defaults():
    s = Settings()
    assert s.env == "development"
    assert s.raw_data_dir == s.data_dir / "raw"
    assert not s.is_production


def test_model_settings_dims():
    m = ModelSettings()
    assert m.text_embed_dim == 768
    assert m.fusion_dim > 0
    assert m.num_classes >= 2
