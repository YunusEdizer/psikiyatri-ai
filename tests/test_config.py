"""Yapılandırma testleri."""

import pytest

from psikiyatri_ai.config import DEFAULT_ANON_SALT, ModelSettings, Settings


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
    assert m.wearable_embed_dim > 0


def test_production_rejects_default_salt():
    with pytest.raises(ValueError, match="PAI_ANON_SALT"):
        Settings(env="production", anon_salt=DEFAULT_ANON_SALT)


def test_production_rejects_default_jwt_and_admin_password():
    with pytest.raises(ValueError, match="PAI_SECRET_KEY"):
        Settings(env="production", anon_salt="guclu-tuz")
    with pytest.raises(ValueError, match="PAI_ADMIN_PASSWORD"):
        Settings(env="production", anon_salt="guclu-tuz", secret_key="guclu-jwt")


def test_production_ok_with_custom_secrets():
    s = Settings(
        env="production",
        anon_salt="guclu-gizli-tuz-123",
        secret_key="guclu-jwt-anahtari",
        admin_password="guclu-parola-42",
    )
    assert s.is_production


def test_resolved_database_url_defaults_to_sqlite():
    s = Settings()
    assert s.resolved_database_url.startswith("sqlite:///")
    assert s.resolved_database_url.endswith("psikiyatri.db")


def test_resolved_database_url_respects_override():
    s = Settings(database_url="postgresql+psycopg://u:p@host/db")
    assert s.resolved_database_url == "postgresql+psycopg://u:p@host/db"
