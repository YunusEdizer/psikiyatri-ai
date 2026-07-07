"""Merkezi yapılandırma.

Tüm ayarlar ortam değişkenleri (`PAI_` öneki) veya `.env` dosyası
üzerinden okunur. Kod içinde sabit yol/parametre kullanmaktan kaçının;
bunun yerine `get_settings()` ile bu nesneye erişin.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ModelSettings(BaseSettings):
    """Modalite ve füzyon modeli hiperparametreleri."""

    # Modalite başına gömme (embedding) boyutu
    vision_embed_dim: int = 256
    audio_embed_dim: int = 256
    text_embed_dim: int = 768  # BERTurk gizli boyutu

    # Füzyon sonrası ortak temsil boyutu
    fusion_dim: int = 512

    # Sınıflandırılacak durum sayısı (ör. kontrol vs. çeşitli tanılar)
    num_classes: int = 2

    dropout: float = 0.3


class Settings(BaseSettings):
    """Uygulama genel ayarları."""

    model_config = SettingsConfigDict(
        env_prefix="PAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: str = "development"
    log_level: str = "INFO"

    # Dizinler
    data_dir: Path = PROJECT_ROOT / "data"
    model_dir: Path = PROJECT_ROOT / "models"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Güvenlik / gizlilik
    anon_salt: str = Field(
        default="degistirin-bu-degeri-uretimde",
        description="Hasta kimliklerini anonimleştirmede kullanılan gizli tuz.",
    )

    # Model hiperparametreleri
    model: ModelSettings = Field(default_factory=ModelSettings)

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def is_production(self) -> bool:
        return self.env.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    """Önbelleğe alınmış tekil ayar nesnesini döndürür."""
    return Settings()
