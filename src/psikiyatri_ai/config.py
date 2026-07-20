"""Merkezi yapılandırma.

Tüm ayarlar ortam değişkenleri (`PAI_` öneki) veya `.env` dosyası
üzerinden okunur. Kod içinde sabit yol/parametre kullanmaktan kaçının;
bunun yerine `get_settings()` ile bu nesneye erişin.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Üretimde mutlaka değiştirilmesi gereken güvensiz varsayılan tuz.
DEFAULT_ANON_SALT = "degistirin-bu-degeri-uretimde"

# Üretimde mutlaka değiştirilmesi gereken güvensiz varsayılan JWT anahtarı.
DEFAULT_SECRET_KEY = "degistirin-jwt-anahtarini-uretimde"

# Üretimde mutlaka değiştirilmesi gereken varsayılan yönetici parolası.
DEFAULT_ADMIN_PASSWORD = "degistir123"


class ModelSettings(BaseSettings):
    """Modalite ve füzyon modeli hiperparametreleri."""

    # Modalite başına gömme (embedding) boyutu
    vision_embed_dim: int = 256
    audio_embed_dim: int = 256
    text_embed_dim: int = 768  # BERTurk gizli boyutu
    wearable_embed_dim: int = 128  # giyilebilir sensör gömme boyutu

    # Metin (BERTurk) ince ayarı: donuk gövdede son N transformer katmanını
    # eğitime aç (0 = tam donuk; sadece proje/füzyon başları öğrenir).
    text_unfreeze_layers: int = 0

    # Füzyon sonrası ortak temsil boyutu
    fusion_dim: int = 512

    # Sınıflandırılacak durum sayısı (ör. kontrol vs. çeşitli tanılar).
    # 2 = ikili tarama; >2 = çok-hastalık (kontrol + depresyon/anksiyete/
    # bipolar/şizofreni). Deney tasarımına/manifest etiketlerine göre artırılır.
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
        default=DEFAULT_ANON_SALT,
        description="Hasta kimliklerini anonimleştirmede kullanılan gizli tuz.",
    )
    # Denetim (audit) günlüğü yolu (None => <data>/audit/audit.jsonl)
    audit_log_path: Path | None = None

    # Veritabanı bağlantısı (None => <data>/psikiyatri.db SQLite dosyası).
    # Üretimde PostgreSQL önerilir: postgresql+psycopg://user:pass@host/db
    database_url: str | None = None

    # Kimlik doğrulama (JWT)
    secret_key: str = Field(
        default=DEFAULT_SECRET_KEY,
        description="JWT imzalama anahtarı; üretimde güçlü rastgele değer verin.",
    )
    access_token_expire_minutes: int = 480  # bir klinik mesai günü

    # İlk açılışta oluşturulan yönetici hesabı (kullanıcı tablosu boşsa)
    admin_username: str = "admin"
    admin_password: str = Field(
        default=DEFAULT_ADMIN_PASSWORD,
        description="İlk yönetici parolası; üretimde mutlaka değiştirin.",
    )

    # Model hiperparametreleri
    model: ModelSettings = Field(default_factory=ModelSettings)

    @model_validator(mode="after")
    def _enforce_production_secret(self) -> Settings:
        """Üretim ortamında güvensiz varsayılanları reddeder."""
        if self.env.lower() != "production":
            return self
        insecure = {
            "PAI_ANON_SALT": self.anon_salt == DEFAULT_ANON_SALT,
            "PAI_SECRET_KEY": self.secret_key == DEFAULT_SECRET_KEY,
            "PAI_ADMIN_PASSWORD": self.admin_password == DEFAULT_ADMIN_PASSWORD,
        }
        bad = [name for name, is_default in insecure.items() if is_default]
        if bad:
            raise ValueError(
                "Üretimde şu ayarlar güçlü gizli değerlere ayarlanmalıdır "
                f"(varsayılanları güvensizdir): {', '.join(bad)}"
            )
        return self

    @property
    def resolved_database_url(self) -> str:
        """Etkin veritabanı bağlantı dizesi (varsayılan: yerel SQLite)."""
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.data_dir / 'psikiyatri.db').as_posix()}"

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
