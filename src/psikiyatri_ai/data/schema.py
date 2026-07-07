"""Veri sözleşmeleri (data contracts).

Modaliteler arasında dolaşan veri yapılarını tek yerde tanımlar. Ham
dosya yolları yerine bu tipleri kullanmak, katmanları birbirinden
ayrıştırır ve doğrulamayı kolaylaştırır.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class Modality(StrEnum):
    VISION = "vision"
    AUDIO = "audio"
    TEXT = "text"


class DiagnosisLabel(StrEnum):
    """Etiket taksonomisi (deney tasarımına göre genişletilir)."""

    KONTROL = "kontrol"  # sağlıklı birey
    DEPRESYON = "depresyon"
    ANKSIYETE = "anksiyete"
    BIPOLAR = "bipolar"
    SIZOFRENI = "sizofreni"
    BELIRSIZ = "belirsiz"


class Sample(BaseModel):
    """Tek bir seansa ait multimodal örnek (anonimleştirilmiş)."""

    pseudonym: str = Field(..., description="Anonim hasta takma-kimliği")
    session_id: str = Field(..., description="Seans tanımlayıcısı")

    video_path: Path | None = None
    audio_path: Path | None = None
    transcript: str | None = Field(None, description="Maskeli konuşma metni")

    # Önceden çıkarılmış öznitelikler (.npz). Verildiğinde ham medya yerine
    # doğrudan bu dosya okunur (hızlı yol; ağır bağımlılık gerektirmez).
    features_path: Path | None = None

    label: DiagnosisLabel | None = None

    def available_modalities(self) -> list[Modality]:
        mods: list[Modality] = []
        if self.video_path is not None:
            mods.append(Modality.VISION)
        if self.audio_path is not None:
            mods.append(Modality.AUDIO)
        if self.transcript:
            mods.append(Modality.TEXT)
        return mods


__all__ = ["DiagnosisLabel", "Modality", "Sample"]
