"""API istek/yanıt şemaları (pydantic)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Karar destek isteği.

    Ham medya yerine, önceden çıkarılmış modalite gömmeleri gönderilir.
    (Ham medya yükleme uç noktası ayrı bir sürümde eklenecektir.)
    """

    session_id: str = Field(..., description="Anonim seans tanımlayıcısı")
    vision_embedding: list[float] | None = None
    audio_embedding: list[float] | None = None
    text_embedding: list[float] | None = None


class ModalityContribution(BaseModel):
    modality: str
    weight: float = Field(..., ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    """Karar destek yanıtı.

    ``probabilities`` sınıf olasılıklarıdır. ``incongruence`` modaliteler
    arası uyumsuzluk skorudur (yüksek = ses/yüz/içerik çelişkisi).
    Yanıt **teşhis değildir**; uzmana yardımcı bir sinyaldir.
    """

    session_id: str
    probabilities: dict[str, float]
    predicted_label: str
    incongruence: float = Field(..., ge=0.0, le=1.0)
    modality_contributions: list[ModalityContribution]
    disclaimer: str = (
        "Bu çıktı klinik karar desteği amaçlıdır ve kesin teşhis yerine "
        "geçmez. Nihai karar yetkili sağlık uzmanına aittir."
    )


class HealthResponse(BaseModel):
    status: str
    version: str
    env: str


__all__ = [
    "HealthResponse",
    "ModalityContribution",
    "PredictRequest",
    "PredictResponse",
]
