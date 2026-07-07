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


class DemoRequest(BaseModel):
    """Görsel demo arayüzünden gelen istek.

    Ham gömme yerine yüksek seviyeli seçimler alınır; sunucu senaryoya
    uygun sentetik bir seans üretip tam modelden geçirir.
    """

    scenario: str = Field("rastgele", description="'kontrol' | 'depresyon' | 'rastgele'")
    signal: float = Field(0.8, ge=0.0, le=2.0, description="Sınıf sinyali gücü")
    seed: int = Field(0, ge=0, description="Tekrarüretilebilirlik tohumu")
    modalities: list[str] = Field(
        default_factory=lambda: ["vision", "audio"],
        description="Etkin modaliteler",
    )


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


class ReceivedFile(BaseModel):
    modality: str
    filename: str
    size_kb: float


class SessionAnalysisResponse(BaseModel):
    """Klinik seans değerlendirme raporu (yükleme akışı)."""

    session_id: str
    created_at: str
    pseudonym: str
    evaluator: str
    received: list[ReceivedFile]
    modalities: list[str]
    analysis_available: bool
    method: str
    high_risk: bool = False
    probabilities: dict[str, float] | None = None
    predicted_label: str | None = None
    risk_level: str | None = None
    matched_indicators: dict[str, list[str]] = Field(default_factory=dict)
    transcript_preview: str | None = None
    note: str | None = None
    disclaimer: str
    prototype_note: str


class HealthResponse(BaseModel):
    status: str
    version: str
    env: str


__all__ = [
    "DemoRequest",
    "HealthResponse",
    "ModalityContribution",
    "PredictRequest",
    "PredictResponse",
    "ReceivedFile",
    "SessionAnalysisResponse",
]
