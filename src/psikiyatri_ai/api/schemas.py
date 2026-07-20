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
    wearable_embedding: list[float] | None = None


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


class AcousticObservation(BaseModel):
    """Betimsel akustik gözlem (tanı değil)."""

    duration_s: float
    pitch_hz: float | None = None
    pitch_variability_hz: float | None = None
    loudness_rms: float
    voiced_ratio: float
    qualitative: dict[str, str] = Field(default_factory=dict)


class FaceObservation(BaseModel):
    """Betimsel yüz gözlemi (tanı değil)."""

    frames_analyzed: int
    face_detected_ratio: float
    blink: float | None = None
    expressivity: float | None = None
    smile: float | None = None
    brow_lower: float | None = None
    qualitative: dict[str, str] = Field(default_factory=dict)


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
    # Ses/görüntü/giyilebilir derin öğrenme skorlaması yapıldıysa doldurulur
    incongruence: float | None = None
    modality_contributions: list[ModalityContribution] = Field(default_factory=list)
    scored_modalities: list[str] = Field(default_factory=list)
    # Betimsel gözlemler (tanı füzyonundan bağımsız): kaydedilen sesten
    # yorumlanabilir akustik ölçütler; görüntü yüz analizi beklemede ise işaret.
    audio_observation: AcousticObservation | None = None
    vision_observation: FaceObservation | None = None
    vision_pending: bool = False
    # Otomatik görüşme: ön bilgi (ÖB) bağlamı + uzman uyarısı gerektiren yanıtlar.
    preinfo: list[dict] = Field(default_factory=list)
    critical_flags: list[dict] = Field(default_factory=list)
    note: str | None = None
    disclaimer: str
    prototype_note: str


class HealthResponse(BaseModel):
    status: str
    version: str
    env: str


# --- Kimlik doğrulama / kullanıcı yönetimi -----------------------------------


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    full_name: str
    role: str


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=8)
    full_name: str = ""
    role: str = Field("psikolog", pattern="^(yonetici|psikiyatrist|psikolog)$")


class UserOut(BaseModel):
    id: int
    username: str
    full_name: str
    role: str
    is_active: bool


# --- Hasta / görüşme yönetimi -------------------------------------------------


class PatientCreate(BaseModel):
    """Hasta kaydı — doğrudan tanımlayıcı istenmez.

    ``identifier`` verilirse sunucuda geri döndürülemez takma-kimliğe
    çevrilir ve **saklanmaz**; verilmezse rastgele takma-kimlik üretilir.
    """

    identifier: str | None = Field(
        None, description="Kurum içi kimlik (saklanmaz; takma-kimliğe çevrilir)"
    )
    birth_year: int | None = Field(None, ge=1900, le=2026)
    gender: str | None = None
    cultural_context: str | None = None
    referral_note: str | None = Field(None, description="Sevk/başvuru notu (maskelenir)")


class PatientOut(BaseModel):
    id: int
    pseudonym: str
    birth_year: int | None = None
    gender: str | None = None
    cultural_context: str | None = None
    referral_note_masked: str | None = None
    created_at: str
    interview_count: int = 0
    last_risk_level: str | None = None


class InterviewSummary(BaseModel):
    id: int
    session_id: str
    created_at: str
    evaluator: str | None = None
    severity: str | None = None
    symptom_duration_weeks: int | None = None
    predicted_label: str | None = None
    risk_level: str | None = None
    method: str | None = None


class PatientDetail(PatientOut):
    interviews: list[InterviewSummary] = Field(default_factory=list)


class DoctorNoteCreate(BaseModel):
    content: str = Field(..., min_length=1, description="Not içeriği (maskelenir)")


class DoctorNoteOut(BaseModel):
    id: int
    author: str | None = None
    content_masked: str
    created_at: str


# --- Karar destek profili -------------------------------------------------------


class DiagnosisEntry(BaseModel):
    label: str
    label_tr: str
    probability: float
    support: list[str] = Field(default_factory=list)


class DecisionRisk(BaseModel):
    level: str
    factors: list[str] = Field(default_factory=list)


class DecisionConfidence(BaseModel):
    level: str
    warnings: list[str] = Field(default_factory=list)


class DecisionProfile(BaseModel):
    """Açıklanabilir karar destek profili (kural tabanlı motor çıktısı)."""

    diagnosis_profile: list[DiagnosisEntry] = Field(default_factory=list)
    risk: DecisionRisk
    confidence: DecisionConfidence
    clinical_findings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    disclaimer: str


class InterviewReport(SessionAnalysisResponse):
    """Kalıcı görüşme raporu: seans analizi + karar destek profili."""

    interview_id: int
    patient_id: int
    patient_pseudonym: str
    symptom_duration_weeks: int | None = None
    severity: str | None = None
    decision: DecisionProfile | None = None
    notes: list[DoctorNoteOut] = Field(default_factory=list)


__all__ = [
    "DecisionConfidence",
    "DecisionProfile",
    "DecisionRisk",
    "DemoRequest",
    "DiagnosisEntry",
    "DoctorNoteCreate",
    "DoctorNoteOut",
    "HealthResponse",
    "InterviewReport",
    "InterviewSummary",
    "ModalityContribution",
    "PatientCreate",
    "PatientDetail",
    "PatientOut",
    "PredictRequest",
    "PredictResponse",
    "ReceivedFile",
    "SessionAnalysisResponse",
    "TokenResponse",
    "UserCreate",
    "UserOut",
]
