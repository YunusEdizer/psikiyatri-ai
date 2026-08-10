"""Veritabanı modelleri.

Başvuru iş planındaki tablo kümesinin karşılığı: kullanıcı (rol tabanlı
erişim), hasta, görüşme (seans), doktor notu ve analiz/risk sonucu.

KVKK: `Patient` yalnızca takma-kimlik ve klinik bağlam alanları taşır;
serbest metinler (`*_masked`) yazılmadan önce `mask_pii` ile maskelenir.
Hasta silindiğinde bağlı tüm kayıtlar da silinir (unutulma hakkı).
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Role(enum.StrEnum):
    """Kullanıcı rolleri — yetki düzeyleri panelde ve API'de ayrışır."""

    YONETICI = "yonetici"
    PSIKIYATRIST = "psikiyatrist"
    PSIKOLOG = "psikolog"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(128), default="")
    role: Mapped[str] = mapped_column(String(32), default=Role.PSIKOLOG.value)
    password_hash: Mapped[str] = mapped_column(String(256))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(primary_key=True)
    pseudonym: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    # Klinisyen rumuzu: YALNIZ baş harfler saklanır (to_initials ile zorlanır;
    # ör. "A.Y."). Gerçek ad-soyad asla yazılmaz; pid↔kimlik eşlemesi kliniğin
    # kendi kayıt sisteminde tutulur (pseudonimizasyon anahtarı ayrı ilkesi).
    clinician_label: Mapped[str | None] = mapped_column(String(16), default=None)
    birth_year: Mapped[int | None] = mapped_column(Integer, default=None)
    gender: Mapped[str | None] = mapped_column(String(16), default=None)
    cultural_context: Mapped[str | None] = mapped_column(String(64), default=None)
    referral_note_masked: Mapped[str | None] = mapped_column(Text, default=None)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    interviews: Mapped[list[Interview]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )


class Interview(Base):
    """Görüşme (seans) kaydı ve yapılandırılmış klinik girdiler."""

    __tablename__ = "interviews"

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    evaluator_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), default=None)
    transcript_masked: Mapped[str | None] = mapped_column(Text, default=None)
    symptom_duration_weeks: Mapped[int | None] = mapped_column(Integer, default=None)
    severity: Mapped[str | None] = mapped_column(String(16), default=None)  # hafif|orta|şiddetli
    # KVKK açık rıza kaydı (özellikle kamera/mikrofonlu otomatik görüşme).
    consent: Mapped[bool] = mapped_column(Boolean, default=False)
    consent_at: Mapped[str | None] = mapped_column(String(32), default=None)  # ISO zaman damgası
    # Klinisyen karar döngüsü: uzmanın AI çıktısına ilişkin nihai görüşü.
    clinician_agreement: Mapped[str | None] = mapped_column(
        String(16), default=None  # katiliyorum | kismen | katilmiyorum
    )
    clinician_assessment: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    patient: Mapped[Patient] = relationship(back_populates="interviews")
    notes: Mapped[list[DoctorNote]] = relationship(
        back_populates="interview", cascade="all, delete-orphan"
    )
    analyses: Mapped[list[AnalysisResult]] = relationship(
        back_populates="interview", cascade="all, delete-orphan"
    )


class DoctorNote(Base):
    __tablename__ = "doctor_notes"

    id: Mapped[int] = mapped_column(primary_key=True)
    interview_id: Mapped[int] = mapped_column(ForeignKey("interviews.id"), index=True)
    author_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), default=None)
    content_masked: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    interview: Mapped[Interview] = relationship(back_populates="notes")


class AnalysisResult(Base):
    """Bir görüşmenin model çıktısı + karar destek profili (risk dâhil)."""

    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    interview_id: Mapped[int] = mapped_column(ForeignKey("interviews.id"), index=True)
    method: Mapped[str] = mapped_column(String(160))
    predicted_label: Mapped[str | None] = mapped_column(String(32), default=None)
    risk_level: Mapped[str | None] = mapped_column(String(16), default=None)
    high_risk: Mapped[bool] = mapped_column(Boolean, default=False)
    incongruence: Mapped[float | None] = mapped_column(Float, default=None)
    probabilities: Mapped[dict | None] = mapped_column(JSON, default=None)
    modality_contributions: Mapped[list | None] = mapped_column(JSON, default=None)
    matched_indicators: Mapped[dict | None] = mapped_column(JSON, default=None)
    scored_modalities: Mapped[list | None] = mapped_column(JSON, default=None)
    decision: Mapped[dict | None] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    interview: Mapped[Interview] = relationship(back_populates="analyses")


__all__ = ["AnalysisResult", "Base", "DoctorNote", "Interview", "Patient", "Role", "User"]
