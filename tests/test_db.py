"""Veritabanı modeli testleri (bellek-içi SQLite)."""

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from psikiyatri_ai.db.models import (
    AnalysisResult,
    Base,
    DoctorNote,
    Interview,
    Patient,
    Role,
    User,
)


@pytest.fixture()
def session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s
    engine.dispose()


def _seed(s: Session) -> Patient:
    user = User(username="dr.test", role=Role.PSIKIYATRIST.value, password_hash="x")
    s.add(user)
    s.flush()
    patient = Patient(pseudonym="pid_abc123", birth_year=1990, created_by=user.id)
    s.add(patient)
    s.flush()
    interview = Interview(
        patient_id=patient.id,
        session_id="SES-1",
        evaluator_id=user.id,
        transcript_masked="maskeli metin",
        symptom_duration_weeks=6,
        severity="orta",
    )
    s.add(interview)
    s.flush()
    s.add(DoctorNote(interview_id=interview.id, author_id=user.id, content_masked="not"))
    s.add(
        AnalysisResult(
            interview_id=interview.id,
            method="test",
            predicted_label="depresyon",
            risk_level="orta",
            probabilities={"kontrol": 0.3, "depresyon": 0.7},
            decision={"risk": {"level": "orta", "factors": []}},
        )
    )
    s.commit()
    return patient


def test_relationships(session):
    patient = _seed(session)
    assert len(patient.interviews) == 1
    interview = patient.interviews[0]
    assert interview.notes[0].content_masked == "not"
    assert interview.analyses[0].probabilities["depresyon"] == 0.7
    assert interview.analyses[0].decision["risk"]["level"] == "orta"


def test_cascade_delete_erases_all_linked_records(session):
    patient = _seed(session)
    session.delete(patient)
    session.commit()
    assert session.scalar(select(Interview)) is None
    assert session.scalar(select(DoctorNote)) is None
    assert session.scalar(select(AnalysisResult)) is None
    # kullanıcı kaydı silinmez
    assert session.scalar(select(User)) is not None


def test_pseudonym_unique(session):
    _seed(session)
    session.add(Patient(pseudonym="pid_abc123"))
    with pytest.raises(IntegrityError):
        session.commit()
