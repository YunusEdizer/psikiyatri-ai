"""Hasta ve görüşme yönetimi uç noktaları (kimlik doğrulama zorunlu).

Klinik akış: hasta kaydı → görüşme (medya/transkript + yapılandırılmış
klinik girdiler) → analiz + karar destek profili → kalıcı rapor.

KVKK:
- Doğrudan tanımlayıcı saklanmaz; ``identifier`` sunucuda takma-kimliğe
  çevrilir ve atılır. Serbest metinler maskelenerek yazılır.
- Ham medya (video/ses/sensör) analiz sonrası silinir, kalıcı saklanmaz.
- ``DELETE /patients/{id}`` unutulma hakkını uygular (kaskatlı silme).
- Tüm erişim/işleme olayları denetim günlüğüne işlenir.
"""

from __future__ import annotations

import secrets

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..data import audit
from ..data.anonymization import mask_pii, pseudonymize
from ..db.database import get_db
from ..db.models import AnalysisResult, DoctorNote, Interview, Patient, Role, User
from ..decision import generate_decision
from ..modalities.text.preprocess import normalize_text
from . import session as session_logic
from .schemas import (
    DoctorNoteCreate,
    DoctorNoteOut,
    InterviewReport,
    InterviewSummary,
    PatientCreate,
    PatientDetail,
    PatientOut,
)
from .security import get_current_user, require_roles
from .uploads import cleanup, collect_uploads

router = APIRouter(tags=["klinik"])

_SEVERITIES = ("", "hafif", "orta", "şiddetli")


# --- yardımcılar ---------------------------------------------------------------


def _get_patient(db: Session, patient_id: int) -> Patient:
    patient = db.get(Patient, patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="Hasta kaydı bulunamadı.")
    return patient


def _latest_analysis(interview: Interview) -> AnalysisResult | None:
    return interview.analyses[-1] if interview.analyses else None


def _evaluator_name(db: Session, user_id: int | None) -> str | None:
    if user_id is None:
        return None
    user = db.get(User, user_id)
    return (user.full_name or user.username) if user else None


def _patient_out(db: Session, patient: Patient) -> PatientOut:
    last_risk = None
    if patient.interviews:
        analysis = _latest_analysis(patient.interviews[-1])
        last_risk = analysis.risk_level if analysis else None
    return PatientOut(
        id=patient.id,
        pseudonym=patient.pseudonym,
        birth_year=patient.birth_year,
        gender=patient.gender,
        cultural_context=patient.cultural_context,
        referral_note_masked=patient.referral_note_masked,
        created_at=patient.created_at.isoformat(timespec="seconds"),
        interview_count=len(patient.interviews),
        last_risk_level=last_risk,
    )


def _interview_summary(db: Session, interview: Interview) -> InterviewSummary:
    analysis = _latest_analysis(interview)
    return InterviewSummary(
        id=interview.id,
        session_id=interview.session_id,
        created_at=interview.created_at.isoformat(timespec="seconds"),
        evaluator=_evaluator_name(db, interview.evaluator_id),
        severity=interview.severity,
        symptom_duration_weeks=interview.symptom_duration_weeks,
        predicted_label=analysis.predicted_label if analysis else None,
        risk_level=analysis.risk_level if analysis else None,
        method=analysis.method if analysis else None,
    )


def _note_out(db: Session, note: DoctorNote) -> DoctorNoteOut:
    return DoctorNoteOut(
        id=note.id,
        author=_evaluator_name(db, note.author_id),
        content_masked=note.content_masked,
        created_at=note.created_at.isoformat(timespec="seconds"),
    )


# --- hasta CRUD -----------------------------------------------------------------


@router.post("/patients", response_model=PatientOut, status_code=201)
def create_patient(
    payload: PatientCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> PatientOut:
    if payload.identifier and payload.identifier.strip():
        pseudonym = pseudonymize(payload.identifier)
    else:
        pseudonym = f"pid_{secrets.token_hex(8)}"
    if db.scalar(select(Patient).where(Patient.pseudonym == pseudonym)) is not None:
        raise HTTPException(
            status_code=409, detail=f"Bu kimlikle kayıtlı hasta zaten var: {pseudonym}"
        )
    patient = Patient(
        pseudonym=pseudonym,
        birth_year=payload.birth_year,
        gender=payload.gender,
        cultural_context=payload.cultural_context,
        referral_note_masked=mask_pii(payload.referral_note) if payload.referral_note else None,
        created_by=user.id,
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    audit.log_event("patient_created", subject=pseudonym, detail={"by": user.username})
    return _patient_out(db, patient)


@router.get("/patients", response_model=list[PatientOut])
def list_patients(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> list[PatientOut]:
    patients = db.scalars(select(Patient).order_by(Patient.created_at.desc())).all()
    audit.log_event("patient_list", detail={"by": user.username, "count": len(patients)})
    return [_patient_out(db, p) for p in patients]


@router.get("/patients/{patient_id}", response_model=PatientDetail)
def get_patient(
    patient_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> PatientDetail:
    patient = _get_patient(db, patient_id)
    audit.log_event("patient_viewed", subject=patient.pseudonym, detail={"by": user.username})
    base = _patient_out(db, patient)
    interviews = [
        _interview_summary(db, i)
        for i in sorted(patient.interviews, key=lambda x: x.created_at, reverse=True)
    ]
    return PatientDetail(**base.model_dump(), interviews=interviews)


@router.delete("/patients/{patient_id}", status_code=204)
def delete_patient(
    patient_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_roles(Role.YONETICI)),
) -> None:
    """KVKK unutulma hakkı: hastayı ve tüm bağlı kayıtları kalıcı siler."""
    patient = _get_patient(db, patient_id)
    pseudonym = patient.pseudonym
    db.delete(patient)  # görüşme/not/analiz kayıtları kaskatlı silinir
    db.commit()
    audit.log_event("patient_erased", subject=pseudonym, detail={"by": admin.username})


# --- görüşme + analiz --------------------------------------------------------------


@router.post(
    "/patients/{patient_id}/interviews",
    response_model=InterviewReport,
    status_code=201,
)
async def create_interview(
    patient_id: int,
    transcript: str = Form(""),
    symptom_duration_weeks: int | None = Form(None, ge=0, le=2600),
    severity: str = Form(""),
    doctor_note: str = Form(""),
    video: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
    wearable: UploadFile | None = File(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> InterviewReport:
    """Görüşme kaydeder: analiz + karar destek profili üretip kalıcılaştırır."""
    if severity not in _SEVERITIES:
        raise HTTPException(
            status_code=422, detail="Geçersiz şiddet düzeyi (hafif/orta/şiddetli)."
        )
    patient = _get_patient(db, patient_id)

    received, media_paths, tmp_files = await collect_uploads(video, audio, wearable)
    try:
        has_transcript = bool(transcript and transcript.strip())
        if not received and not has_transcript:
            raise HTTPException(
                status_code=422,
                detail="En az bir kayıt (video/ses/giyilebilir) veya transkript gerekli.",
            )

        evaluator = user.full_name or user.username
        report = session_logic.analyze_session(
            patient.pseudonym, evaluator, transcript, received, media_paths
        )
        masked_note = mask_pii(normalize_text(doctor_note)) if doctor_note.strip() else None
        decision = generate_decision(
            probabilities=report.get("probabilities"),
            predicted_label=report.get("predicted_label"),
            risk_level=report.get("risk_level"),
            high_risk=bool(report.get("high_risk")),
            incongruence=report.get("incongruence"),
            scored_modalities=report.get("scored_modalities") or [],
            matched_indicators=report.get("matched_indicators") or {},
            trained=bool(report.get("trained")),
            symptom_duration_weeks=symptom_duration_weeks,
            severity=severity or None,
            doctor_note_masked=masked_note,
            has_transcript=has_transcript,
        )
        # Karar motorunun yükselttiği risk düzeyini rapora da yansıt.
        if report.get("analysis_available"):
            report["risk_level"] = decision["risk"]["level"]

        interview = Interview(
            patient_id=patient.id,
            session_id=report["session_id"],
            evaluator_id=user.id,
            transcript_masked=(
                mask_pii(normalize_text(transcript)) if has_transcript else None
            ),
            symptom_duration_weeks=symptom_duration_weeks,
            severity=severity or None,
        )
        db.add(interview)
        db.flush()

        analysis = AnalysisResult(
            interview_id=interview.id,
            method=report.get("method") or "-",
            predicted_label=report.get("predicted_label"),
            risk_level=report.get("risk_level"),
            high_risk=bool(report.get("high_risk")),
            incongruence=report.get("incongruence"),
            probabilities=report.get("probabilities"),
            modality_contributions=report.get("modality_contributions"),
            matched_indicators=report.get("matched_indicators"),
            scored_modalities=report.get("scored_modalities"),
            decision=decision,
        )
        db.add(analysis)

        notes: list[DoctorNote] = []
        if masked_note:
            note = DoctorNote(
                interview_id=interview.id, author_id=user.id, content_masked=masked_note
            )
            db.add(note)
            notes.append(note)

        db.commit()
        db.refresh(interview)
        audit.log_event(
            "interview_created",
            subject=patient.pseudonym,
            detail={
                "by": user.username,
                "interview_id": interview.id,
                "modalities": report.get("modalities"),
            },
        )

        return InterviewReport(
            **report,
            interview_id=interview.id,
            patient_id=patient.id,
            patient_pseudonym=patient.pseudonym,
            symptom_duration_weeks=symptom_duration_weeks,
            severity=severity or None,
            decision=decision,
            notes=[_note_out(db, n) for n in notes],
        )
    finally:
        cleanup(tmp_files)


@router.post(
    "/patients/{patient_id}/interviews/auto",
    response_model=InterviewReport,
    status_code=201,
)
async def create_auto_interview(
    patient_id: int,
    answers: str = Form("[]"),
    video: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> InterviewReport:
    """Otomatik (seçenekli) görüşmeyi hasta kaydına kalıcı olarak işler.

    ``answers`` = ``[{"id","selected","other"}, ...]``. Belirti yanıtları
    kanonik transkripte çevrilir; ön bilgi (ÖB) bağlam olarak, kritik
    yanıtlar hem raporda hem otomatik bir klinisyen-notu olarak saklanır.
    """
    import json  # noqa: PLC0415

    from ..data.interview_questions import (  # noqa: PLC0415
        build_interview_transcript,
        build_preinfo_summary,
        critical_flags,
    )

    try:
        parsed = json.loads(answers)
        if not isinstance(parsed, list):
            raise ValueError("liste bekleniyor")
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"'answers' geçersiz: {exc}") from exc

    patient = _get_patient(db, patient_id)
    transcript = build_interview_transcript(parsed)
    preinfo = build_preinfo_summary(parsed)
    flags = critical_flags(parsed)

    received, media_paths, tmp_files = await collect_uploads(video, audio, None)
    try:
        if not received and not transcript and not preinfo:
            raise HTTPException(status_code=422, detail="En az bir yanıt veya kayıt gerekli.")

        evaluator = user.full_name or user.username
        report = session_logic.analyze_session(
            patient.pseudonym, evaluator, transcript, received, media_paths
        )
        decision = generate_decision(
            probabilities=report.get("probabilities"),
            predicted_label=report.get("predicted_label"),
            risk_level=report.get("risk_level"),
            high_risk=bool(report.get("high_risk")),
            incongruence=report.get("incongruence"),
            scored_modalities=report.get("scored_modalities") or [],
            matched_indicators=report.get("matched_indicators") or {},
            trained=bool(report.get("trained")),
            has_transcript=bool(transcript),
        )
        if report.get("analysis_available"):
            report["risk_level"] = decision["risk"]["level"]
        report["preinfo"] = preinfo
        report["critical_flags"] = flags

        interview = Interview(
            patient_id=patient.id,
            session_id=report["session_id"],
            evaluator_id=user.id,
            # TAM maskeli transkripti sakla (transcript_preview yalnızca 500
            # karakterlik görüntüleme kesitidir; kalıcı kayda tamamı yazılır).
            transcript_masked=(
                mask_pii(normalize_text(transcript)) if transcript else None
            ),
        )
        db.add(interview)
        db.flush()

        # Ön bilgi + kritik bayrakları karar profiline iliştir (kalıcı, JSON).
        decision_stored = {**decision, "preinfo": preinfo, "critical_flags": flags}
        db.add(AnalysisResult(
            interview_id=interview.id,
            method=report.get("method") or "-",
            predicted_label=report.get("predicted_label"),
            risk_level=report.get("risk_level"),
            high_risk=bool(report.get("high_risk")),
            incongruence=report.get("incongruence"),
            probabilities=report.get("probabilities"),
            modality_contributions=report.get("modality_contributions"),
            matched_indicators=report.get("matched_indicators"),
            scored_modalities=report.get("scored_modalities"),
            decision=decision_stored,
        ))

        # Kritik yanıtlar → otomatik klinisyen notu (kayıtta görünür bıraksın).
        notes: list[DoctorNote] = []
        if flags:
            summary = "SİSTEM UYARISI — uzman değerlendirmesi gerektiren yanıtlar:\n" + "\n".join(
                f"- {f['question']} → {f['answer']} ({f['warn']})" for f in flags
            )
            note = DoctorNote(interview_id=interview.id, author_id=user.id, content_masked=summary)
            db.add(note)
            notes.append(note)

        db.commit()
        db.refresh(interview)
        audit.log_event(
            "auto_interview_created",
            subject=patient.pseudonym,
            detail={"by": user.username, "interview_id": interview.id, "flags": len(flags)},
        )

        return InterviewReport(
            **report,
            interview_id=interview.id,
            patient_id=patient.id,
            patient_pseudonym=patient.pseudonym,
            decision=decision,
            notes=[_note_out(db, n) for n in notes],
        )
    finally:
        cleanup(tmp_files)


@router.get("/interviews/{interview_id}", response_model=InterviewReport)
def get_interview(
    interview_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> InterviewReport:
    """Kalıcı görüşme raporunu getirir (ham medya saklanmadığından meta yoktur)."""
    interview = db.get(Interview, interview_id)
    if interview is None:
        raise HTTPException(status_code=404, detail="Görüşme kaydı bulunamadı.")
    patient = interview.patient
    analysis = _latest_analysis(interview)
    audit.log_event(
        "interview_viewed", subject=patient.pseudonym, detail={"by": user.username}
    )

    modalities = list(analysis.scored_modalities or []) if analysis else []
    if interview.transcript_masked and "text" not in modalities:
        modalities.append("text")
    # Otomatik görüşmede ön bilgi + kritik bayraklar decision JSON'una gömülür;
    # yeniden yüklemede yapılandırılmış alanlara geri çıkar (aksi hâlde
    # DecisionProfile şeması bu fazladan anahtarları eler ve bağlam kaybolur).
    stored_decision = (analysis.decision or {}) if analysis else {}
    preinfo = stored_decision.get("preinfo", []) if isinstance(stored_decision, dict) else []
    critical = (
        stored_decision.get("critical_flags", []) if isinstance(stored_decision, dict) else []
    )
    return InterviewReport(
        session_id=interview.session_id,
        created_at=interview.created_at.isoformat(timespec="seconds"),
        pseudonym=patient.pseudonym,
        evaluator=_evaluator_name(db, interview.evaluator_id) or "-",
        received=[],
        modalities=modalities,
        analysis_available=analysis is not None and analysis.predicted_label is not None,
        method=analysis.method if analysis else "-",
        high_risk=bool(analysis.high_risk) if analysis else False,
        probabilities=analysis.probabilities if analysis else None,
        predicted_label=analysis.predicted_label if analysis else None,
        risk_level=analysis.risk_level if analysis else None,
        matched_indicators=(analysis.matched_indicators or {}) if analysis else {},
        transcript_preview=(
            interview.transcript_masked[:500] if interview.transcript_masked else None
        ),
        incongruence=analysis.incongruence if analysis else None,
        modality_contributions=(analysis.modality_contributions or []) if analysis else [],
        scored_modalities=(analysis.scored_modalities or []) if analysis else [],
        disclaimer=session_logic.DISCLAIMER,
        prototype_note=session_logic.PROTOTYPE_NOTE,
        interview_id=interview.id,
        patient_id=patient.id,
        patient_pseudonym=patient.pseudonym,
        symptom_duration_weeks=interview.symptom_duration_weeks,
        severity=interview.severity,
        decision=analysis.decision if analysis else None,
        preinfo=preinfo,
        critical_flags=critical,
        notes=[_note_out(db, n) for n in interview.notes],
    )


@router.post("/interviews/{interview_id}/notes", response_model=DoctorNoteOut, status_code=201)
def add_doctor_note(
    interview_id: int,
    payload: DoctorNoteCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> DoctorNoteOut:
    interview = db.get(Interview, interview_id)
    if interview is None:
        raise HTTPException(status_code=404, detail="Görüşme kaydı bulunamadı.")
    note = DoctorNote(
        interview_id=interview.id,
        author_id=user.id,
        content_masked=mask_pii(normalize_text(payload.content)),
    )
    db.add(note)
    db.commit()
    db.refresh(note)
    audit.log_event(
        "doctor_note_added",
        subject=interview.patient.pseudonym,
        detail={"by": user.username, "interview_id": interview.id},
    )
    return _note_out(db, note)


# --- denetim günlüğü ----------------------------------------------------------------


@router.get("/audit/events")
def audit_events(
    limit: int = 100,
    _admin: User = Depends(require_roles(Role.YONETICI)),
) -> list[dict]:
    """Son denetim olaylarını döndürür (yalnızca yönetici)."""
    events = audit.read_events()
    return events[-max(1, min(limit, 1000)) :][::-1]


__all__ = ["router"]
