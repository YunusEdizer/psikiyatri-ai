"""Otomatik görüşme uçları (çoktan seçmeli protokol).

Yönlendirmeli görüşme akışı: panel soruları ``GET /interview/questions``
ile çeker (ÖB1-ÖB3 ön bilgi + Q1-Q24 belirti; her biri çoktan seçmeli
şıklar + serbest açıklama + gerekiyorsa kritik uyarı taşır). Danışanın
seçimlerini/açıklamalarını toplar ve kayıtlarla birlikte
``POST /interview/analyze``'a gönderir. Yanıtlar sunucuda kanonik
transkripte çevrilir; ön bilgi (ÖB) tanı modeline verilmez, raporda
bağlam olarak sunulur. Soru seti statiktir ve hasta verisi içermez.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from ..data.interview_questions import (
    INTERVIEW_QUESTIONS,
    build_interview_transcript,
    build_preinfo_summary,
    critical_flags,
)
from . import session
from .schemas import SessionAnalysisResponse
from .uploads import cleanup, collect_uploads

router = APIRouter(prefix="/interview", tags=["görüşme"])


class InterviewQuestionOut(BaseModel):
    id: int
    key: str
    text: str
    category: str
    section: str
    options: list[str]
    warn: str | None = None


class InterviewQuestionsResponse(BaseModel):
    count: int
    questions: list[InterviewQuestionOut]
    note: str


class TranscriptionResponse(BaseModel):
    text: str
    engine: str  # "whisper-yerel" | "yok"


@router.get("/questions", response_model=InterviewQuestionsResponse)
def list_questions() -> InterviewQuestionsResponse:
    """Görüşme protokolündeki soruları (şıklarıyla) sıralı döndürür."""
    return InterviewQuestionsResponse(
        count=len(INTERVIEW_QUESTIONS),
        questions=[InterviewQuestionOut(**q) for q in INTERVIEW_QUESTIONS],
        note=(
            "Bu sorular tanı koymaz; DSM-5 ana belirti kümelerini tarayan "
            "yarı yapılandırılmış bir ön görüşme protokolüdür. Ön bilgi (ÖB) "
            "soruları bağlam amaçlıdır. Nihai değerlendirme yetkili sağlık "
            "uzmanına aittir; sistem ilaç değişikliği önermez."
        ),
    )


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(audio: UploadFile = File(...)) -> TranscriptionResponse:
    """Kısa bir ses klibini yerelde (sunucuda) Türkçe metne çevirir.

    Tarayıcının bulut STT hizmeti yerine faster-whisper kullanılır; ham ses
    üçüncü tarafa gitmez (KVKK). faster-whisper kurulu değilse boş metin +
    ``engine="yok"`` döner; istemci tarayıcı STT'sine geri düşebilir. Ses
    geçici dosyaya yazılır ve işlem sonrası silinir.
    """
    from ..modalities.audio.transcribe import is_available, transcribe_audio

    if not is_available():
        return TranscriptionResponse(text="", engine="yok")

    received, media_paths, tmp_files = await collect_uploads(None, audio, None)
    try:
        path = media_paths.get("audio")
        if path is None:
            raise HTTPException(status_code=422, detail="Ses dosyası gerekli.")
        text = transcribe_audio(path) or ""
        return TranscriptionResponse(text=text, engine="whisper-yerel")
    finally:
        cleanup(tmp_files)


@router.post("/analyze", response_model=SessionAnalysisResponse)
async def analyze_interview(
    pseudonym: str = Form(""),
    evaluator: str = Form(""),
    answers: str = Form("[]"),
    video: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
) -> SessionAnalysisResponse:
    """Görüşme yanıtlarını + kayıtları analiz edip rapor döndürür.

    ``answers`` alanı ``[{"id", "selected", "other"}, ...]`` biçiminde JSON
    dizesidir (eski ``{"id", "answer"}`` serbest metin de kabul edilir);
    sunucuda kanonik görüşme transkriptine çevrilir. Ön bilgi (ÖB) yanıtları
    raporda bağlam olarak eklenir, tanı modeline verilmez. Uyarılı sorulara
    verilen kritik yanıtlar için uzman bayrağı üretilir. Kayıtlar geçici
    dosyaya yazılır, analiz sonrası silinir; kalıcı kayıt tutulmaz.
    """
    try:
        parsed = json.loads(answers)
        if not isinstance(parsed, list):
            raise ValueError("liste bekleniyor")
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail=f"'answers' geçersiz JSON: {exc}"
        ) from exc

    transcript = build_interview_transcript(parsed)
    preinfo = build_preinfo_summary(parsed)
    flags = critical_flags(parsed)
    received, media_paths, tmp_files = await collect_uploads(video, audio, None)
    try:
        if not received and not transcript and not preinfo:
            raise HTTPException(
                status_code=422,
                detail="En az bir yanıt veya kayıt (video/ses) gerekli.",
            )
        report = session.analyze_session(
            pseudonym, evaluator, transcript, received, media_paths
        )
        report["preinfo"] = preinfo
        report["critical_flags"] = flags
        return SessionAnalysisResponse(**report)
    finally:
        cleanup(tmp_files)
