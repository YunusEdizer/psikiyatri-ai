"""FastAPI uygulaması.

Uç noktalar:
- ``GET  /health``           : servis sağlık kontrolü
- ``POST /predict``          : modalite gömmelerinden karar destek çıktısı
- ``POST /session/analyze``  : kayıtsız hızlı seans analizi (demo)
- ``/auth/*``                : giriş + kullanıcı yönetimi (`routes_auth`)
- ``/patients/*``            : klinik hasta/görüşme akışı (`routes_patients`)

Model, uygulama yaşam döngüsünde bir kez oluşturulur. Eğitilmiş ağırlıklar
mevcutsa ``PAI_MODEL_DIR`` altından yüklenir; yoksa rastgele başlatılmış
model ile (geliştirme modunda) çalışır ve uyarı loglar. Açılışta veritabanı
tabloları oluşturulur ve (tablo boşsa) ilk yönetici hesabı tohumlanır.
"""

from __future__ import annotations

from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy.orm import Session

from .. import __version__
from ..config import get_settings
from ..data.schema import DiagnosisLabel
from ..db.database import get_engine, init_db
from ..logging_conf import logger, setup_logging
from ..training.checkpoint import labels_for_state, load_metadata
from ..training.model import PsikiyatriModel
from . import demo, routes_auth, routes_patients, session
from .schemas import (
    DemoRequest,
    HealthResponse,
    ModalityContribution,
    PredictRequest,
    PredictResponse,
    SessionAnalysisResponse,
)
from .security import ensure_admin_user
from .uploads import cleanup, collect_uploads

_STATIC_DIR = Path(__file__).parent / "static"

# Eğitilmiş checkpoint yoksa ikili varsayılan etiket eşlemesi
_DEFAULT_LABELS = [DiagnosisLabel.KONTROL.value, DiagnosisLabel.DEPRESYON.value]

_state: dict = {}


app = FastAPI(
    title="Psikiyatri AI — Karar Destek API",
    description="Multimodal derin öğrenme ile psikiyatrik tanı karar desteği (Eurotech-1).",
    version=__version__,
)

# Register routers BEFORE any direct routes
app.include_router(routes_auth.router)
app.include_router(routes_patients.router)


@app.on_event("startup")
async def startup_event():
    logger.info("=== STARTUP EVENT CALLED ===")
    logger.info(f"App routes before startup: {len(app.routes)}")
    setup_logging()
    settings = get_settings()
    # Veritabanı: tablolar + (boşsa) ilk yönetici hesabı
    init_db()
    with Session(get_engine()) as db:
        ensure_admin_user(db)
    # Etiket sırası ve sınıf sayısı: önce classes.json yan-verisi; yoksa
    # checkpoint'in sınıflandırıcı başından türetilir (boyut uyuşmazlığı ile
    # yükleme çökmesini önlemek için); o da yoksa ikili varsayılana düşülür.
    weights = settings.model_dir / "psikiyatri_model.pt"
    state = torch.load(weights, map_location="cpu") if weights.exists() else None
    meta = load_metadata(settings.model_dir)
    if meta:
        labels = list(meta["classes"])
    elif state is not None:
        labels = labels_for_state(state, _DEFAULT_LABELS)
    else:
        labels = list(_DEFAULT_LABELS)
    model_cfg = settings.model.model_copy(update={"num_classes": len(labels)})
    # Servis yolu önceden hesaplanmış gömmeleri alır; yalnızca füzyon başı
    # gerekir. Kodlayıcıları oluşturmayarak ağır bağımlılıklardan kaçınırız.
    model = PsikiyatriModel(model_cfg, enabled_modalities=())
    if state is not None:
        # strict=False: tam checkpoint kodlayıcı ağırlıklarını da içerir;
        # burada yalnızca füzyon anahtarları yüklenir.
        model.load_state_dict(state, strict=False)
        logger.info("Eğitilmiş füzyon ağırlıkları yüklendi: {} | sınıflar: {}", weights, labels)
    else:
        logger.warning(
            "Eğitilmiş ağırlık bulunamadı ({}); rastgele başlatılmış model "
            "kullanılıyor. Üretimde kullanmayın.",
            weights,
        )
    model.eval()
    _state["model"] = model
    _state["labels"] = labels


@app.on_event("shutdown")
def shutdown_event():
    _state.clear()


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Klinik panel (PWA)."""
    html = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/manifest.webmanifest", include_in_schema=False)
def manifest() -> FileResponse:
    return FileResponse(
        _STATIC_DIR / "manifest.webmanifest", media_type="application/manifest+json"
    )


@app.get("/sw.js", include_in_schema=False)
def service_worker() -> FileResponse:
    return FileResponse(_STATIC_DIR / "sw.js", media_type="application/javascript")


@app.get("/icon.svg", include_in_schema=False)
def app_icon() -> FileResponse:
    return FileResponse(_STATIC_DIR / "icon.svg", media_type="image/svg+xml")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(status="ok", version=__version__, env=settings.env)


@app.post("/session/analyze", response_model=SessionAnalysisResponse)
async def session_analyze(
    pseudonym: str = Form(""),
    evaluator: str = Form(""),
    transcript: str = Form(""),
    video: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
    wearable: UploadFile | None = File(None),
) -> SessionAnalysisResponse:
    """Klinik yükleme akışı: kayıtları alır, işler ve rapor döndürür.

    Ses/görüntü/giyilebilir kayıtlar geçici dosyalara yazılıp gerçek füzyon
    modelinden geçirilmeye çalışılır; ilgili bağımlılık yoksa metin ön
    analizine geri düşülür. Geçici dosyalar yanıt sonrası silinir.
    """
    received, media_paths, tmp_files = await collect_uploads(video, audio, wearable)
    try:
        if not received and not (transcript and transcript.strip()):
            raise HTTPException(
                status_code=422,
                detail="En az bir kayıt (video/ses/giyilebilir) veya transkript gerekli.",
            )

        report = session.analyze_session(
            pseudonym, evaluator, transcript, received, media_paths
        )
        return SessionAnalysisResponse(**report)
    finally:
        cleanup(tmp_files)


@app.post("/demo/predict", response_model=PredictResponse)
def demo_predict(req: DemoRequest) -> PredictResponse:
    """Görsel arayüz için: senaryoya göre sentetik seans üretip modelden geçirir."""
    try:
        result = demo.run_demo(req.scenario, req.signal, req.seed, req.modalities)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictResponse(
        session_id=f"demo-{req.seed}",
        probabilities=result["probabilities"],
        predicted_label=result["predicted_label"],
        incongruence=result["incongruence"],
        modality_contributions=[
            ModalityContribution(**c) for c in result["modality_contributions"]
        ],
    )


def _to_batch(req: PredictRequest) -> dict[str, torch.Tensor]:
    """İstekteki gömmeleri (B=1) tensör sözlüğüne çevirir."""
    settings = get_settings().model
    mapping = {
        "vision": (req.vision_embedding, settings.vision_embed_dim),
        "audio": (req.audio_embedding, settings.audio_embed_dim),
        "text": (req.text_embedding, settings.text_embed_dim),
        "wearable": (req.wearable_embedding, settings.wearable_embed_dim),
    }
    embeddings: dict[str, torch.Tensor] = {}
    for name, (values, expected_dim) in mapping.items():
        if values is None:
            continue
        if len(values) != expected_dim:
            raise HTTPException(
                status_code=422,
                detail=f"'{name}' gömme boyutu {len(values)}, beklenen {expected_dim}.",
            )
        embeddings[name] = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
    if not embeddings:
        raise HTTPException(status_code=422, detail="En az bir modalite gömmesi gerekli.")
    return embeddings


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    model = _state["model"]
    labels = _state["labels"]
    embeddings = _to_batch(req)
    out = model.fuse(embeddings)
    probs = torch.softmax(out.logits, dim=-1).squeeze(0)

    prob_map = {label: float(probs[i]) for i, label in enumerate(labels)}
    pred_idx = int(probs.argmax().item())

    present = [m for m in ("vision", "audio", "text", "wearable") if m in embeddings]
    contributions = [
        ModalityContribution(modality=m, weight=float(out.attention[0, i]))
        for i, m in enumerate(present)
    ]

    return PredictResponse(
        session_id=req.session_id,
        probabilities=prob_map,
        predicted_label=labels[pred_idx],
        incongruence=float(out.incongruence[0]),
        modality_contributions=contributions,
    )
