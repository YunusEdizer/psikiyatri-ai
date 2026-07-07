"""FastAPI uygulaması.

Uç noktalar:
- ``GET  /health``  : servis sağlık kontrolü
- ``POST /predict`` : modalite gömmelerinden karar destek çıktısı

Model, uygulama yaşam döngüsünde bir kez oluşturulur. Eğitilmiş ağırlıklar
mevcutsa ``PAI_MODEL_DIR`` altından yüklenir; yoksa rastgele başlatılmış
model ile (geliştirme modunda) çalışır ve uyarı loglar.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException

from .. import __version__
from ..config import get_settings
from ..data.schema import DiagnosisLabel
from ..logging_conf import logger, setup_logging
from ..training.model import PsikiyatriModel
from .schemas import (
    HealthResponse,
    ModalityContribution,
    PredictRequest,
    PredictResponse,
)

# num_classes=2 varsayılanına göre etiket eşlemesi
_LABELS = [DiagnosisLabel.KONTROL.value, DiagnosisLabel.DEPRESYON.value]

_state: dict[str, PsikiyatriModel] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    # Servis yolu önceden hesaplanmış gömmeleri alır; yalnızca füzyon başı
    # gerekir. Kodlayıcıları oluşturmayarak ağır bağımlılıklardan kaçınırız.
    model = PsikiyatriModel(settings.model, enabled_modalities=())
    weights = settings.model_dir / "psikiyatri_model.pt"
    if weights.exists():
        # strict=False: tam checkpoint kodlayıcı ağırlıklarını da içerir;
        # burada yalnızca füzyon anahtarları yüklenir.
        model.load_state_dict(torch.load(weights, map_location="cpu"), strict=False)
        logger.info("Eğitilmiş füzyon ağırlıkları yüklendi: {}", weights)
    else:
        logger.warning(
            "Eğitilmiş ağırlık bulunamadı ({}); rastgele başlatılmış model "
            "kullanılıyor. Üretimde kullanmayın.",
            weights,
        )
    model.eval()
    _state["model"] = model
    yield
    _state.clear()


app = FastAPI(
    title="Psikiyatri AI — Karar Destek API",
    description="Multimodal derin öğrenme ile psikiyatrik tanı karar desteği (Eurotech-1).",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(status="ok", version=__version__, env=settings.env)


def _to_batch(req: PredictRequest) -> dict[str, torch.Tensor]:
    """İstekteki gömmeleri (B=1) tensör sözlüğüne çevirir."""
    settings = get_settings().model
    mapping = {
        "vision": (req.vision_embedding, settings.vision_embed_dim),
        "audio": (req.audio_embedding, settings.audio_embed_dim),
        "text": (req.text_embedding, settings.text_embed_dim),
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
    embeddings = _to_batch(req)
    out = model.fuse(embeddings)
    probs = torch.softmax(out.logits, dim=-1).squeeze(0)

    prob_map = {label: float(probs[i]) for i, label in enumerate(_LABELS)}
    pred_idx = int(probs.argmax().item())

    present = [m for m in ("vision", "audio", "text") if m in embeddings]
    contributions = [
        ModalityContribution(modality=m, weight=float(out.attention[0, i]))
        for i, m in enumerate(present)
    ]

    return PredictResponse(
        session_id=req.session_id,
        probabilities=prob_map,
        predicted_label=_LABELS[pred_idx],
        incongruence=float(out.incongruence[0]),
        modality_contributions=contributions,
    )
