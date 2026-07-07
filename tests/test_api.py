"""API testleri (FastAPI TestClient)."""

from fastapi.testclient import TestClient

from psikiyatri_ai.api.main import app
from psikiyatri_ai.config import get_settings


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_predict_with_text_only():
    dim = get_settings().model.text_embed_dim
    payload = {
        "session_id": "seans-1",
        "text_embedding": [0.0] * dim,
    }
    with TestClient(app) as client:
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["session_id"] == "seans-1"
        assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-4
        assert len(body["modality_contributions"]) == 1


def test_predict_rejects_wrong_dim():
    payload = {"session_id": "s", "text_embedding": [0.0, 1.0]}
    with TestClient(app) as client:
        r = client.post("/predict", json=payload)
        assert r.status_code == 422


def test_predict_requires_a_modality():
    with TestClient(app) as client:
        r = client.post("/predict", json={"session_id": "s"})
        assert r.status_code == 422
