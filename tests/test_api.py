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


def test_index_serves_ui():
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "Karar Destek" in r.text


def test_demo_predict_returns_prediction():
    payload = {
        "scenario": "depresyon",
        "signal": 1.2,
        "seed": 7,
        "modalities": ["vision", "audio"],
    }
    with TestClient(app) as client:
        r = client.post("/demo/predict", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["predicted_label"] in ("kontrol", "depresyon")
        assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-4
        assert len(body["modality_contributions"]) == 2


def test_demo_predict_single_modality():
    payload = {"scenario": "kontrol", "modalities": ["audio"]}
    with TestClient(app) as client:
        r = client.post("/demo/predict", json=payload)
        assert r.status_code == 200
        assert len(r.json()["modality_contributions"]) == 1


def test_demo_requires_a_modality():
    with TestClient(app) as client:
        r = client.post("/demo/predict", json={"scenario": "kontrol", "modalities": []})
        assert r.status_code == 422


def test_session_analyze_with_transcript():
    data = {
        "pseudonym": "pid_test",
        "evaluator": "Dr. Test",
        "transcript": "Kendimi çok umutsuz ve yorgun hissediyorum, uyuyamıyorum.",
    }
    with TestClient(app) as client:
        r = client.post("/session/analyze", data=data)
        assert r.status_code == 200
        body = r.json()
        assert body["analysis_available"] is True
        assert body["predicted_label"] in ("kontrol", "depresyon")
        assert body["matched_indicators"]  # belirti ifadeleri saptanmalı
        assert body["risk_level"] in ("düşük", "orta", "yüksek")


def test_session_analyze_masks_pii_in_preview():
    data = {"transcript": "Hasta 12345678901 numaralı, üzgün ve karamsar."}
    with TestClient(app) as client:
        r = client.post("/session/analyze", data=data)
        assert r.status_code == 200
        assert "12345678901" not in r.json()["transcript_preview"]


def test_session_analyze_file_upload():
    files = {"audio": ("seans.wav", b"RIFFxxxxWAVE", "audio/wav")}
    data = {"pseudonym": "pid_x"}
    with TestClient(app) as client:
        r = client.post("/session/analyze", data=data, files=files)
        assert r.status_code == 200
        body = r.json()
        assert body["analysis_available"] is False  # transkript yok
        assert len(body["received"]) == 1
        assert body["received"][0]["modality"] == "audio"


def test_session_analyze_requires_input():
    with TestClient(app) as client:
        r = client.post("/session/analyze", data={"pseudonym": "x"})
        assert r.status_code == 422
