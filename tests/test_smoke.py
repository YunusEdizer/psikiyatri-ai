"""Uçtan uca smoke testi — uygulama ayağa kalkar ve temel akış çalışır.

CI'da (pytest) otomatik koşar; ağır modalite bağımlılıkları kurulu
olmasa bile geçmelidir (metin ön analizine zarif düşüş).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from psikiyatri_ai import __version__
from psikiyatri_ai.api.main import app


def test_health_ok():
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["version"] == __version__
        # model_card / model_temperature alanları şemada mevcut (None olabilir)
        assert "model_card" in body


def test_index_served():
    with TestClient(app) as c:
        r = c.get("/")
        assert r.status_code == 200
        assert "Psikiyatr" in r.text  # panel kabuğu döner


def test_security_headers_present():
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.headers.get("X-Content-Type-Options") == "nosniff"
        assert r.headers.get("X-Frame-Options") == "DENY"
        assert "camera" in (r.headers.get("Permissions-Policy") or "")


def test_interview_questions_and_analyze_flow():
    with TestClient(app) as c:
        q = c.get("/interview/questions").json()
        assert q["count"] == 27
        # Belirti sorusundan (Q1) bir yanıtla hızlı analiz akışı
        q1 = next(x["id"] for x in q["questions"] if x["key"] == "Q1")
        import json

        r = c.post(
            "/interview/analyze",
            data={"answers": json.dumps([{"id": q1, "selected": "Kötü",
                                          "other": "kendimi kötü hissediyorum"}])},
        )
        assert r.status_code == 200
        assert r.json()["analysis_available"] is True


def test_manifest_openapi_available():
    with TestClient(app) as c:
        assert c.get("/openapi.json").status_code == 200
        assert c.get("/manifest.webmanifest").status_code == 200
