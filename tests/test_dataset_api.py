"""Veri seti gözatma API testleri (``/dataset/*``).

Geçici bir manifest ile: özet, filtreleme/sayfalama ve detay uç
noktalarını + kimlik doğrulama zorunluluğunu doğrular.
"""

import json

import pytest
from fastapi.testclient import TestClient

from psikiyatri_ai.config import DEFAULT_ADMIN_PASSWORD, get_settings
from psikiyatri_ai.db import database


def _write_manifest(data_dir):
    processed = data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "pseudonym": f"pid_{i:016x}",
            "session_id": f"TEST_{i:04d}",
            "transcript": f"Kendimi {'kaygılı' if i % 2 else 'çökkün'} hissediyorum. Kayıt {i}.",
            "consent": True,
            "consent_ref": "ARASTIRMA-VERI-SETI/test",
            "label": "anksiyete" if i % 2 else "depresyon",
        }
        for i in range(1, 8)
    ]
    path = processed / "nlp_manifest.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    return records


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "PAI_DATABASE_URL", "sqlite:///" + (tmp_path / "test.db").as_posix()
    )
    monkeypatch.setenv("PAI_AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("PAI_DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    database.reset_engine()
    # Önbelleği sıfırla (modül durumu testler arasında taşınmasın)
    from psikiyatri_ai.api import routes_dataset

    routes_dataset._cache.update(mtime=None, records=[], by_id={})
    _write_manifest(tmp_path / "data")
    from psikiyatri_ai.api.main import app

    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()
    database.reset_engine()


def _login(client) -> dict:
    r = client.post(
        "/auth/login", data={"username": "admin", "password": DEFAULT_ADMIN_PASSWORD}
    )
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["access_token"]}


def test_dataset_requires_auth(client):
    assert client.get("/dataset/summary").status_code == 401
    assert client.get("/dataset/records").status_code == 401
    assert client.get("/dataset/records/TEST_0001").status_code == 401


def test_dataset_summary(client):
    headers = _login(client)
    s = client.get("/dataset/summary", headers=headers).json()
    assert s["available"] is True
    assert s["total"] == 7
    assert s["label_counts"] == {"anksiyete": 4, "depresyon": 3}
    assert s["consent_ref"] == "ARASTIRMA-VERI-SETI/test"


def test_dataset_records_filter_and_pagination(client):
    headers = _login(client)
    # Etiket filtresi
    page = client.get("/dataset/records?label=depresyon", headers=headers).json()
    assert page["total"] == 3
    assert all(r["label"] == "depresyon" for r in page["items"])
    # Sayfalama
    page = client.get("/dataset/records?offset=5&limit=2", headers=headers).json()
    assert page["total"] == 7
    assert len(page["items"]) == 2
    # Metin araması
    page = client.get("/dataset/records?q=çökkün", headers=headers).json()
    assert page["total"] == 3


def test_dataset_record_detail_and_404(client):
    headers = _login(client)
    d = client.get("/dataset/records/TEST_0003", headers=headers).json()
    assert d["session_id"] == "TEST_0003"
    assert "kaygılı" in d["transcript"]
    assert d["label"] == "anksiyete"
    assert client.get("/dataset/records/YOK_9999", headers=headers).status_code == 404


def test_dataset_missing_manifest_graceful(client, tmp_path):
    headers = _login(client)
    (tmp_path / "data" / "processed" / "nlp_manifest.jsonl").unlink()
    s = client.get("/dataset/summary", headers=headers).json()
    assert s["available"] is False
    assert s["total"] == 0
