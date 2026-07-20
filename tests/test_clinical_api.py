"""Klinik akış API testleri: kimlik doğrulama + hasta/görüşme yönetimi.

Her test, geçici bir SQLite veritabanı ve denetim günlüğü ile izole
çalışır (fixture ortam değişkenlerini yamalar ve motoru sıfırlar).
"""

import pytest
from fastapi.testclient import TestClient

from psikiyatri_ai.config import DEFAULT_ADMIN_PASSWORD, get_settings
from psikiyatri_ai.db import database


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "PAI_DATABASE_URL", "sqlite:///" + (tmp_path / "test.db").as_posix()
    )
    monkeypatch.setenv("PAI_AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    get_settings.cache_clear()
    database.reset_engine()
    from psikiyatri_ai.api.main import app

    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()
    database.reset_engine()


def _login(client, username="admin", password=DEFAULT_ADMIN_PASSWORD) -> dict:
    r = client.post("/auth/login", data={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["access_token"]}


def test_login_wrong_password_rejected(client):
    r = client.post("/auth/login", data={"username": "admin", "password": "yanlis"})
    assert r.status_code == 401


def test_patient_endpoints_require_auth(client):
    assert client.get("/patients").status_code == 401
    assert client.post("/patients", json={}).status_code == 401
    assert client.get("/interviews/1").status_code == 401


def test_admin_seeded_and_me_endpoint(client):
    headers = _login(client)
    me = client.get("/auth/me", headers=headers).json()
    assert me["username"] == "admin"
    assert me["role"] == "yonetici"


def test_full_clinical_flow(client):
    admin = _login(client)

    # 1) Yönetici psikolog hesabı açar; psikolog kullanıcı açamaz.
    r = client.post(
        "/auth/users",
        json={
            "username": "psk.demir",
            "password": "guclu-parola-1",
            "full_name": "Psk. Demir",
            "role": "psikolog",
        },
        headers=admin,
    )
    assert r.status_code == 201, r.text
    psk = _login(client, "psk.demir", "guclu-parola-1")
    r = client.post(
        "/auth/users",
        json={"username": "x2", "password": "guclu-parola-2", "role": "psikolog"},
        headers=psk,
    )
    assert r.status_code == 403

    # 2) Hasta kaydı — kurum içi kimlik saklanmadan takma-kimliğe çevrilir.
    r = client.post(
        "/patients",
        json={
            "identifier": "dosya-42",
            "birth_year": 1992,
            "gender": "kadın",
            "referral_note": "Hasta 12345678901 TC ile sevk edildi.",
        },
        headers=psk,
    )
    assert r.status_code == 201, r.text
    patient = r.json()
    assert patient["pseudonym"].startswith("pid_")
    assert "dosya-42" not in patient["pseudonym"]
    assert "12345678901" not in (patient["referral_note_masked"] or "")

    # Aynı kimlikle ikinci kayıt reddedilir.
    r = client.post("/patients", json={"identifier": "dosya-42"}, headers=psk)
    assert r.status_code == 409

    # 3) Görüşme: transkript + klinik girdiler → analiz + karar profili.
    r = client.post(
        f"/patients/{patient['id']}/interviews",
        data={
            "transcript": "Kendimi çok umutsuz hissediyorum, yaşamak istemiyorum.",
            "symptom_duration_weeks": "30",
            "severity": "şiddetli",
            "doctor_note": "Hasta Ahmet Bey görüşmede içe kapanıktı, tel 0532 111 22 33.",
        },
        headers=psk,
    )
    assert r.status_code == 201, r.text
    report = r.json()
    assert report["analysis_available"] is True
    assert report["patient_pseudonym"] == patient["pseudonym"]
    decision = report["decision"]
    assert decision["risk"]["level"] == "yüksek"  # yüksek-risk ifadesi
    assert any("kronikleşme" in f for f in decision["clinical_findings"])
    assert decision["recommendations"]
    assert len(report["notes"]) == 1
    assert "0532" not in report["notes"][0]["content_masked"]
    assert "[AD]" in report["notes"][0]["content_masked"]

    # 4) Hasta detayı görüşmeyi listeler.
    detail = client.get(f"/patients/{patient['id']}", headers=psk).json()
    assert detail["interview_count"] == 1
    assert detail["interviews"][0]["risk_level"] == "yüksek"

    # 5) Kalıcı rapor yeniden okunabilir.
    stored = client.get(f"/interviews/{report['interview_id']}", headers=psk).json()
    assert stored["predicted_label"] == report["predicted_label"]
    assert stored["decision"]["risk"]["level"] == "yüksek"
    assert stored["symptom_duration_weeks"] == 30

    # 6) Görüşmeye sonradan not eklenebilir.
    r = client.post(
        f"/interviews/{report['interview_id']}/notes",
        json={"content": "Kontrol randevusu planlandı."},
        headers=psk,
    )
    assert r.status_code == 201

    # 7) KVKK silme: psikolog silemez, yönetici siler; kayıt izlenebilir.
    assert client.delete(f"/patients/{patient['id']}", headers=psk).status_code == 403
    assert client.delete(f"/patients/{patient['id']}", headers=admin).status_code == 204
    assert client.get(f"/patients/{patient['id']}", headers=psk).status_code == 404

    # 8) Denetim günlüğü yalnızca yöneticiye açık ve silmeyi kaydetmiş.
    assert client.get("/audit/events", headers=psk).status_code == 403
    events = client.get("/audit/events", headers=admin).json()
    actions = [e["action"] for e in events]
    assert "patient_erased" in actions
    assert "interview_created" in actions


def test_patient_without_identifier_gets_random_pseudonym(client):
    headers = _login(client)
    a = client.post("/patients", json={}, headers=headers).json()
    b = client.post("/patients", json={}, headers=headers).json()
    assert a["pseudonym"] != b["pseudonym"]
    assert a["pseudonym"].startswith("pid_")


def test_interview_requires_some_input(client):
    headers = _login(client)
    patient = client.post("/patients", json={}, headers=headers).json()
    r = client.post(
        f"/patients/{patient['id']}/interviews", data={"transcript": ""}, headers=headers
    )
    assert r.status_code == 422


def test_interview_rejects_invalid_severity(client):
    headers = _login(client)
    patient = client.post("/patients", json={}, headers=headers).json()
    r = client.post(
        f"/patients/{patient['id']}/interviews",
        data={"transcript": "üzgünüm", "severity": "felaket"},
        headers=headers,
    )
    assert r.status_code == 422


def test_auto_interview_persists_to_patient(client):
    import json

    from psikiyatri_ai.data.interview_questions import INTERVIEW_QUESTIONS

    by_key = {q["key"]: q["id"] for q in INTERVIEW_QUESTIONS}
    headers = _login(client)
    pid = client.post(
        "/patients", json={"birth_year": 1990, "gender": "Kadın"}, headers=headers
    ).json()["id"]

    answers = [
        {"id": by_key["ÖB1"], "selected": "Evet, tanı konuldu ve hâlen takip/tedavi altındayım"},
        {"id": by_key["Q1"], "selected": "Çok kötü/baş etmekte zorlanıyorum",
         "other": "Kendimi çok mutsuz ve değersiz hissediyorum."},
        {"id": by_key["Q6"], "selected": "Şu anda düşünce var, plan yok"},
    ]
    r = client.post(
        f"/patients/{pid}/interviews/auto",
        data={"answers": json.dumps(answers)},
        headers=headers,
    )
    assert r.status_code == 201, r.text
    body = r.json()
    # Ön bilgi + kritik bayrak raporda
    assert any(p["key"] == "ÖB1" for p in body["preinfo"])
    assert any(f["key"] == "Q6" for f in body["critical_flags"])
    # Kritik yanıt otomatik klinisyen notu olarak eklendi
    assert any("SİSTEM UYARISI" in n["content_masked"] for n in body["notes"])

    # Görüşme hasta kaydına kalıcı işlendi
    detail = client.get(f"/patients/{pid}", headers=headers).json()
    assert len(detail["interviews"]) == 1


def test_auto_interview_requires_auth(client):
    r = client.post("/patients/1/interviews/auto", data={"answers": "[]"})
    assert r.status_code == 401
