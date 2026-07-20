"""Sunucu-taraflı transkripsiyon (yerel Whisper) uç testleri.

Model yükü/indirmesi olmadan çalışması için ``transcribe_audio`` ve
``is_available`` taklit edilir.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from psikiyatri_ai.api import routes_interview as ri
from psikiyatri_ai.api.main import app
from psikiyatri_ai.modalities.audio import transcribe as tr


def _audio_file():
    return {"audio": ("yanit.webm", b"\x00\x01\x02stub", "audio/webm")}


def test_transcribe_returns_text(monkeypatch):
    monkeypatch.setattr("psikiyatri_ai.modalities.audio.transcribe.is_available", lambda: True)
    monkeypatch.setattr(
        "psikiyatri_ai.modalities.audio.transcribe.transcribe_audio",
        lambda path, **k: "kendimi kötü hissediyorum",
    )
    with TestClient(app) as c:
        r = c.post("/interview/transcribe", files=_audio_file())
        assert r.status_code == 200
        body = r.json()
        assert body["engine"] == "whisper-yerel"
        assert body["text"] == "kendimi kötü hissediyorum"


def test_transcribe_graceful_when_unavailable(monkeypatch):
    monkeypatch.setattr("psikiyatri_ai.modalities.audio.transcribe.is_available", lambda: False)
    with TestClient(app) as c:
        r = c.post("/interview/transcribe", files=_audio_file())
        assert r.status_code == 200
        assert r.json() == {"text": "", "engine": "yok"}


def test_transcribe_requires_audio():
    with TestClient(app) as c:
        assert c.post("/interview/transcribe").status_code == 422


def test_module_is_available_reflects_import(monkeypatch):
    # faster-whisper kurulu ortamda True olmalı; değilse False
    assert tr.is_available() in (True, False)
    assert ri.TranscriptionResponse(text="x", engine="whisper-yerel").text == "x"
