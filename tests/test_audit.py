"""Denetim günlüğü (audit log) testleri."""

from psikiyatri_ai.data import audit


def test_log_and_read(tmp_path):
    path = tmp_path / "audit.jsonl"
    audit.log_event("session_analyze", subject="pid_x", detail={"scored": ["wearable"]}, path=path)
    audit.log_event("manifest_ingest", subject="pid_y", path=path)

    events = audit.read_events(path)
    assert len(events) == 2
    assert events[0]["action"] == "session_analyze"
    assert events[0]["subject"] == "pid_x"
    assert events[0]["detail"] == {"scored": ["wearable"]}
    assert "ts" in events[0]


def test_read_missing_returns_empty(tmp_path):
    assert audit.read_events(tmp_path / "yok.jsonl") == []
