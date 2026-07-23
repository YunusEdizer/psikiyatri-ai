"""Otomatik görüşme testleri: seçenekli soru seti, transkript, ön bilgi,
kritik bayraklar ve API uçları."""

import json

from fastapi.testclient import TestClient

from psikiyatri_ai.api.main import app
from psikiyatri_ai.data.interview_questions import (
    INTERVIEW_QUESTIONS,
    build_interview_transcript,
    build_preinfo_summary,
    critical_flags,
)

_BY_KEY = {q["key"]: q["id"] for q in INTERVIEW_QUESTIONS}


def test_question_set_structure():
    # 3 ön bilgi (ÖB) + 24 belirti (Q) = 27
    assert len(INTERVIEW_QUESTIONS) == 27
    pre = [q for q in INTERVIEW_QUESTIONS if q["section"] == "on_bilgi"]
    sym = [q for q in INTERVIEW_QUESTIONS if q["section"] == "belirti"]
    assert [q["key"] for q in pre] == ["ÖB1", "ÖB2", "ÖB3"]
    assert [q["key"] for q in sym] == [f"Q{i}" for i in range(1, 25)]
    assert [q["id"] for q in INTERVIEW_QUESTIONS] == list(range(1, 28))


def test_every_question_has_options():
    for q in INTERVIEW_QUESTIONS:
        assert q["text"].strip()
        assert q["category"].strip()
        assert len(q["options"]) >= 3
        assert q["options"][-1].startswith("Diğer")


def test_risk_question_has_warning():
    q6 = next(q for q in INTERVIEW_QUESTIONS if q["key"] == "Q6")
    assert q6["category"] == "risk"
    assert q6["warn"] and "güvenlik" in q6["warn"].lower()
    assert "zarar" in q6["text"]


def test_transcript_builder_uses_selected_and_other():
    answers = [
        {"id": _BY_KEY["ÖB1"], "selected": "Evet, tanı konuldu ve hâlen takip/tedavi altındayım"},
        {"id": _BY_KEY["Q1"], "selected": "Çok kötü/baş etmekte zorlanıyorum", "other": ""},
        {"id": _BY_KEY["Q2"], "selected": "Diğer (belirtiniz)", "other": "hiç keyif almıyorum"},
        {"id": _BY_KEY["Q3"], "selected": "", "other": "  "},
    ]
    t = build_interview_transcript(answers)
    assert "tanı konuldu" not in t  # ÖB tanı modeline verilmez
    assert "Danışan: Çok kötü/baş etmekte zorlanıyorum" in t
    assert "Danışan: hiç keyif almıyorum" in t
    assert t.count("Görüşmeci:") == 2


def test_transcript_backward_compat_answer_field():
    answers = [{"id": _BY_KEY["Q1"], "answer": "Kendimi yorgun hissediyorum."}]
    assert "Danışan: Kendimi yorgun hissediyorum." in build_interview_transcript(answers)


def test_preinfo_summary_captures_only_ob():
    answers = [
        {"id": _BY_KEY["ÖB2"], "selected": "Evet, düzenli kullanıyorum"},
        {"id": _BY_KEY["Q1"], "selected": "İyi"},
    ]
    pre = build_preinfo_summary(answers)
    assert len(pre) == 1 and pre[0]["key"] == "ÖB2" and "düzenli" in pre[0]["answer"]


def test_critical_flags_only_symptom_warnings():
    answers = [
        {"id": _BY_KEY["ÖB1"], "selected": "Evet, geçmişte tanı konuldu; şu anda takipte değilim"},
        {"id": _BY_KEY["Q6"], "selected": "Şu anda düşünce var, plan yok"},
        {"id": _BY_KEY["Q18"], "selected": "Hayır"},
    ]
    assert {f["key"] for f in critical_flags(answers)} == {"Q6"}


def test_transcript_builder_empty_input():
    assert build_interview_transcript([]) == ""
    assert build_interview_transcript([{"id": _BY_KEY["Q1"], "selected": "", "other": ""}]) == ""


def test_questions_endpoint_serves_options():
    with TestClient(app) as client:
        body = client.get("/interview/questions").json()
        assert body["count"] == 27
        first = body["questions"][0]
        assert first["key"] == "ÖB1" and first["section"] == "on_bilgi" and first["options"]
        assert "tanı koymaz" in body["note"]


def test_analyze_endpoint_with_selected_answers():
    answers = [
        {"id": _BY_KEY["Q1"], "selected": "Çok kötü/baş etmekte zorlanıyorum",
         "other": "Kendimi çok mutsuz ve değersiz hissediyorum."},
        {"id": _BY_KEY["Q6"], "selected": "Şu anda düşünce var, plan yok"},
    ]
    with TestClient(app) as client:
        r = client.post(
            "/interview/analyze",
            data={"pseudonym": "pid_test", "answers": json.dumps(answers)},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["analysis_available"] is True
        assert body["probabilities"]
        assert any(f["key"] == "Q6" for f in body["critical_flags"])
        # Kendine zarar bayrağı riski yükseltmeli (seçenek metni sözlükte olmasa da)
        assert body["high_risk"] is True
        assert body["risk_level"] == "yüksek"


def test_analyze_endpoint_preinfo_in_report():
    answers = [
        {"id": _BY_KEY["ÖB1"], "selected": "Evet, tanı konuldu ve hâlen takip/tedavi altındayım"},
        {"id": _BY_KEY["Q1"], "selected": "Kötü", "other": "kendimi kötü hissediyorum"},
    ]
    with TestClient(app) as client:
        r = client.post(
            "/interview/analyze",
            data={"pseudonym": "pid_test", "answers": json.dumps(answers)},
        )
        assert r.status_code == 200
        assert any(p["key"] == "ÖB1" for p in r.json()["preinfo"])


def test_analyze_endpoint_rejects_bad_json():
    with TestClient(app) as client:
        assert client.post("/interview/analyze", data={"answers": "bozuk["}).status_code == 422


def test_analyze_endpoint_requires_content():
    with TestClient(app) as client:
        assert client.post("/interview/analyze", data={"answers": "[]"}).status_code == 422
