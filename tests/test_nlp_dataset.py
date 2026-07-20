"""``data/nlp_dataset`` aktarım testleri.

Excel'e bağımlı kalmadan (openpyxl'ı taklit ederek) transkript kurma,
etiket eşleme, sızıntı temizliği ve anonimleştirme davranışını doğrular.
"""

from __future__ import annotations

import json

import pytest

from psikiyatri_ai.data import nlp_dataset as nd
from psikiyatri_ai.data.datamodule import load_manifest
from psikiyatri_ai.data.schema import DiagnosisLabel

_DIAG_COL = nd._DIAGNOSIS_COLS[0]  # "Hastalık tanısı"
_LONG_COL = nd._LONG_TEXT_COLS[0]  # "Hastanın_Uzun_Anlatımı"


def _fake_row(diag: str = "Yaygın Anksiyete Bozukluğu") -> dict:
    row = {nd._ID_COL: "TR_0001", _DIAG_COL: diag}
    for i in range(1, 25):
        row[f"Q{i}_soru"] = f"Ankete verilen {i}. yanıt."
    row[_LONG_COL] = (
        f"Hasta uzun anlatımında yakınmalarını {diag} profiliyle uyumlu ifade etti. "
        "İletişim için 0555 123 45 67 numarasını ve Dr. Ahmet Yılmaz'ı belirtti."
    )
    return row


_Q_COLS = [f"Q{i}_soru" for i in range(1, 25)]


def test_diagnosis_map_covers_all_dataset_labels() -> None:
    # Eşlemenin tüm hedefleri geçerli taksonomi üyesi olmalı.
    for label in nd.DIAGNOSIS_MAP.values():
        assert isinstance(label, DiagnosisLabel)
    assert nd.DIAGNOSIS_MAP["Travma Sonrası Stres Bozukluğu"] is DiagnosisLabel.TSSB
    assert nd.DIAGNOSIS_MAP["Obsesif Kompulsif Bozukluk"] is DiagnosisLabel.OKB


def test_scrub_removes_diagnosis_names() -> None:
    text = "Belirtiler Yaygın Anksiyete Bozukluğu ve OKB ile uyumludur."
    scrubbed = nd.scrub_diagnosis_leak(text)
    assert "Anksiyete Bozukluğu" not in scrubbed
    assert "OKB" not in scrubbed
    assert "[TANI]" in scrubbed


def test_build_transcript_no_leak_and_masks_pii() -> None:
    row = _fake_row("Sosyal Anksiyete Bozukluğu")
    transcript = nd.build_transcript(row, _Q_COLS, _LONG_COL)
    # Tanı adı sızıntısı temizlenmeli
    assert "Sosyal Anksiyete Bozukluğu" not in transcript
    # PII maskelenmeli (telefon + unvan+ad)
    assert "0555" not in transcript
    assert "[TELEFON]" in transcript
    assert "Ahmet" not in transcript
    # Anket yanıtları korunmalı
    assert "Ankete verilen 1. yanıt." in transcript


def test_iter_samples_maps_and_anonymizes(monkeypatch) -> None:
    header = [nd._ID_COL, _DIAG_COL, _LONG_COL, *_Q_COLS]

    def _row_values(diag: str) -> tuple:
        r = _fake_row(diag)
        return tuple(r.get(h) for h in header)

    rows = [
        _row_values("Majör Depresif Bozukluk"),
        _row_values("Panik Bozukluk"),
        _row_values("Bilinmeyen Tanı XYZ"),  # atlanmalı
    ]

    def fake_load(_path):
        return header, iter(rows)

    monkeypatch.setattr(nd, "_load_rows", fake_load)

    stats = nd.IngestStats()
    samples = list(nd.iter_samples("dummy.xlsx", salt="test-salt", stats=stats))

    assert len(samples) == 2
    assert stats.skipped_unknown_label == 1
    assert samples[0].label is DiagnosisLabel.DEPRESYON
    assert samples[1].label is DiagnosisLabel.PANIK
    # Anonim: takma-kimlik üretilmiş, ham Hasta_ID transkriptte yok
    assert samples[0].pseudonym.startswith("pid_")
    assert samples[0].consent is True


def test_build_manifest_end_to_end(tmp_path, monkeypatch) -> None:
    header = [nd._ID_COL, _DIAG_COL, _LONG_COL, *_Q_COLS]
    rows = [tuple(_fake_row("Şizofreni").get(h) for h in header)]

    monkeypatch.setattr(nd, "_load_rows", lambda _p: (header, iter(rows)))
    # Excel varlık kontrolünü geç
    monkeypatch.setattr(nd.Path, "exists", lambda self: True)

    out = tmp_path / "nlp_manifest.jsonl"
    written, stats = nd.build_manifest_from_nlp_dataset("dummy.xlsx", out_path=out)

    assert written == out
    assert stats.written == 1
    # Üretilen manifest, eğitim hattının yükleyicisiyle okunabilmeli
    samples = load_manifest(out)
    assert len(samples) == 1
    assert samples[0].label is DiagnosisLabel.SIZOFRENI
    # Satır geçerli JSON ve tanı adı sızıntısı içermiyor
    record = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert "Şizofreni" not in record["transcript"]


def test_iter_samples_requires_q_columns(monkeypatch) -> None:
    monkeypatch.setattr(nd, "_load_rows", lambda _p: (["Hasta_ID", "X"], iter([])))
    with pytest.raises(ValueError, match="Anket"):
        list(nd.iter_samples("dummy.xlsx"))
