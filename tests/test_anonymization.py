"""Anonimleştirme testleri — gizlilik güvenceleri."""

import pytest

from psikiyatri_ai.data.anonymization import (
    anonymize_record,
    mask_pii,
    pseudonymize,
)


def test_pseudonymize_deterministic():
    a = pseudonymize("12345678901", salt="test")
    b = pseudonymize("12345678901", salt="test")
    assert a == b  # aynı girdi + tuz => aynı takma-kimlik
    assert a.startswith("pid_")


def test_pseudonymize_salt_changes_output():
    assert pseudonymize("hasta-1", salt="s1") != pseudonymize("hasta-1", salt="s2")


def test_pseudonymize_rejects_empty():
    with pytest.raises(ValueError):
        pseudonymize("   ")


def test_mask_pii_removes_identifiers():
    text = "Hasta 12345678901 numaralı, 0555 444 33 22, a@b.com ile ulaşılabilir."
    masked = mask_pii(text)
    assert "12345678901" not in masked
    assert "a@b.com" not in masked
    assert "[TCKN]" in masked
    assert "[EPOSTA]" in masked


def test_anonymize_record_combines_both():
    res = anonymize_record("hasta-42", "TC 12345678901")
    assert res.pseudonym.startswith("pid_")
    assert "12345678901" not in (res.masked_text or "")


def test_mask_pii_masks_titled_and_honorific_names():
    masked = mask_pii("Görüşmede Dr. Ahmet Yılmaz ve Ayşe Hanım hazır bulundu.")
    assert "Ahmet" not in masked
    assert "Yılmaz" not in masked
    assert "Ayşe" not in masked
    assert "[AD]" in masked
    # Unvan/hitap korunur (bağlam için)
    assert "Dr." in masked
    assert "Hanım" in masked


def test_to_initials_reduces_full_name():
    from psikiyatri_ai.data.anonymization import to_initials

    # Tam ad yazılsa bile yalnız baş harfler saklanır (KVKK güvencesi)
    assert to_initials("Ahmet Yılmaz") == "A.Y."
    assert to_initials("ayşe fatma kaya") == "A.F.K."
    assert to_initials("A.Y.") == "A.Y."
    assert to_initials("  ") == ""
    assert to_initials("123 456") == ""  # harf yoksa boş
    # Türkçe i → İ
    assert to_initials("irem demir") == "İ.D."
    # En fazla 4 belirteç
    assert to_initials("a b c d e f") == "A.B.C.D."
