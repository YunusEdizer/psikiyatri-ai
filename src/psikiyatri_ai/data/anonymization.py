"""Hasta verisi anonimleştirme.

Sağlık verisi KVKK ve uluslararası etik standartlar gereği işlenmeden
önce anonimleştirilir. Bu modül; doğrudan tanımlayıcıları (ad, TC kimlik,
dosya no vb.) geri döndürülemez şekilde takma-kimliğe (pseudonym) çevirir.

Not: Takma-kimlikleme, gizli bir tuz (salt) ile HMAC-SHA256 kullanır.
Tuz sızarsa yeniden kimliklendirme riski doğar; bu nedenle tuz üretimde
bir sır yöneticisinde (secret manager) saklanmalıdır.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass

from ..config import get_settings

# Basit TR kişisel veri desenleri (metin gövdesinde maskeleme için)
_TC_KIMLIK_RE = re.compile(r"\b[1-9]\d{10}\b")
_PHONE_RE = re.compile(r"\b(?:0|\+90)?\s?5\d{2}\s?\d{3}\s?\d{2}\s?\d{2}\b")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")


@dataclass(frozen=True)
class AnonymizationResult:
    """Anonimleştirme çıktısı."""

    pseudonym: str
    masked_text: str | None = None


def pseudonymize(identifier: str, *, salt: str | None = None) -> str:
    """Doğrudan tanımlayıcıyı geri döndürülemez takma-kimliğe çevirir.

    Aynı `identifier` + `salt` her zaman aynı takma-kimliği üretir; bu,
    kişiyi tanımlamadan boylamsal (longitudinal) eşleştirmeye izin verir.
    """
    if not identifier or not identifier.strip():
        raise ValueError("Boş tanımlayıcı takma-kimliklenemez.")

    salt = salt if salt is not None else get_settings().anon_salt
    digest = hmac.new(
        salt.encode("utf-8"),
        identifier.strip().encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"pid_{digest[:16]}"


def mask_pii(text: str) -> str:
    """Serbest metin içindeki olası kişisel veriyi maskeler.

    Konuşma içeriği (NLP modalitesi) uzman notları içerebilir; model
    eğitimine girmeden önce TC kimlik, telefon ve e-posta maskelenir.
    """
    text = _TC_KIMLIK_RE.sub("[TCKN]", text)
    text = _PHONE_RE.sub("[TELEFON]", text)
    text = _EMAIL_RE.sub("[EPOSTA]", text)
    return text


def anonymize_record(identifier: str, text: str | None = None) -> AnonymizationResult:
    """Bir kaydı tek adımda anonimleştirir (takma-kimlik + metin maskeleme)."""
    return AnonymizationResult(
        pseudonym=pseudonymize(identifier),
        masked_text=mask_pii(text) if text is not None else None,
    )


__all__ = ["AnonymizationResult", "anonymize_record", "mask_pii", "pseudonymize"]
