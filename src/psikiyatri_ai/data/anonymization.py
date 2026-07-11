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

# Türkçe özel ad sezgisel maskelemesi (tam NER değil, ihtiyatlı desenler).
# Büyük harfle başlayan bir sözcük (Türkçe karakterler dâhil).
_NAME = r"[A-ZÇĞİÖŞÜ][a-zçğıöşü]+"
# Unvan + ad: "Dr. Ahmet", "Prof. Ayşe Yılmaz", "Uzm. Mehmet".
_TITLE_NAME_RE = re.compile(
    rf"\b(Dr|Dr\.|Doç|Doç\.|Doc|Prof|Prof\.|Op|Op\.|Uz|Uzm|Uzm\.|Hemş|Psk)\.?\s+"
    rf"(?P<ad>{_NAME}(?:\s+{_NAME})?)"
)
# Ad + hitap: "Ahmet Bey", "Ayşe Hanım", "Mehmet Hoca".
_NAME_HONORIFIC_RE = re.compile(
    rf"\b({_NAME})\s+(Bey|Bey'e|Hanım|Hanımefendi|Beyefendi|Hoca)\b"
)


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
    eğitimine girmeden önce TC kimlik, telefon, e-posta ve (sezgisel
    olarak) unvan/hitapla geçen özel adlar maskelenir.

    Not: Ad maskelemesi ihtiyatlı desenlere dayanır (tam NER değildir);
    unvansız/hitapsız geçen adları yakalamayabilir. Kapsamlı ad tespiti
    ileri fazda bir Türkçe NER modeliyle güçlendirilecektir.
    """
    text = _TC_KIMLIK_RE.sub("[TCKN]", text)
    text = _PHONE_RE.sub("[TELEFON]", text)
    text = _EMAIL_RE.sub("[EPOSTA]", text)
    # Unvan sözcüğünü koru, adı maskele: "Dr. Ahmet" -> "Dr. [AD]".
    text = _TITLE_NAME_RE.sub(lambda m: m.group(0).replace(m.group("ad"), "[AD]"), text)
    # "Ahmet Bey" -> "[AD] Bey".
    text = _NAME_HONORIFIC_RE.sub(r"[AD] \2", text)
    return text


def anonymize_record(identifier: str, text: str | None = None) -> AnonymizationResult:
    """Bir kaydı tek adımda anonimleştirir (takma-kimlik + metin maskeleme)."""
    return AnonymizationResult(
        pseudonym=pseudonymize(identifier),
        masked_text=mask_pii(text) if text is not None else None,
    )


__all__ = ["AnonymizationResult", "anonymize_record", "mask_pii", "pseudonymize"]
