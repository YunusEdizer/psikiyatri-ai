"""Konuşma metni ön işleme (Türkçe).

Ham transkript, kişisel veri maskeleme (bkz. ``data.anonymization``)
sonrası burada normalize edilir. Ağır tokenizer bağımlılıkları model
katmanında yüklenir; bu modül yalnızca hafif metin temizliği yapar.
"""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Boşlukları sadeleştirir ve Unicode'u NFC'ye normalize eder.

    Türkçe karakterler (ç, ğ, ı, İ, ö, ş, ü) korunur; büyük/küçük harf
    dönüşümü modele bırakılır (BERTurk cased/uncased seçimine göre).
    """
    text = unicodedata.normalize("NFC", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


__all__ = ["normalize_text"]
