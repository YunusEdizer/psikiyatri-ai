"""Türkçe konuşma içeriği için sözlük-tabanlı ön analiz.

Bu modül, eğitilmiş NLP modeli (BERTurk) devreye girene kadar **gerçek,
girdi-güdümlü** bir ön risk sinyali sağlar. Transkriptte psikiyatrik
belirtilerle ilişkili anahtar sözcükleri sayarak basit bir olasılık
üretir. Sonuç şeffaftır: hangi ifadelerin eşleştiği rapora eklenir.

> Bu bir yardımcı ön analizdir; klinik teşhis aracı değildir. Nihai model
> çok-modaliteli derin öğrenmeye dayanacaktır (bkz. fusion/).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Belirti kategorileri ve ilişkili Türkçe ifadeler (kök/gövde eşleşmeli)
LEXICON: dict[str, tuple[str, ...]] = {
    "çökkün duygudurum": (
        "üzgün", "mutsuz", "umutsuz", "çaresiz", "boşluk", "değersiz",
        "suçlu", "karamsar", "ağla", "keder", "hüzün",
    ),
    "ilgi kaybı": (
        "istemiyorum", "zevk almıyorum", "ilgim yok", "anlamsız",
        "vazgeç", "isteksiz", "keyif almıyorum",
    ),
    "bitkinlik/uyku": (
        "yorgun", "bitkin", "uyuyamıyorum", "uykusuz", "halsiz",
        "enerjim yok", "uyku düzenim",
    ),
    "kaygı": (
        "endişe", "kaygı", "panik", "korku", "gergin", "huzursuz",
        "tedirgin", "kötü bir şey olacak",
    ),
    "yüksek risk": (
        "yaşamak istemiyorum", "intihar", "ölmek", "zarar vermek",
        "kendime zarar", "bitsin",
    ),
}

# "Yüksek risk" ifadelerine ekstra ağırlık verilir
_CATEGORY_WEIGHT = {"yüksek risk": 3.0}
_DEFAULT_WEIGHT = 1.0

# Doygunluk için: bu kadar ağırlıklı eşleşme ~ üst olasılığa yaklaştırır
_SATURATION = 6.0


@dataclass
class TextRiskResult:
    """Sözlük-tabanlı ön analiz çıktısı."""

    depression_probability: float  # [0, 1]
    matched: dict[str, list[str]] = field(default_factory=dict)  # kategori -> ifadeler
    high_risk: bool = False

    @property
    def score(self) -> float:
        return self.depression_probability


# Olumsuzlama belirteçleri: yüksek-risk ifadesi bir olumsuzlamayla
# geçiyorsa risk bayrağı üretilmez. Türkçe olumsuzluk çoğunlukla ifadeden
# SONRA gelir (ek/sözcük): "... yok", "... değil", "... istemiyorum".
_NEG_AFTER = (
    "yok", "yoktur", "değil", "olmadı", "olmuyor",
    "istemiyorum", "istemem", "düşünmüyorum", "düşünmem", "geçmiyor",
)
# Az sayıda belirteç ifadeden ÖNCE gelir: "asla intihar düşünmem".
_NEG_BEFORE = ("asla", "hiçbir zaman")


def _is_negated(lowered: str, match_start: int, match_end: int) -> bool:
    """Eşleşen yüksek-risk ifadesi aynı tümcecikte olumsuzlanmış mı?

    Hem ifadeden SONRAKİ pencere (ek/sözcük olumsuzlaması, ~35 krk) hem de
    ÖNCEKİ pencere (~25 krk; "asla/hiçbir zaman") tümcecik sınırına
    (.,;) kadar taranır. Örn. "kendime zarar verme düşüncem yok",
    "ölmek istemiyorum", "asla intihar düşünmem" → olumsuz.
    """
    def _clause(s: str, from_end: bool) -> str:
        for sep in (".", ",", ";"):
            s = s.rsplit(sep, 1)[-1] if from_end else s.split(sep, 1)[0]
        return s

    tail = _clause(lowered[match_end : match_end + 35], from_end=False)
    if any(neg in tail for neg in _NEG_AFTER):
        return True
    head = _clause(lowered[max(0, match_start - 25) : match_start], from_end=True)
    return any(neg in head for neg in _NEG_BEFORE)


def analyze_text_risk(text: str) -> TextRiskResult:
    """Transkriptten sözlük-tabanlı ön risk olasılığı üretir.

    Args:
        text: (Tercihen PII'den arındırılmış) konuşma metni.

    Returns:
        TextRiskResult — olasılık, eşleşen ifadeler ve yüksek-risk bayrağı.
    """
    lowered = _lower_tr(text)
    matched: dict[str, list[str]] = {}
    weighted = 0.0
    high_risk = False

    for category, terms in LEXICON.items():
        hits: list[str] = []
        for t in terms:
            m = re.search(re.escape(_lower_tr(t)), lowered)
            if not m:
                continue
            # Yüksek-risk ifadesi olumsuzlanmışsa (ör. "... yok") sayma.
            if category == "yüksek risk" and _is_negated(lowered, m.start(), m.end()):
                continue
            hits.append(t)
        if hits:
            matched[category] = hits
            weighted += _CATEGORY_WEIGHT.get(category, _DEFAULT_WEIGHT) * len(hits)
            if category == "yüksek risk":
                high_risk = True

    # Doygunluk eğrisi: ağırlıklı eşleşme -> [0, 1]
    prob = 1.0 - pow(2.71828, -weighted / _SATURATION)
    return TextRiskResult(
        depression_probability=round(prob, 4), matched=matched, high_risk=high_risk
    )


def _lower_tr(text: str) -> str:
    """Türkçe duyarlı küçük harfe çevirme (İ/I ayrımı)."""
    return text.replace("İ", "i").replace("I", "ı").lower()


__all__ = ["LEXICON", "TextRiskResult", "analyze_text_risk"]
