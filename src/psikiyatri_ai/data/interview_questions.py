"""Yapılandırılmış klinik tarama soru seti (seçenekli).

Kaynak: ``psikiyatrik_nlp_sorular_secenekli`` Excel formu (ÖB1-ÖB3 ön bilgi +
Q1-Q24 belirti soruları). Her soru; danışana sunulan **çoktan seçmeli**
şıklar, ``Diğer (belirtiniz)`` serbest açıklaması ve gerektiğinde bir
**kritik uyarı** taşır. Sorular NLP korpusundaki ``Q1..Q24`` sütunlarıyla
hizalıdır; böylece görüşme yanıtları eğitim verisiyle aynı şemaya oturur.

Sorular tanı KOYMAZ; DSM-5 ana belirti kümelerini tarayan yarı
yapılandırılmış bir ön görüşme protokolüdür. ``risk`` (Q6) kendine zarar
riskini; psikotik ve güvenlik uyarılı sorular uzman değerlendirmesi
gerektiren yanıtları işaretler. Ön bilgi soruları (ÖB) tanı modeline
VERİLMEZ (önceki tanı yanlılığını önlemek için); raporda bağlam sunulur.
"""

from __future__ import annotations

from typing import TypedDict


class InterviewQuestion(TypedDict):
    """Tek bir tarama sorusu (çoktan seçmeli + serbest açıklama)."""

    id: int
    key: str  # NLP korpusu / form karşılığı (ÖB1..ÖB3, Q1..Q24)
    text: str  # Danışana sunulan soru metni
    category: str  # Belirti kümesi (arayüzde gruplama/ilerleme için)
    section: str  # "on_bilgi" | "belirti"
    options: list[str]  # Çoktan seçmeli şıklar (son eleman: Diğer)
    warn: str | None  # Kritik/uzman uyarısı (varsa)


# Tanı modeline verilmeyen, yalnızca bağlam olan ön bilgi soruları.
PREINFO_KEYS = ("ÖB1", "ÖB2", "ÖB3")

INTERVIEW_QUESTIONS: list[InterviewQuestion] = [
    {
        "id": 1, "key": "ÖB1", "category": "on_bilgi",
        "section": "on_bilgi",
        "text": "Daha önce bir psikiyatri uzmanı, psikolog veya başka bir ruh sağlığı uzmanı tarafından size herhangi bir psikiyatrik/psikolojik tanı konuldu mu?",
        "warn": "Tanı varsa adı ve yaklaşık konulma yılı açıklama alanına yazılabilir.",
        "options": [
            "Hayır, daha önce tanı konulmadı",
            "Evet, geçmişte tanı konuldu; şu anda takipte değilim",
            "Evet, tanı konuldu ve hâlen takip/tedavi altındayım",
            "Bir değerlendirme yapıldı ancak tanıdan emin değilim",
            "Tanı konulup konulmadığını bilmiyorum/hatırlamıyorum",
            "Yanıtlamak istemiyorum",
            "Diğer (belirtiniz)",
        ],
    },
    {
        "id": 2, "key": "ÖB2", "category": "on_bilgi",
        "section": "on_bilgi",
        "text": "Hâlen veya yakın dönemde psikiyatrik/psikolojik bir sorun için ilaç kullanıyor musunuz?",
        "warn": "İlaç bırakma veya doz değişikliği yalnızca hekim önerisiyle yapılmalıdır.",
        "options": [
            "Hayır, kullanmıyorum",
            "Evet, düzenli kullanıyorum",
            "Evet, düzensiz kullanıyorum",
            "İlaç reçete edildi ancak başlamadım",
            "Kendi kararımla bıraktım",
            "Hekim önerisiyle bıraktım",
            "İlaç kullanıp kullanmadığımdan/ilacın niteliğinden emin değilim",
            "Yanıtlamak istemiyorum",
            "Diğer (belirtiniz)",
        ],
    },
    {
        "id": 3, "key": "ÖB3", "category": "on_bilgi",
        "section": "on_bilgi",
        "text": "Kullandığınız psikiyatrik ilaç veya ilaçlar hangi gruptadır? (Birden fazla grup varsa açıklama alanında belirtiniz.)",
        "warn": "İlaç adı, dozu, kullanım sıklığı ve ne zamandır kullanıldığı açıklama alanına yazılabilir.",
        "options": [
            "Antidepresan",
            "Antipsikotik",
            "Duygudurum düzenleyici",
            "Kaygı giderici/sakinleştirici",
            "Uyku düzenleyici ilaç",
            "DEHB/dikkat düzenleyici ilaç",
            "Birden fazla ilaç grubu kullanıyorum",
            "İlaç grubunu bilmiyorum; adını açıklama alanına yazacağım",
            "Yanıtlamak istemiyorum",
            "Diğer (belirtiniz)",
        ],
    },
    {
        "id": 4, "key": "Q1", "category": "genel",
        "section": "belirti",
        "text": "Son zamanlarda genel olarak kendinizi nasıl hissediyorsunuz?",
        "warn": None,
        "options": [
            "Çok iyi",
            "İyi",
            "Karışık/değişken",
            "Kötü",
            "Çok kötü/baş etmekte zorlanıyorum",
            "Yanıtlamak istemiyorum",
            "Diğer (belirtiniz)",
        ],
    },
    {
        "id": 5, "key": "Q2", "category": "depresyon",
        "section": "belirti",
        "text": "Eskiden keyif aldığınız etkinliklerden ne ölçüde keyif alıyorsunuz?",
        "warn": None,
        "options": [
            "Eskisi kadar",
            "Biraz daha az",
            "Belirgin daha az",
            "Neredeyse hiç",
            "Bu tür etkinliğim yok",
            "Yanıtlamak istemiyorum",
            "Diğer (belirtiniz)",
        ],
    },
    {
        "id": 6, "key": "Q3", "category": "depresyon",
        "section": "belirti",
        "text": "Uyku düzeniniz nasıldır?",
        "warn": None,
        "options": [
            "Düzenli ve dinlendirici",
            "Uykuya dalmakta zorlanıyorum",
            "Gece sık uyanıyorum",
            "Erken uyanıyorum",
            "Fazla uyuyorum",
            "Çok az uyuyorum",
            "Düzensiz",
            "Diğer (belirtiniz)",
        ],
    },
    {
        "id": 7, "key": "Q4", "category": "depresyon",
        "section": "belirti",
        "text": "İştahınızda değişiklik oldu mu?",
        "warn": None,
        "options": [
            "Değişmedi",
            "Biraz azaldı",
            "Belirgin azaldı",
            "Biraz arttı",
            "Belirgin arttı",
            "Çok değişken",
            "Diğer (belirtiniz)",
        ],
    },
    {
        "id": 8, "key": "Q5", "category": "depresyon",
        "section": "belirti",
        "text": "Kendinizi değersiz veya suçlu hissediyor musunuz?",
        "warn": None,
        "options": [
            "Hiç",
            "Nadiren",
            "Bazen",
            "Sık sık",
            "Neredeyse her gün/çok yoğun",
            "Yanıtlamak istemiyorum",
            "Diğer (belirtiniz)",
        ],
    },
    {
        "id": 9, "key": "Q6", "category": "risk",
        "section": "belirti",
        "text": "Kendinize zarar verme veya yaşamınıza son verme düşünceniz oldu mu?",
        "warn": "Acil güvenlik değerlendirmesi",
        "options": [
            "Hayır",
            "Geçmişte oldu, şimdi yok",
            "Son dönemde oldu, plan yok",
            "Şu anda düşünce var, plan yok",
            "Şu anda düşünce ve plan var",
            "Yalnızca uzmanla yanıtlamak istiyorum",
            "Diğer (belirtiniz)",
        ],
    },
    {
        "id": 10, "key": "Q7", "category": "anksiyete",
        "section": "belirti",
        "text": "Yoğun kaygı veya endişe yaşıyor musunuz?",
        "warn": None,
        "options": [
            "Hiç",
            "Nadiren",
            "Bazen",
            "Sık sık",
            "Neredeyse sürekli/işlevimi bozuyor",
            "Diğer (belirtiniz)",
        ],
    },
    {
        "id": 11, "key": "Q8", "category": "panik",
        "section": "belirti",
        "text": "Panik atağa benzer ani yoğun korku nöbetleri geçiriyor musunuz?",
        "warn": None,
        "options": [
            "Hiç",
            "Geçmişte oldu",
            "Ayda birden az",
            "Ayda birkaç kez",
            "Haftada bir veya daha sık",
            "Neredeyse her gün",
            "Emin değilim",
            "Diğer (belirtiniz)",
        ],
    },
    {
        "id": 12, "key": "Q9", "category": "sosyal_anksiyete",
        "section": "belirti",
        "text": "İnsanların sizi yargıladığını düşünür müsünüz?",
        "warn": None,
        "options": [
            "Hiç",
            "Nadiren",
            "Bazen",
            "Sık sık",
            "Neredeyse her sosyal durumda",
            "Diğer (belirtiniz)",
        ],
    },
    {
        "id": 13, "key": "Q10", "category": "sosyal_anksiyete",
        "section": "belirti",
        "text": "Sosyal ortamlardan kaçındığınız olur mu?",
        "warn": None,
        "options": [
            "Hiç",
            "Nadiren",
            "Bazen",
            "Sık sık",
            "Neredeyse tamamen",
            "Diğer (belirtiniz)",
        ],
    },
    {
        "id": 14, "key": "Q11", "category": "okb",
        "section": "belirti",
        "text": "İstemediğiniz hâlde zihninize gelen rahatsız edici düşünceler olur mu?",
        "warn": None,
        "options": [
            "Hiç",
            "Nadiren",
            "Bazen",
            "Sık sık",
            "Çok sık/işlevimi bozuyor",
            "Diğer (belirtiniz)",
        ],
    },
    {
        "id": 15, "key": "Q12", "category": "okb",
        "section": "belirti",
        "text": "Rahatlamak için tekrarladığınız davranışlar var mı?",
        "warn": None,
        "options": [
            "Hayır",
            "Nadiren",
            "Bazen",
            "Sık sık",
            "Günde uzun zaman alıyor",
            "Emin değilim",
            "Diğer (belirtiniz)",
        ],
    },
    {
        "id": 16, "key": "Q13", "category": "tssb",
        "section": "belirti",
        "text": "Geçmişte yaşadığınız kötü olaylar istemeden aklınıza gelir mi?",
        "warn": None,
        "options": [
            "Hiç",
            "Nadiren",
            "Bazen",
            "Sık sık",
            "Çok sık/yeniden yaşıyor gibi",
            "Böyle bir olay yaşamadım",
            "Diğer (belirtiniz)",
        ],
    },
    {
        "id": 17, "key": "Q14", "category": "tssb",
        "section": "belirti",
        "text": "Rahatsız edici veya tekrarlayan kabuslar görüyor musunuz?",
        "warn": None,
        "options": [
            "Hiç",
            "Ayda birden az",
            "Ayda birkaç kez",
            "Haftada bir veya daha sık",
            "Neredeyse her gece",
            "Diğer (belirtiniz)",
        ],
    },
    {
        "id": 18, "key": "Q15", "category": "bipolar",
        "section": "belirti",
        "text": "Normalden belirgin biçimde daha enerjik veya coşkulu dönemleriniz olur mu?",
        "warn": None,
        "options": [
            "Hayır",
            "Kısa süreli ve nadiren",
            "Bazen birkaç gün",
            "Sık ve birkaç gün",
            "Bir hafta veya daha uzun",
            "Emin değilim",
            "Diğer (belirtiniz)",
        ],
    },
    {
        "id": 19, "key": "Q16", "category": "bipolar",
        "section": "belirti",
        "text": "Çok az uyuduğunuz hâlde enerjik olduğunuz dönemler olur mu?",
        "warn": None,
        "options": [
            "Hayır",
            "Nadiren, bir gece",
            "Bazen 2-3 gün",
            "Sık, birkaç gün",
            "Bir hafta veya daha uzun",
            "Emin değilim",
            "Diğer (belirtiniz)",
        ],
    },
    {
        "id": 20, "key": "Q17", "category": "bipolar",
        "section": "belirti",
        "text": "Düşünmeden riskli kararlar alma eğiliminiz oldu mu?",
        "warn": None,
        "options": [
            "Hiç",
            "Nadiren",
            "Bazen",
            "Sık sık",
            "Ciddi sonuçlara yol açtı",
            "Diğer (belirtiniz)",
        ],
    },
    {
        "id": 21, "key": "Q18", "category": "psikotik",
        "section": "belirti",
        "text": "Başkalarının duymadığı sesler duyduğunuz oldu mu?",
        "warn": "Uzman değerlendirmesi",
        "options": [
            "Hayır",
            "Uyku-uyanıklık arasında",
            "Geçmişte oldu",
            "Nadiren oluyor",
            "Sık oluyor",
            "Şu anda oluyor",
            "Emin değilim",
            "Diğer (belirtiniz)",
        ],
    },
    {
        "id": 22, "key": "Q19", "category": "psikotik",
        "section": "belirti",
        "text": "Başkalarının görmediği şeyler gördüğünüz oldu mu?",
        "warn": "Uzman değerlendirmesi",
        "options": [
            "Hayır",
            "Uyku-uyanıklık arasında",
            "Geçmişte oldu",
            "Nadiren oluyor",
            "Sık oluyor",
            "Şu anda oluyor",
            "Emin değilim",
            "Diğer (belirtiniz)",
        ],
    },
    {
        "id": 23, "key": "Q20", "category": "psikotik",
        "section": "belirti",
        "text": "İnsanların size zarar vermeye çalıştığını düşündüğünüz olur mu?",
        "warn": "Güvenlik değerlendirmesi",
        "options": [
            "Hiç",
            "Nadiren/kuşku düzeyinde",
            "Bazen",
            "Sık sık",
            "Neredeyse sürekli/güvende hissetmiyorum",
            "Diğer (belirtiniz)",
        ],
    },
    {
        "id": 24, "key": "Q21", "category": "psikotik",
        "section": "belirti",
        "text": "Düşüncelerinizin dışarıdan kontrol edildiğini hisseder misiniz?",
        "warn": "Uzman değerlendirmesi",
        "options": [
            "Hiç",
            "Nadiren/emin değilim",
            "Bazen",
            "Sık sık",
            "Neredeyse sürekli",
            "Diğer (belirtiniz)",
        ],
    },
    {
        "id": 25, "key": "Q22", "category": "duygudurum",
        "section": "belirti",
        "text": "Duygularınız hızlı değişir mi?",
        "warn": None,
        "options": [
            "Hiç",
            "Nadiren",
            "Bazen",
            "Sık sık",
            "Gün içinde birçok kez/işlevimi bozuyor",
            "Diğer (belirtiniz)",
        ],
    },
    {
        "id": 26, "key": "Q23", "category": "iliskisel",
        "section": "belirti",
        "text": "Yakın ilişkilerinizde belirgin iniş çıkışlar yaşar mısınız?",
        "warn": None,
        "options": [
            "Hayır",
            "Nadiren",
            "Bazen",
            "Sık sık",
            "Çok yoğun/ilişkiyi zorlaştırıyor",
            "Yakın ilişkim yok",
            "Diğer (belirtiniz)",
        ],
    },
    {
        "id": 27, "key": "Q24", "category": "durtu",
        "section": "belirti",
        "text": "Öfke patlamaları yaşadığınız olur mu?",
        "warn": None,
        "options": [
            "Hiç",
            "Nadiren",
            "Bazen",
            "Sık sık",
            "Çok sık/zarar riski doğuruyor",
            "Diğer (belirtiniz)",
        ],
    },
]


_DECLINE = {"Yanıtlamak istemiyorum", "Yalnızca uzmanla yanıtlamak istiyorum"}


def _compose_answer(q: InterviewQuestion, selected: str, other: str) -> str:
    """Seçilen şık + serbest açıklamayı tek bir doğal yanıt metnine çevirir."""
    selected = (selected or "").strip()
    other = (other or "").strip()
    if selected.startswith("Diğer"):
        return other  # serbest metin doğrudan yanıttır
    if not selected:
        return other
    if other:
        return f"{selected} - {other}"
    return selected


def build_interview_transcript(answers: list[dict]) -> str:
    """Belirti (Q1-Q24) yanıtlarını analiz hattının beklediği transkripte çevirir.

    Her yanıt ``{"id", "selected", "other"}`` (yeni) veya ``{"id", "answer"}``
    (eski, serbest metin) biçiminde olabilir. Ön bilgi (ÖB) soruları ve boş
    yanıtlar transkripte YAZILMAZ.
    """
    by_id = {q["id"]: q for q in INTERVIEW_QUESTIONS}
    parts: list[str] = []
    for a in answers:
        qid = a.get("id")
        q = by_id.get(qid) if isinstance(qid, int) else None
        if q is None or q["section"] != "belirti":
            continue
        if "answer" in a and "selected" not in a:
            answer = str(a.get("answer") or "").strip()
        else:
            answer = _compose_answer(q, a.get("selected", ""), a.get("other", ""))
        if not answer:
            continue
        parts.append(f"Görüşmeci: {q['text']}\nDanışan: {answer}")
    return "\n\n".join(parts)


def build_preinfo_summary(answers: list[dict]) -> list[dict]:
    """Ön bilgi (ÖB) yanıtlarını raporda gösterilecek bağlam listesine çevirir.

    Tanı modeline verilmez; yalnızca klinisyene sunulan bağlamdır.
    """
    by_id = {q["id"]: q for q in INTERVIEW_QUESTIONS}
    out: list[dict] = []
    for a in answers:
        qid = a.get("id")
        q = by_id.get(qid) if isinstance(qid, int) else None
        if q is None or q["section"] != "on_bilgi":
            continue
        answer = _compose_answer(q, a.get("selected", ""), a.get("other", ""))
        if not answer:
            continue
        out.append({"key": q["key"], "question": q["text"], "answer": answer})
    return out


def critical_flags(answers: list[dict]) -> list[dict]:
    """Uzman/güvenlik uyarısı gerektiren yanıtları işaretler.

    Uyarılı bir soruda danışan ilk (olumsuz) şık veya çekimser dışında bir
    yanıt verdiyse bayrak üretir; Q6 (kendine zarar) her zaman öncelikli.
    """
    by_id = {q["id"]: q for q in INTERVIEW_QUESTIONS}
    flags: list[dict] = []
    for a in answers:
        qid = a.get("id")
        q = by_id.get(qid) if isinstance(qid, int) else None
        # Yalnızca belirti sorularının uyarıları klinik bayraktır; ÖB
        # uyarıları bilgilendirme notudur (risk yükseltmez).
        if q is None or q["section"] != "belirti" or not q["warn"]:
            continue
        selected = str(a.get("selected", "")).strip()
        other = str(a.get("other", "")).strip()
        answer = _compose_answer(q, selected, other)
        if not answer:
            continue
        negative = q["options"][0] if q["options"] else ""
        if selected == negative or selected in _DECLINE:
            continue
        flags.append({"key": q["key"], "question": q["text"],
                      "answer": answer, "warn": q["warn"]})
    return flags


__all__ = [
    "INTERVIEW_QUESTIONS",
    "PREINFO_KEYS",
    "InterviewQuestion",
    "build_interview_transcript",
    "build_preinfo_summary",
    "critical_flags",
]
