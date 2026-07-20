"""Kontrol (sağlıklı) sınıfı için sentetik örnek üretimi.

NLP korpusunda kontrol (sağlıklı birey) sınıfı YOKTUR; bu nedenle model
"belirgin bulgu yok" kararını üretemez (bkz. çekimserlik çözümü,
``api/session.py``). Kalıcı çözüm gerçek kontrol grubu verisidir; o
toplanana dek, korpusun kendisi de sentetik olduğundan **yöntemsel olarak
tutarlı** biçimde kontrol örnekleri üretiriz.

Her örnek, 24 tarama sorusuna (Q1-Q24) sağlıklı/patolojik-olmayan doğal
Türkçe yanıtlardan kurulur (havuzdan tohumlu rastgele seçim → çeşitlilik).
Üretilen metin korpus stiliyle (satır = bir Q yanıtı) aynıdır; böylece
eğitimde aynı şemaya oturur. Etiket: ``kontrol``.

> Üretilen kontrol verisi tamamen yapaydır, hiçbir gerçek bireyi temsil
> etmez; yalnızca sınıf dengesini ve "sağlıklı" kararını mümkün kılmak
> içindir.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from .anonymization import pseudonymize
from .schema import DiagnosisLabel, Sample

CONSENT_REF = "SENTETIK-KONTROL/uretilmis (gercek birey degildir)"

# Her tarama sorusu (Q1-Q24) için sağlıklı/patolojik-olmayan yanıt havuzu.
# Doğal birinci-şahıs Türkçe; kademeli çeşitlilik için 3-4 varyant.
_HEALTHY: dict[int, list[str]] = {
    1: ["Kendimi genel olarak iyi ve dengeli hissediyorum.",
        "Son zamanlarda ruh hâlim iyi, keyfim yerinde.",
        "Genel olarak huzurlu ve enerjik hissediyorum."],
    2: ["Sevdiğim etkinliklerden eskisi gibi keyif alıyorum.",
        "Hobilerime ilgim yerinde, keyifle vakit geçiriyorum.",
        "Eskiden zevk aldığım şeylerden hâlâ keyif alıyorum."],
    3: ["Uykum düzenli ve dinlendirici.",
        "Gece rahat uyuyor, dinlenmiş uyanıyorum.",
        "Uyku düzenimde bir sorun yok."],
    4: ["İştahım normal, belirgin bir değişiklik yok.",
        "Yeme düzenim her zamanki gibi.",
        "İştahımda kayda değer bir değişiklik olmadı."],
    5: ["Kendimi değersiz veya suçlu hissetmiyorum.",
        "Özgüvenim yerinde, gereksiz suçluluk duymuyorum.",
        "Değersizlik ya da suçluluk hissim yok."],
    6: ["Kendime zarar verme düşüncem yok.",
        "Yaşamıma dair olumsuz bir düşüncem kesinlikle yok.",
        "Kendime zarar verme gibi bir düşünceye hiç kapılmadım."],
    7: ["Belirgin bir kaygı yaşamıyorum.",
        "Genelde sakin ve rahatım, yoğun endişem yok.",
        "Kaygı düzeyim normal, beni zorlayan bir endişe yok."],
    8: ["Panik atak geçirmiyorum.",
        "Ani yoğun korku nöbetlerim olmuyor.",
        "Panik benzeri bir atak yaşamıyorum."],
    9: ["İnsanların beni yargıladığını düşünmüyorum.",
        "Başkalarının değerlendirmesi beni tedirgin etmiyor.",
        "Yargılanma kaygım yok."],
    10: ["Sosyal ortamlara rahatça giriyorum.",
         "İnsanlarla bir arada olmaktan keyif alıyorum.",
         "Sosyal ortamlardan kaçınmıyorum."],
    11: ["Rahatsız edici istem dışı düşüncelerim yok.",
         "Zihnime takılan rahatsız edici düşünceler olmuyor.",
         "İstem dışı düşünce sorunum yok."],
    12: ["Tekrarlayan davranışlarım yok.",
         "Kontrol etme, sayma gibi tekrarlarım olmuyor.",
         "Ritüel tarzı tekrar davranışım yok."],
    13: ["Geçmiş olaylar günlük hayatımı meşgul etmiyor.",
         "Eski kötü anılar istemsizce aklıma gelmiyor.",
         "Geçmiş beni rahatsız edecek biçimde canlanmıyor."],
    14: ["Kabus görmüyorum, uykum huzurlu.",
         "Rahatsız edici kabuslarım olmuyor.",
         "Kötü rüya sorunu yaşamıyorum."],
    15: ["Aşırı enerjik veya coşkulu dönemlerim olmuyor.",
         "Olağandışı taşkın enerji dönemlerim yok.",
         "Enerjim dengeli, aşırı coşku yaşamıyorum."],
    16: ["Az uyuduğumda yorgun hissederim, bu olağan.",
         "Uykusuz kaldığımda dinç olmam, normal tepki veriyorum.",
         "Az uykuyla enerjik kalma gibi bir durumum yok."],
    17: ["Riskli veya ani kararlar alma eğilimim yok.",
         "Kararlarımı düşünerek veririm, dürtüsel davranmam.",
         "Aşırı riskli kararlar almıyorum."],
    18: ["Başkalarının duymadığı sesler duymuyorum.",
         "İşitsel bir yanılsama yaşamıyorum.",
         "Olmayan sesler duyduğum olmadı."],
    19: ["Olmayan şeyler görmüyorum.",
         "Görsel bir yanılsama yaşamıyorum.",
         "Başkalarının görmediği şeyleri görmüyorum."],
    20: ["İnsanların bana zarar vermeye çalıştığını düşünmüyorum.",
         "Çevremdekilere güveniyorum, komplo hissim yok.",
         "Zarar göreceğime dair bir kuşkum yok."],
    21: ["Düşüncelerimin kontrol edildiği hissine kapılmıyorum.",
         "Zihnime dışarıdan müdahale edildiğini düşünmüyorum.",
         "Düşüncelerim bana ait, kontrol edilme hissim yok."],
    22: ["Duygularım genel olarak dengeli.",
         "Gün içinde ani duygu değişimleri yaşamıyorum.",
         "Duygudurumum tutarlı ve dengeli."],
    23: ["Yakın ilişkilerim istikrarlı.",
         "İlişkilerimde belirgin iniş çıkış yaşamıyorum.",
         "Yakın ilişkilerim dengeli ve sağlıklı."],
    24: ["Öfke patlamaları yaşamıyorum.",
         "Öfkemi kontrol edebiliyorum, taşkınlık olmuyor.",
         "Ani öfke patlamalarım yok."],
}


def _build_transcript(rng: random.Random) -> str:
    """Her Q için havuzdan bir sağlıklı yanıt seçip transkript kurar."""
    return "\n".join(rng.choice(_HEALTHY[i]) for i in range(1, 25))


def iter_control_samples(
    count: int, *, seed: int = 20260720, salt: str | None = None
) -> Iterator[Sample]:
    """``count`` adet sentetik kontrol (sağlıklı) örneği üretir."""
    rng = random.Random(seed)
    for i in range(count):
        sid = f"KONTROL_{i + 1:05d}"
        yield Sample(
            pseudonym=pseudonymize(sid, salt=salt),
            session_id=sid,
            transcript=_build_transcript(rng),
            label=DiagnosisLabel.KONTROL,
            consent=True,
            consent_ref=CONSENT_REF,
        )


__all__ = ["CONSENT_REF", "iter_control_samples"]
