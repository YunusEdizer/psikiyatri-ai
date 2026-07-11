# Veri Yönetimi ve Etik

> Bu proje sağlık verisi (özel nitelikli kişisel veri) işler. Aşağıdaki
> ilkeler proje formundaki "Etik Veri Kullanımı" taahhüdünün teknik
> karşılığıdır ve pazarlığa kapalıdır.

## Yasal Çerçeve

- **KVKK** (6698 sayılı Kişisel Verilerin Korunması Kanunu) — sağlık
  verisi "özel nitelikli" kategoridedir; açık rıza ve ek güvenlik şarttır.
- **GDPR** (yurt dışı işbirliği/ihracat senaryosu için).
- **Etik kurul onayı** — klinik veri toplama öncesi ilgili üniversite/
  hastane etik kurulundan onay alınmalıdır.

## Anonimleştirme İlkeleri

1. **Toplama noktasında ayrıştırma**: doğrudan tanımlayıcılar (ad, TC
   kimlik, dosya no) ham veriden ayrılır; modele hiçbir zaman girmez.
2. **Takma-kimlikleme (pseudonymization)**: `data/anonymization.py`,
   HMAC-SHA256 + gizli tuz ile geri döndürülemez takma-kimlik üretir.
   Aynı kişi farklı seanslarda aynı takma-kimliği alır (boylamsal analiz).
3. **Metin maskeleme**: konuşma transkriptlerindeki TC kimlik, telefon,
   e-posta ve (sezgisel olarak, unvan/hitapla geçen) özel adlar NLP'ye
   girmeden maskelenir. Kapsamlı ad tespiti ileri fazda Türkçe NER ile
   güçlendirilecektir.
4. **Gizli tuz yönetimi**: `PAI_ANON_SALT` üretimde bir sır yöneticisinde
   (secret manager) tutulur; koda veya depoya asla yazılmaz. Üretim
   ortamında (`PAI_ENV=production`) varsayılan tuz **kod tarafından
   reddedilir** (bkz. `config.Settings` doğrulayıcısı).

## Açık Rıza (Consent)

Her `Sample`, `consent` (bool) ve `consent_ref` (imzalı onam referansı)
alanlarını taşır. Rıza durumu manifest'e `meta.json`'dan taşınır ve
denetim günlüğüne kaydedilir. Rızasız kayıtların eğitime alınmaması
klinik SOP ile denetlenir.

## Denetim Günlüğü (Audit Log)

KVKK hesap verebilirlik ilkesi gereği erişim/işleme olayları
`data/audit.py` ile satır-JSON olarak iz bırakır (varsayılan:
`<data>/audit/audit.jsonl`; git dışıdır). Günlüğe yalnızca takma-kimlik
ve olay meta verisi yazılır; doğrudan tanımlayıcı asla yazılmaz. Manifest
üretimi ve seans analizi olayları kaydedilir.

## Veri Depolama Kuralları

- Ham medya (`data/raw/`) ve denetim günlüğü (`data/audit/`) **git'e
  girmez** (bkz. `.gitignore`).
- Veri en az ayrıcalık ilkesiyle, şifreli diskte saklanır.
- Erişim `data/audit.py` ile loglanır; günlükte hassas iz bırakılmaz.

## Veri Yaşam Döngüsü

```
toplama ─► rıza + etik onay ─► anonimleştirme ─► ön işleme ─► eğitim
   │                                                             │
   └─► denetim günlüğü (audit.py)         saklama süresi sonu ◄──┘
                                          (güvenli imha)
```

## Model Kullanım Sınırı

Sistem **karar desteğidir, teşhis aracı değildir**. Her API yanıtı bu
uyarıyı (`disclaimer`) içerir. Nihai sorumluluk yetkili sağlık uzmanındadır.

## Yanlılık (Bias) ve Adalet

- Veri seti cinsiyet, yaş ve sosyoekonomik dağılım açısından dengelenir.
- Metrikler alt gruplar bazında ayrı raporlanır (yalnız toplam doğruluk
  yeterli değildir — bkz. `utils/metrics.py` duyarlılık/özgüllük).
