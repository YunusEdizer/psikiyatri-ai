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
   e-posta gibi kalıntı kişisel veriler NLP'ye girmeden maskelenir.
4. **Gizli tuz yönetimi**: `PAI_ANON_SALT` üretimde bir sır yöneticisinde
   (secret manager) tutulur; koda veya depoya asla yazılmaz.

## Veri Depolama Kuralları

- Ham medya (`data/raw/`) **git'e girmez** (bkz. `.gitignore`).
- Veri en az ayrıcalık ilkesiyle, şifreli diskte saklanır.
- Erişim loglanır; üretimde `diagnose=False` ile hassas iz bırakılmaz.

## Veri Yaşam Döngüsü

```
toplama ─► rıza + etik onay ─► anonimleştirme ─► ön işleme ─► eğitim
                                                                 │
                                          saklama süresi sonu ◄──┘
                                          (güvenli imha)
```

## Model Kullanım Sınırı

Sistem **karar desteğidir, teşhis aracı değildir**. Her API yanıtı bu
uyarıyı (`disclaimer`) içerir. Nihai sorumluluk yetkili sağlık uzmanındadır.

## Yanlılık (Bias) ve Adalet

- Veri seti cinsiyet, yaş ve sosyoekonomik dağılım açısından dengelenir.
- Metrikler alt gruplar bazında ayrı raporlanır (yalnız toplam doğruluk
  yeterli değildir — bkz. `utils/metrics.py` duyarlılık/özgüllük).
