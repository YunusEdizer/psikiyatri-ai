# Mimari

## Genel Bakış

Sistem, **dört modaliteyi** bağımsız kodlayıp ortak bir uzayda birleştiren
**geç füzyon (late fusion)** mimarisini kullanır. Bu yaklaşım eksik
modaliteye dayanıklıdır ve her modalitenin ayrı ölçeklenmesine izin verir.

```
Ham veri              Ön işleme                Kodlayıcı            Füzyon
─────────             ─────────                ─────────           ──────
video        ─► kare örnekle + yüz      ─► VisionEncoder  ─┐
                işaretçisi + geometrik AU                  │
                (MediaPipe)                                │
ses kaydı    ─► MFCC + eGeMAPS          ─► AudioEncoder   ─┤
                (librosa, openSMILE)                       ├─► MultimodalFusion ─► karar
konuşma      ─► normalize + maskele     ─► TextEncoder    ─┤     (dikkat +       + olasılık
                (BERTurk tokenizer)                        │      uyumsuzluk)     + açıklama
giyilebilir  ─► kalp/EEG/hareket → bant ─► WearableEncoder ┘
                gücü + titreme indeksi
```

## Katmanlar

| Katman | Paket | Sorumluluk |
|--------|-------|------------|
| Veri | `data/` | Alma, **anonimleştirme**, denetim günlüğü (`audit.py`), veri sözleşmeleri |
| Modalite | `modalities/{vision,audio,text,wearable}/` | Ön işleme + kodlayıcı (embedding üretimi) |
| Füzyon | `fusion/` | Dikkat tabanlı birleştirme + uyumsuzluk skoru |
| Eğitim | `training/` | Uçtan uca model, eğitim/değerlendirme, checkpoint yan-verisi |
| Servis | `api/` | FastAPI karar destek uç noktaları + klinik panel |
| Yardımcı | `utils/` | Metrikler (ikili + çok-sınıflı) |

## Modalite Sözleşmesi

Her modalite kodlayıcısı `nn.Module` olup `(B, ...) -> (B, embed_dim)`
üretir. Füzyon katmanı yalnızca gömme sözlüğü görür; modalitenin iç
detayından bağımsızdır. Bu sözleşme sayesinde, proje formunda belirtilen
**giyilebilir teknoloji** modalitesi (`modalities/wearable/`) sisteme yeni
bir kodlayıcı olarak eklenmiştir; ileride başka sensörler (ör. saf EEG
kanalı) aynı yolla takılabilir.

## Görüntü: Geometrik AU-Vekili Özellikler

`modalities/vision/preprocess.py`, 468 MediaPipe işaretçisinden
yorumlanabilir geometrik ölçütler türetir: göz açıklığı (EAR, kırpma/göz
hareketi), ağız köşesi eğimi (gülümseme/düşük ağız köşesi), kaş hareketi.
Bu ölçütler FACS Eylem Birimlerinin hafif bir vekilidir; hem uzmana
sunulur hem de modele ek öznitelik olarak verilebilir (bkz.
`FaceFeatures.action_units`).

## Giyilebilir Modalite

`modalities/wearable/`, akıllı saat/EEG/biyometrik sensör serilerini
(kalp atış hızı, HRV, cilt iletkenliği, hareket, titreme, EEG bant
güçleri) sabit kanallı `(T, C)` bir diziye çevirir. Ham EEG/ivme
sinyallerinden **bant gücü** ve **titreme indeksi** (3–6 Hz; Parkinson
vekili) FFT ile hesaplanır. `WearableEncoder` çift yönlü GRU ile gömme
üretir.

## Çok-Sınıflı Tanı

Model ikili tarama (kontrol/depresyon) ile sınırlı değildir. Manifest
etiketlerinden sınıf kümesi otomatik çıkarılır (`infer_classes`) ve model
`num_classes` buna göre boyutlanır (kontrol + depresyon/anksiyete/bipolar/
şizofreni). Servis, etiket sırasını checkpoint yanındaki `classes.json`
yan-verisinden okuyup ikili/çok-sınıflı modeli aynı kodla sunar.

## Ayırt Edici Bileşen: Uyumsuzluk (Incongruence)

`fusion/incongruence.py`, modalite gömmeleri arasındaki ikili kosinüs
mesafesini [0, 1] aralığında bir skora indirger. Bu skor:

1. Füzyon sınıflandırıcısına **ek öznitelik** olarak verilir.
2. API yanıtında uzmana **açıklanabilir sinyal** olarak sunulur.

Hipotez: ses tonu / yüz ifadesi / konuşma içeriği arasındaki yüksek
uyumsuzluk, bastırılmış duygu ve içsel çatışma göstergesi olabilir.

## Eksik Modaliteye Dayanıklılık

`MultimodalFusion.forward`, gömme sözlüğünün alt kümesini kabul eder.
Bir seansta yalnızca ses + metin varsa model yine çalışır; dikkat
ağırlıkları mevcut modaliteler üzerinden normalize edilir.

## Açıklanabilirlik

Her tahmin şunları döndürür:
- Sınıf olasılıkları
- Modalite katkı ağırlıkları (dikkat)
- Uyumsuzluk skoru

Bu, "kara kutu" olmayan, uzmanın gerekçe görebildiği bir karar desteği
sağlar — klinik kabul için kritiktir.

## Klinik Yazılım Katmanı (Temmuz 2026)

Model hattının üzerine, başvuru iş planındaki "karar destek yazılımı"
taahhüdünü karşılayan bir uygulama katmanı eklenmiştir:

```
  Tarayıcı / Mobil (PWA)
        │  JWT
        ▼
  FastAPI ──► routes_auth      (giriş, kullanıcı yönetimi, roller)
        ├──► routes_patients   (hasta / görüşme / not / rapor / KVKK silme)
        │         │
        │         ├─► session.analyze_session  (modalite skorlama)
        │         ├─► decision.generate_decision (açıklanabilir profil)
        │         └─► db (SQLAlchemy: users, patients, interviews,
        │                  doctor_notes, analysis_results)
        └──► audit (JSONL denetim günlüğü — hesap verebilirlik)
```

- **`db/`** — ilişkisel kalıcılık; geliştirmede SQLite, üretimde
  `PAI_DATABASE_URL` ile PostgreSQL. Hasta silme kaskatlıdır (unutulma
  hakkı).
- **`api/security.py`** — PBKDF2 parola özeti, JWT (HS256), rol tabanlı
  bağımlılıklar (`require_roles`). Üretimde varsayılan sırlar reddedilir.
- **`decision/engine.py`** — kural tabanlı, açıklanabilir karar katmanı:
  model olasılıkları + semptom süresi + vaka şiddeti + doktor notu +
  uyumsuzluk → tanı profili, risk düzeyi/etkenleri, güven ve belirsizlik
  uyarıları, öneriler.
- **Panel (PWA)** — `api/static/` altındaki tek sayfalık uygulama;
  service worker uygulama kabuğunu önbellekler, **klinik veri asla
  önbelleklenmez**; mikrofon kaydı MediaRecorder ile alınır.
