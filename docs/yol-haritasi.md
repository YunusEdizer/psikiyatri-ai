# Yol Haritası (18 Ay)

Proje formundaki 6 aşamalı iş planına (Tasarım → Veri Toplama → Ön İşleme
→ Özellik Birleştirme → Model Geliştirme → Doğrulama) dayanır.

| Faz | Ay | Aşama | Çıktı |
|-----|-----|-------|-------|
| 1 | 1–2 | **Tasarım & Altyapı** | Kesitsel-deneysel araştırma tasarımı, etik kurul başvurusu, proje iskeleti (bu depo), veri sözleşmeleri |
| 2 | 3–6 | **Veri Toplama** | Deney (tanılı) + kontrol (sağlıklı) grubu video/ses/metin veri seti; anonimleştirme hattı |
| 3 | 6–9 | **Ön İşleme** | Yüz işaretçisi, MFCC/eGeMAPS, NLP normalizasyon hatlarının otomasyonu |
| 4 | 9–11 | **Özellik Birleştirme** | Modalite kodlayıcıları + ortak uzay izdüşümü; uyumsuzluk metriği doğrulaması |
| 5 | 11–15 | **Model Geliştirme** | Füzyon modeli eğitimi, hiperparametre optimizasyonu, ablasyon çalışmaları |
| 6 | 15–18 | **Doğrulama & Prototip** | Deney/kontrol grubunda performans testi (duyarlılık/özgüllük), TRL6 prototip, API + arayüz, bilimsel yayın |

## Gantt Modül Fazları ile Eşleme

Proje formundaki Gantt şeması, süreç fazlarına ek olarak **modül-odaklı**
çubuklar içerir. Bu depodaki modül karşılıkları:

| Gantt çubuğu | Modül / Paket | Durum |
|--------------|---------------|:-----:|
| Veri Toplama ve Ön İşleme | `data/`, `modalities/*/preprocess` | ✅ İskelet + sentetik hat |
| Görüntü İşleme Modülü | `modalities/vision/` (landmark + geometrik AU) | ✅ |
| Ses İşleme Modülü | `modalities/audio/` (MFCC + eGeMAPS + GRU) | ✅ |
| Doğal Dil İşleme Modülü | `modalities/text/` (BERTurk + sözlük ön analiz) | ✅ |
| **Giyilebilir Teknoloji Entegrasyonu** | `modalities/wearable/` (kalp/EEG/hareket → bant gücü/titreme + GRU) | ✅ |
| Modalite Entegrasyonu ve Test | `fusion/`, `training/`, `tests/` | ✅ |

## Aşama 1 — Şu Anki Durum ✅

- [x] Proje iskeleti ve paket yapısı
- [x] Merkezi yapılandırma (`config.py`) + loglama
- [x] Anonimleştirme modülü + testler (KVKK temeli)
- [x] Modalite kodlayıcı iskeletleri (vision / audio / text)
- [x] Füzyon modeli + uyumsuzluk metriği + testler
- [x] FastAPI karar destek servisi + testler
- [x] Metrikler (duyarlılık/özgüllük/dengeli doğruluk)

## Aşama 2 — Devam Ediyor 🔄

- [x] `data/datamodule.py` — PyTorch `Dataset`/`DataLoader` (anonim veri, stratified ayrım)
- [x] `data/batch.py` — `SampleFeatures` / `ModalityBatch` veri yapıları
- [x] `FeatureExtractor` — manifest → yüz işaretçisi / MFCC+eGeMAPS / BERTurk token
- [x] Modalite kodlayıcı çıktılarını uçtan uca bağlama (`training/model.py`)
- [x] Uçtan uca eğitim döngüsü (`training/train.py`: `fit`, doğrulama, checkpoint)
- [x] `SyntheticDataset` ile hat doğrulaması (smoke) + testler
- [x] Manifest formatı dokümantasyonu (`docs/veri-formati.md`)
- [x] `scripts/build_manifest.py` — ham seans → anonim manifest üretimi
- [x] `scripts/generate_sample_data.py` — sentetik demo veri (öğrenilebilir sinyal)
- [x] Önceden çıkarılmış öznitelik (`.npz`) yolu — hızlı, bağımlılıksız eğitim
- [x] Dosya-tabanlı uçtan uca eğitim doğrulandı (denge-doğruluk 0.73 → 1.00 sentetik veride)
- [ ] Deney protokolü ve etiket taksonomisinin klinik ekiple netleştirilmesi
- [ ] Veri toplama SOP'u (standart işletim prosedürü) ve rıza formları
- [x] CI (test + ruff + mypy) kurulumu — GitHub Actions, Py 3.11/3.12 matrisi
- [x] Klinik karar destek paneli (`/`) — kayıt yükleme + transkript, klinik rapor
- [x] Türkçe sözlük-tabanlı metin ön analizi (`modalities/text/heuristics.py`)
- [x] `/session/analyze` — dosya yükleme + KVKK maskeleme + **gerçek füzyon skorlaması** (bağımlılık varsa, yoksa metin ön analizine zarif geri düşüş)
- [x] **Giyilebilir modalite** (`modalities/wearable/`) — sinyal işleme + kodlayıcı + füzyon entegrasyonu
- [x] **Görüntü geometrik AU-vekili** özellikler (göz açıklığı/kırpma, ağız köşesi, kaş hareketi)
- [x] **Çok-sınıflı (çok-hastalık) hat** — `num_classes` manifestten otomatik; `classes.json` yan-verisi; çok-sınıflı metrikler
- [x] KVKK derinleştirme — isim maskeleme, açık rıza alanları, denetim günlüğü (`audit.py`), üretimde salt zorunluluğu
- [ ] Gerçek veriyle ilk eğitim ve temel (baseline) metrikler

## Aşama 3 — Klinik Yazılım Katmanı ✅ (Temmuz 2026)

Başvuru iş planındaki "karar destek yazılımı" taahhütlerinin karşılığı:

- [x] **Veritabanı katmanı** (`db/`) — kullanıcı, hasta, görüşme, doktor
      notu, analiz/risk tabloları (SQLAlchemy; SQLite → `PAI_DATABASE_URL`
      ile PostgreSQL)
- [x] **Kimlik doğrulama + roller** (`api/security.py`) — JWT, PBKDF2
      parola özeti; yönetici / psikiyatrist / psikolog yetkileri
- [x] **Hasta/görüşme API'si** (`api/routes_patients.py`) — kayıt, liste,
      profil, görüşme analizi (kalıcı), not ekleme, denetim uçları
- [x] **Karar destek motoru** (`decision/engine.py`) — model çıktısı +
      semptom süresi + vaka şiddeti + doktor notu → açıklanabilir tanı
      profili, risk etkenleri, güven/belirsizlik uyarıları, öneriler
- [x] **Düşük güvenli sonuç uyarıları** — belirsiz olasılık farkı, tek
      modalite, eğitilmemiş model, yüksek uyumsuzluk bayrakları
- [x] **Klinik panel v2** — giriş, hasta listesi → profil → görüşme
      geçmişi → yeni görüşme → kalıcı rapor; yönetim ekranı (kullanıcı +
      denetim günlüğü)
- [x] **Mobil (PWA)** — manifest + service worker + mikrofonla ses kaydı;
      "Ana ekrana ekle" ile mobil uygulama deneyimi
- [x] **KVKK tamamlama** — unutulma hakkı (`DELETE /patients/{id}`,
      kaskatlı), üretimde zorunlu sırlar (JWT anahtarı, yönetici parolası),
      uyum matrisi (`docs/kvkk-uyum-matrisi.md`)
- [x] **Pilot dokümanları** — kullanıcı kılavuzu, hızlı başlangıç rehberi

### Kalan işler (pilot + kapanış, Ağustos 2026)

- [ ] Gerçek/kamuya açık veriyle temel (baseline) eğitim + metrik raporu
- [ ] Uzman kullanım testleri (gerçekçi vaka senaryoları) ve geri bildirim turu
- [ ] Pilot kurulum (PostgreSQL + HTTPS) ve performans ölçümü
- [ ] İsteğe bağlı: yerel (native) mobil istemci — PWA taahhüdü karşılıyor,
      native sürüm ürünleşme sonrasına planlandı

## Riskler ve Azaltım

| Risk | Azaltım |
|------|---------|
| Veri toplama gecikmesi (etik onay) | Faz 1'de erken başvuru; kamuya açık veri setleriyle ön prototip |
| Sınıf dengesizliği | Dengeli örnekleme + dengeli doğruluk metriği |
| Modalite eksikliği | Füzyon modeli eksik modaliteye dayanıklı tasarlandı |
| Model açıklanamazlığı | Dikkat + uyumsuzluk skoru uzmana sunulur |
