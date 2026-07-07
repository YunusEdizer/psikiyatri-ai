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
- [x] `/session/analyze` — dosya yükleme + KVKK maskeleme + risk raporu
- [ ] Gerçek veriyle ilk eğitim ve temel (baseline) metrikler

## Riskler ve Azaltım

| Risk | Azaltım |
|------|---------|
| Veri toplama gecikmesi (etik onay) | Faz 1'de erken başvuru; kamuya açık veri setleriyle ön prototip |
| Sınıf dengesizliği | Dengeli örnekleme + dengeli doğruluk metriği |
| Modalite eksikliği | Füzyon modeli eksik modaliteye dayanıklı tasarlandı |
| Model açıklanamazlığı | Dikkat + uyumsuzluk skoru uzmana sunulur |
