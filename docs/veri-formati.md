# Veri Formatı — Manifest

Eğitim, **anonimleştirilmiş** bir JSONL manifest'inden beslenir. Her satır
bir seansı (`Sample`) temsil eder. Manifest yolu varsayılan olarak:

```
<PAI_DATA_DIR>/processed/manifest.jsonl      # varsayılan: ./data/processed/manifest.jsonl
```

## Şema

`data/schema.py` içindeki `Sample` ile doğrulanır:

| Alan | Tip | Zorunlu | Açıklama |
|------|-----|:------:|----------|
| `pseudonym` | string | ✔ | Anonim hasta takma-kimliği (`pid_...`) — bkz. `data/anonymization.py` |
| `session_id` | string | ✔ | Seans tanımlayıcısı |
| `video_path` | string | ✖ | Görüntü modalitesi dosya yolu |
| `audio_path` | string | ✖ | Ses modalitesi dosya yolu |
| `transcript` | string | ✖ | **Maskeli** konuşma metni (PII temizlenmiş) |
| `wearable_path` | string | ✖ | Giyilebilir sensör kaydı (`.csv`/`.npz`) — kanallar: `kalp_atis_hizi, hrv, cilt_iletkenligi, hareket, titreme, eeg_delta, eeg_theta, eeg_alpha, eeg_beta` |
| `features_path` | string | ✖ | Önceden çıkarılmış öznitelik dosyası (`.npz`). Verilirse ham medya yerine bu okunur (hızlı yol) |
| `consent` | bool | ✖ | Katılımcı **açık rızası** alındı mı (KVKK) — varsayılan `false` |
| `consent_ref` | string | ✖ | İmzalı onam formu referansı |
| `label` | string | ✔* | Etiket: `kontrol`, `depresyon`, `anksiyete` (YAB), `sosyal_anksiyete`, `panik`, `okb`, `tssb`, `bipolar`, `sizofreni`, `belirsiz` |

\* Eğitim için etiket zorunludur; çıkarımda gerekmez. Manifest'te bulunan
etiketlere göre model **ikili** (kontrol/depresyon) veya **çok-sınıflı**
(kontrol + çeşitli tanılar) olarak otomatik boyutlanır (`infer_classes`).

## Örnek (`manifest.jsonl`)

```json
{"pseudonym":"pid_a1b2c3d4e5f6a7b8","session_id":"S0001","video_path":"raw/S0001/video.mp4","audio_path":"raw/S0001/audio.wav","transcript":"[TCKN] hasta bugün kendini yorgun hissettiğini belirtti.","label":"depresyon"}
{"pseudonym":"pid_9f8e7d6c5b4a3210","session_id":"S0002","video_path":"raw/S0002/video.mp4","audio_path":"raw/S0002/audio.wav","transcript":"Danışan uyku düzeninin iyi olduğunu ifade etti.","label":"kontrol"}
```

## Önemli Kurallar

1. **Anonimlik**: Manifest'e ad, TC kimlik, dosya no gibi doğrudan
   tanımlayıcı **yazılmaz**. `pseudonym` üretimi için `pseudonymize()`,
   transkript maskelemesi için `mask_pii()` kullanın.
2. **Yol tabanı**: Göreli yollar `PAI_DATA_DIR` köküne göre çözülür.
3. **Tekdüze modalite**: Bir eğitim batch'indeki tüm örnekler aynı
   modalite kümesine sahip olmalıdır (kontrollü toplama). Kısmi modalite
   yalnızca çıkarımda (API füzyon arayüzü) desteklenir.
4. **Sınıf dengesi**: `build_dataloaders` sınıf-dengeli (stratified)
   eğitim/doğrulama ayrımı yapar.

## Manifest Üretimi

İki yol vardır:

### 1. Gerçek ham veriden (üretim) — `scripts/build_manifest.py`

Ham seans klasörlerini tarar, anonimleştirir ve manifest üretir. Beklenen
ham yapı:

```
data/raw/sessions/<session_id>/
    video.mp4        (opsiyonel)
    audio.wav        (opsiyonel)
    wearable.csv     (opsiyonel — giyilebilir sensör serisi)
    transcript.txt   (opsiyonel)
    meta.json        -> {"patient_id": "...", "label": "depresyon",
                         "consent": true, "consent_ref": "ONAM-2025-014"}
```

```bash
python scripts/build_manifest.py --raw-root data/raw/sessions
```

`patient_id` geri döndürülemez takma-kimliğe çevrilir; transkript PII
maskelemesinden geçer; ham medya **kopyalanmaz**, yalnızca yolları kaydedilir.

```
ham seans klasörü ─► anonimleştir (pseudonym + mask) ─► manifest.jsonl
```

### 2. Sentetik demo veri (geliştirme) — `scripts/generate_sample_data.py`

Gerçek veri hazır olmadan tüm hattı denemek için önceden çıkarılmış
öznitelik (`.npz`) + manifest üretir. Sınıflara öğrenilebilir sinyal
enjekte edildiğinden eğitim gerçekten yakınsar.

```bash
python scripts/generate_sample_data.py --count 300 --seed 42 --signal 0.8
psikiyatri-train        # üretilen manifest ile eğitim (modaliteler otomatik algılanır)
```

> Üretilen sentetik veri tamamen yapaydır, hiçbir gerçek hastayı temsil etmez.

### 3. Dış Excel NLP veri setinden — `scripts/ingest_nlp_dataset.py`

Türkçe psikiyatrik NLP korpusunu (`psychiatric_nlp_dataset_*.xlsx`)
anonim manifest'e çevirir. Ham Excel `data/external/` altında tutulur
(git'e **girmez**).

```bash
# Excel'i data/external/ altına koyun, sonra:
python scripts/ingest_nlp_dataset.py \
    --xlsx data/external/psychiatric_nlp_dataset_20260627.xlsx
# -> data/processed/nlp_manifest.jsonl (metin modalitesi)

psikiyatri-train --manifest data/processed/nlp_manifest.jsonl
```

Aktarım ilkeleri (bkz. `psikiyatri_ai.data.nlp_dataset`):

1. **Anonimlik (KVKK).** `Hasta_ID` geri döndürülemez takma-kimliğe
   (`pseudonymize`) çevrilir; transkript PII maskelemesinden (`mask_pii`)
   geçer.
2. **Etiket sızıntısını önleme.** Transkript yalnızca hasta-sesi anket
   cevaplarından (**Q1–Q24**) ve tanı adı temizlenmiş (`[TANI]`) uzun
   anlatıdan kurulur. Doktor notu / kamera / ses analiz sütunları tanının
   adını açıkça içerdiğinden **dışarıda bırakılır** — aksi hâlde metin
   modeli hastalık adını ezberler.
3. **Sadık taksonomi.** Veri setindeki 8 tanı, genişletilmiş
   `DiagnosisLabel` taksonomisine birebir eşlenir:

   | Veri seti tanısı | Etiket |
   |------------------|--------|
   | Majör Depresif Bozukluk | `depresyon` |
   | Yaygın Anksiyete Bozukluğu | `anksiyete` |
   | Sosyal Anksiyete Bozukluğu | `sosyal_anksiyete` |
   | Panik Bozukluk | `panik` |
   | Obsesif Kompulsif Bozukluk | `okb` |
   | Travma Sonrası Stres Bozukluğu | `tssb` |
   | Bipolar Bozukluk | `bipolar` |
   | Şizofreni | `sizofreni` |

> Bu korpus tümüyle **metin (NLP)** modalitesidir; ses/görüntü/giyilebilir
> alanları boştur. Model, manifest'te bulunan etiketlere göre otomatik
> boyutlanır (`infer_classes`).
