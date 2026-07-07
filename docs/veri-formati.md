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
| `label` | string | ✔* | Etiket: `kontrol`, `depresyon`, `anksiyete`, `bipolar`, `sizofreni`, `belirsiz` |

\* Eğitim için etiket zorunludur; çıkarımda gerekmez.

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

## Manifest Üretimi (öneri)

Ham veriden manifest üreten betik ileride `scripts/build_manifest.py`
altına eklenecektir; anonimleştirme adımı bu betiğin ilk aşamasıdır:

```
ham seans klasörü ─► anonimleştir (pseudonym + mask) ─► manifest.jsonl
```
