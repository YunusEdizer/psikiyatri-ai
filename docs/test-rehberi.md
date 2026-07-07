# Test Rehberi

Projeyi test etmenin üç katmanı. Tüm komutlar proje kök dizininden çalışır:

```powershell
cd C:\Users\edize\source\repos\psikiyatri-ai
```

> İlk kez kuruyorsan: `pip install -e ".[dev]"`

---

## 1. Otomatik testler (en hızlı sağlık kontrolü)

```powershell
pytest                          # tüm testler (27) geçmeli
pytest --cov=psikiyatri_ai      # + kapsam raporu (~%80)
pytest tests/test_fusion.py -v  # tek dosya, ayrıntılı
ruff check src tests scripts    # kod kalitesi → "All checks passed!"
mypy src                        # tip denetimi → "Success"
```

**Beklenen:** `27 passed`. Bir şey bozulursa hangi test/satır olduğu bildirilir.

---

## 2. Eğitim hattı (uçtan uca öğreniyor mu?)

```powershell
# Sentetik veri üret (300 örnek + manifest)
python scripts/generate_sample_data.py --count 300 --signal 0.8

# Eğit — modaliteler otomatik algılanır, model öğrenir, checkpoint kaydeder
psikiyatri-train
```

**Beklenen:** denge-doğruluk `~0.9 → 1.0`, kayıp `~0.6 → 0.0`, `models/psikiyatri_model.pt` oluşur.
Bu, **manifest → DataLoader → kodlayıcı → füzyon → öğrenme** zincirinin çalıştığını kanıtlar.

> **Kontrol testi:** `--signal 0.0` verirsen model şansta (0.5) kalmalı — böylece
> modelin gerçekten veriden öğrendiğini (ezberlemediğini) doğrularsın.

---

## 3. API'yi canlı deneme (en görsel yol)

```powershell
python -m uvicorn psikiyatri_ai.api.main:app --reload
```

Sunucu açılınca tarayıcıda:

### En kolayı — Swagger arayüzü: http://127.0.0.1:8000/docs
Kod yazmadan `/predict`'e tıkla → **Try it out** → örnek JSON gönder → yanıtı gör.

Örnek istek gövdesi:
```json
{
  "session_id": "demo-1",
  "vision_embedding": [0.1, 0.1, "... toplam 256 değer ..."],
  "audio_embedding":  [0.2, 0.2, "... toplam 256 değer ..."]
}
```

Örnek yanıt:
```json
{
  "predicted_label": "depresyon",
  "probabilities": {"kontrol": 0.485, "depresyon": 0.515},
  "incongruence": 0.325,
  "modality_contributions": [
    {"modality": "vision", "weight": 0.482},
    {"modality": "audio",  "weight": 0.518}
  ],
  "disclaimer": "Bu çıktı klinik karar desteği amaçlıdır..."
}
```

Sağlık kontrolü: http://127.0.0.1:8000/health → `{"status":"ok"}`

Komut satırından (PowerShell) hızlı istek:
```powershell
$body = @{ session_id="demo-1"; vision_embedding=@(0..255 | % {0.1}); audio_embedding=@(0..255 | % {0.2}) } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -Body $body -ContentType "application/json"
```

---

## 4. GitHub'da otomatik test (CI)

Her `git push`'ta [Actions](https://github.com/YunusEdizer/psikiyatri-ai/actions) testleri
Linux'ta Python 3.11 + 3.12 ile otomatik koşar. Yeşil tik = her şey yolunda.
