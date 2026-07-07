# Psikiyatri AI — Multimodal Karar Destek Sistemi

> **Yapay Zekaya Dayalı Derin Öğrenme ile Psikiyatrik/Psikolojik Hastalıklarda Karar Vermeyi Destekleme**

| | |
|---|---|
| **Firma** | Eurotech Bilişim A.Ş. |
| **Proje Kodu** | Eurotech-1 |
| **Proje Yöneticisi** | Prof. Dr. Metin Toprak |
| **Süre** | 18 Ay (02.02.2025 – 02.08.2026) |
| **Teknoloji Hazırlık Seviyesi** | TRL6 — Prototip, benzetilmiş ortamda test |
| **Ana Teknoloji Alanı** | Yapay Zeka, Bilgisayar Yazılımları |

---

## Amaç

Psikiyatrik/psikolojik hastalıkların teşhisinde uzmana **karar destek** sağlayan, üç veri kaynağını (modalite) birleştiren bir yapay zeka sistemi geliştirmek:

1. **Görüntü (Vision)** — Yüz ifadesi, göz/yüz kası hareketleri, mikro-ifadeler
2. **Ses (Audio)** — Ses tonu, frekans, hız, tonlama (MFCC, eGeMAPS)
3. **Metin (Text/NLP)** — Konuşma içeriğinin doğal dil işleme ile analizi (Türkçe)
4. **Füzyon (Fusion)** — Modaliteler arası **uyum/uyumsuzluk** analizi

> ⚠️ Bu sistem **klinik karar desteği** amaçlıdır; hekimin/uzmanın yerini almaz, teşhisi yalnızca **destekler**.

## Ayırt Edici Yön

Mevcut çalışmalar modaliteleri senkron analiz ederken, bu proje **ses tonu ↔ yüz ifadesi ↔ konuşma içeriği** arasındaki *uyumsuzlukları* saptayarak bastırılmış duygular ve içsel çatışmalar gibi belirtileri ortaya çıkarmayı hedefler.

## Mimari

```
                 ┌──────────────┐
  video/görüntü ─►  Vision      ├──┐
                 └──────────────┘  │
                 ┌──────────────┐  │   ┌──────────┐   ┌───────────────┐
  ses kaydı    ─►  Audio        ├──┼──►│  Fusion  ├──►│ Karar Desteği │
                 └──────────────┘  │   └──────────┘   │  (olasılık +  │
                 ┌──────────────┐  │                  │  açıklama)    │
  konuşma metni─►  Text (NLP)   ├──┘                  └───────────────┘
                 └──────────────┘
```

Ayrıntı: [`docs/mimari.md`](docs/mimari.md)

## Proje Yapısı

```
psikiyatri-ai/
├── src/psikiyatri_ai/     # Ana Python paketi
│   ├── config.py          # Merkezi yapılandırma (pydantic-settings)
│   ├── data/              # Veri alma, anonimleştirme, veri modülü
│   ├── modalities/        # vision / audio / text alt-modelleri
│   ├── fusion/            # Multimodal füzyon modeli
│   ├── training/          # Eğitim ve değerlendirme
│   └── api/               # FastAPI servis katmanı
├── config/                # YAML yapılandırmaları
├── data/                  # Veri (git'e dahil değil)
├── docs/                  # Proje dokümantasyonu
├── notebooks/             # Araştırma/deney defterleri
├── tests/                 # pytest testleri
└── docker/                # Konteynerizasyon
```

## Kurulum

```bash
# Python 3.11+ gereklidir
python -m venv .venv
.venv\Scripts\activate        # Windows (PowerShell: .venv\Scripts\Activate.ps1)
pip install -e ".[dev]"

# Ortam değişkenlerini ayarla
copy .env.example .env
```

## Kullanım

```bash
# API servisini başlat
uvicorn psikiyatri_ai.api.main:app --reload

# Testleri çalıştır
pytest

# Kod kalitesi
ruff check src tests
```

## Etik ve Veri Gizliliği

Sağlık verisi işlendiği için proje **KVKK** ve uluslararası etik standartlara uyumlu tasarlanır. Tüm hasta verisi işleme öncesi **anonimleştirilir**. Ayrıntı: [`docs/veri-yonetimi-ve-etik.md`](docs/veri-yonetimi-ve-etik.md).

## Yol Haritası

18 aylık iş planı: [`docs/yol-haritasi.md`](docs/yol-haritasi.md)

## Lisans

Özel/Tescilli (Proprietary) — Eurotech Bilişim A.Ş. Tüm hakları saklıdır.
