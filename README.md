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

Psikiyatrik/psikolojik hastalıkların teşhisinde uzmana **karar destek** sağlayan, dört veri kaynağını (modalite) birleştiren bir yapay zeka sistemi geliştirmek:

1. **Görüntü (Vision)** — Yüz işaretçileri + geometrik AU-vekili özellikler (göz açıklığı/kırpma, ağız köşesi, kaş hareketi)
2. **Ses (Audio)** — Ses tonu, frekans, hız, tonlama (MFCC, eGeMAPS)
3. **Metin (Text/NLP)** — Konuşma içeriğinin doğal dil işleme ile analizi (Türkçe)
4. **Giyilebilir (Wearable)** — Kalp atış hızı, cilt iletkenliği, EEG bant güçleri, hareket/titreme (Parkinson vekili)

Bu modaliteler **Füzyon (Fusion)** katmanında dikkat tabanlı birleştirilir ve modaliteler arası **uyum/uyumsuzluk** analizi yapılır. Sistem hem **ikili tarama** (kontrol/depresyon) hem de **çok-hastalık** (anksiyete/bipolar/şizofreni) sınıflandırmasını destekler.

> ⚠️ Bu sistem **klinik karar desteği** amaçlıdır; hekimin/uzmanın yerini almaz, teşhisi yalnızca **destekler**.

## Ayırt Edici Yön

Mevcut çalışmalar modaliteleri senkron analiz ederken, bu proje **ses tonu ↔ yüz ifadesi ↔ konuşma içeriği** arasındaki *uyumsuzlukları* saptayarak bastırılmış duygular ve içsel çatışmalar gibi belirtileri ortaya çıkarmayı hedefler.

## Mimari

```
                 ┌──────────────┐
  video/görüntü ─►  Vision      ├──┐
                 └──────────────┘  │
                 ┌──────────────┐  │
  ses kaydı    ─►  Audio        ├──┤   ┌──────────┐   ┌───────────────┐
                 └──────────────┘  ├──►│  Fusion  ├──►│ Karar Desteği │
                 ┌──────────────┐  │   └──────────┘   │  (olasılık +  │
  konuşma metni─►  Text (NLP)   ├──┤                  │  açıklama)    │
                 └──────────────┘  │                  └───────────────┘
                 ┌──────────────┐  │
  sensör verisi─►  Wearable     ├──┘
                 └──────────────┘
```

Ayrıntı: [`docs/mimari.md`](docs/mimari.md)

## Klinik Yazılım Katmanı

Model hattının üzerinde, psikolog/psikiyatristlerin günlük pratikte
kullanabileceği bir **klinik karar destek yazılımı** yer alır:

- **Kalıcı kayıt** — hasta, görüşme, doktor notu ve analiz sonuçları
  ilişkisel veritabanında (varsayılan SQLite; üretimde `PAI_DATABASE_URL`
  ile PostgreSQL).
- **Kimlik doğrulama + roller** — JWT ile giriş; *yönetici /
  psikiyatrist / psikolog* rol yetkileri.
- **Klinik panel (PWA)** — hasta listesi → profil → görüşme geçmişi →
  yeni görüşme (transkript, dosya veya **mikrofon kaydı**, belirti
  süresi, vaka şiddeti, doktor notu) → kalıcı rapor. Mobil cihazda
  "Ana ekrana ekle" ile uygulama gibi çalışır.
- **Karar destek motoru** — model çıktısını yapılandırılmış klinik
  girdilerle birleştirip açıklanabilir *tanı profili + risk düzeyi +
  güven/belirsizlik uyarıları + öneriler* üretir (`decision/engine.py`).
- **KVKK** — takma-kimliklendirme, metin maskeleme, ham medyanın
  saklanmaması, denetim günlüğü ekranı, unutulma hakkı
  (`DELETE /patients/{id}`). Ayrıntı: [`docs/kvkk-uyum-matrisi.md`](docs/kvkk-uyum-matrisi.md).

## Proje Yapısı

```
psikiyatri-ai/
├── src/psikiyatri_ai/     # Ana Python paketi
│   ├── config.py          # Merkezi yapılandırma (pydantic-settings)
│   ├── data/              # Veri alma, anonimleştirme, denetim günlüğü, veri modülü
│   ├── db/                # SQLAlchemy modelleri (kullanıcı/hasta/görüşme/analiz)
│   ├── decision/          # Kural tabanlı açıklanabilir karar destek motoru
│   ├── modalities/        # vision / audio / text / wearable alt-modelleri
│   ├── fusion/            # Multimodal füzyon + uyumsuzluk metriği
│   ├── training/          # Eğitim, değerlendirme, checkpoint yan-verisi
│   └── api/               # FastAPI servis + kimlik doğrulama + klinik panel (PWA)
├── config/                # YAML yapılandırmaları
├── data/                  # Veri + yerel SQLite veritabanı (git'e dahil değil)
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
# API servisini + klinik paneli başlat
uvicorn psikiyatri_ai.api.main:app --reload
#   → Klinik panel (PWA): http://127.0.0.1:8000/
#   → API dokümanı:       http://127.0.0.1:8000/docs

# Testleri çalıştır
pytest

# Kod kalitesi
ruff check src tests
```

İlk açılışta veritabanı oluşturulur ve `admin` hesabı tohumlanır
(geliştirme parolası: `degistir123`; üretimde varsayılanlar reddedilir).

- Kurulum + ilk klinik akış: [`docs/hizli-baslangic.md`](docs/hizli-baslangic.md)
- Klinisyen kullanım kılavuzu: [`docs/kullanici-kilavuzu.md`](docs/kullanici-kilavuzu.md)
- Ayrıntılı test/çalıştırma adımları: [`docs/test-rehberi.md`](docs/test-rehberi.md)

## Sürekli Entegrasyon (CI)

Her `push` ve `pull_request`'te GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) şunları çalıştırır:

- **Ruff** (lint) — `src`, `tests`, `scripts`
- **Pytest** — kapsam raporu ile (Python 3.11 ve 3.12 matrisi)
- **mypy** (tip denetimi) — bilgilendirme amaçlı (CI'yı kırmaz)

> GitHub'a bağladıktan sonra rozet: `![CI](https://github.com/<OWNER>/psikiyatri-ai/actions/workflows/ci.yml/badge.svg)` (`<OWNER>` kısmını kendi kullanıcı/organizasyon adınızla değiştirin).

## Etik ve Veri Gizliliği

Sağlık verisi işlendiği için proje **KVKK** ve uluslararası etik standartlara uyumlu tasarlanır. Tüm hasta verisi işleme öncesi **anonimleştirilir**; erişimler rol tabanlı ve denetim günlüklüdür. Ayrıntı: [`docs/veri-yonetimi-ve-etik.md`](docs/veri-yonetimi-ve-etik.md) ve [`docs/kvkk-uyum-matrisi.md`](docs/kvkk-uyum-matrisi.md).

## Yol Haritası

18 aylık iş planı: [`docs/yol-haritasi.md`](docs/yol-haritasi.md)

## Lisans

Özel/Tescilli (Proprietary) — Eurotech Bilişim A.Ş. Tüm hakları saklıdır.
