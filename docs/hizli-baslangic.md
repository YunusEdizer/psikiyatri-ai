# Hızlı Başlangıç (Kurulum + İlk Çalıştırma)

Pilot kurulum veya geliştirme ortamı için adım adım rehber.
Klinik kullanım rehberi: `docs/kullanici-kilavuzu.md`.

## 1. Gereksinimler

- Python 3.11+ (3.12 önerilir)
- (Üretim) PostgreSQL 14+ — geliştirmede gerekmez, SQLite otomatik

## 2. Kurulum

```bash
git clone <repo-url> && cd psikiyatri-ai
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -e ".[dev]"
```

İsteğe bağlı modalite ekstraları (derin öğrenme ile ses/görüntü skorlaması):

```bash
pip install -e ".[audio]"    # librosa, torchaudio…
pip install -e ".[vision]"   # opencv, mediapipe…
```

## 3. Yapılandırma

```bash
copy .env.example .env        # Linux/macOS: cp
```

`.env` içinde geliştirme için varsayılanlar yeterlidir. Üretimde
**zorunlu** değişkenler (aksi hâlde uygulama açılmaz):

```env
PAI_ENV=production
PAI_ANON_SALT=<güçlü-rastgele-değer>
PAI_SECRET_KEY=<güçlü-rastgele-değer>
PAI_ADMIN_PASSWORD=<güçlü-parola>
PAI_DATABASE_URL=postgresql+psycopg://kullanici:parola@sunucu:5432/psikiyatri
```

## 4. Çalıştırma

```bash
uvicorn psikiyatri_ai.api.main:app --reload
```

- Panel (PWA): <http://127.0.0.1:8000/>
- API dokümanı: <http://127.0.0.1:8000/docs>

İlk açılışta veritabanı tabloları otomatik oluşturulur ve **ilk
yönetici hesabı** tohumlanır:

| Alan | Varsayılan (geliştirme) |
|------|--------------------------|
| Kullanıcı adı | `admin` (`PAI_ADMIN_USERNAME`) |
| Parola | `degistir123` (`PAI_ADMIN_PASSWORD`) |

> Üretimde varsayılan parola reddedilir; ilk girişten sonra da parolayı
> değiştirin.

## 5. İlk Klinik Akış (2 dakika)

1. `admin` ile giriş yapın.
2. **Yönetim → Yeni Kullanıcı** ile kendinize psikiyatrist/psikolog
   hesabı açın; o hesapla yeniden giriş yapın.
3. **Hastalar → + Yeni Hasta** — dosya numarası girin (takma-kimliğe
   çevrilir).
4. Hastayı açın → **+ Yeni Görüşme** — örnek transkript:
   *"Kendimi çok umutsuz ve yorgun hissediyorum, uyuyamıyorum."*
   Belirti süresi: 6 hafta, şiddet: orta.
5. **Analiz Et ve Kaydet** — rapor ve karar destek profili sağda.

## 6. Mobil Kurulum (PWA)

Telefonun tarayıcısından panel adresini açın →
**Ana ekrana ekle / Yükle**. Uygulama tam ekran açılır; mikrofon kaydı
görüşme formundaki 🎙 düğmesiyle çalışır.

## 7. Test ve Kalite

```bash
pytest                    # tüm testler
ruff check src tests      # lint
mypy                      # tip denetimi (bilgilendirme)
```

## 8. Model Eğitimi (isteğe bağlı)

Sentetik demo verisiyle uçtan uca eğitim:

```bash
python scripts/generate_sample_data.py --root data/sample --n 60
psikiyatri-train --manifest data/sample/manifest.jsonl --epochs 5
```

Eğitilen ağırlıklar `models/` altına yazılır ve API açılışta otomatik
yükler. Ayrıntı: `docs/test-rehberi.md`, `docs/veri-formati.md`.

## 9. Sık Karşılaşılan Sorunlar

| Belirti | Çözüm |
|---------|-------|
| "Eğitilmiş ağırlık bulunamadı" uyarısı | Normaldir (demo modu). Eğitim yapın veya ağırlık dosyasını `models/` altına koyun. |
| Ses/video "alındı · tam hatta işlenecek" | İlgili ekstra kurulu değil: `pip install -e ".[audio]"` / `".[vision]"`. |
| Üretimde açılışta `ValueError` | Zorunlu sır değişkenleri eksik — bölüm 3'e bakın. |
| Mikrofon çalışmıyor | Tarayıcı izni verin; HTTPS (veya localhost) gerekir. |
