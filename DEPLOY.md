# Yayına Alma Rehberi (Docker + Caddy)

Bu rehber, uygulamayı bir sunucuda (VPS) `psikiyatri.siteniz.com` gibi bir
subdomain'de, otomatik HTTPS ile yayına almak içindir. Tüm ağır iş Docker'a
paketlenmiştir; hedef: **klonla → .env doldur → modeli kopyala → tek komut**.

> Sunucu önerisi: Ubuntu 22.04+, 2-4 vCPU, 4-8 GB RAM, 25 GB disk
> (Hetzner CX22/CX32, DigitalOcean vb. ~5-15 €/ay). GPU gerekmez.

---

## 0. Ön koşullar

- Sunucuda Docker + Docker Compose eklentisi kurulu olmalı:
  ```bash
  curl -fsSL https://get.docker.com | sh
  ```
- Alan adı panelinde **DNS A kaydı**: `psikiyatri.siteniz.com → SUNUCU_IP`
  (Caddy, sertifikayı bu kayıt üzerinden otomatik alır.)
- Güvenlik duvarında yalnızca **22, 80, 443** portları açık olsun.

## 1. Depoyu klonla

```bash
git clone https://github.com/YunusEdizer/psikiyatri-ai.git
cd psikiyatri-ai
```

## 2. Model dosyalarını kopyala (git'te DEĞİLDİR)

Eğitilmiş model boyut nedeniyle depoda tutulmaz. Geliştirici makinesinden
sunucuya kopyalanmalıdır (yalnızca 2 dosya zorunlu):

```bash
# Geliştirici makinesinden (örnek):
scp models/psikiyatri_model.pt models/classes.json kullanici@SUNUCU_IP:~/psikiyatri-ai/models/
```

| Dosya | Zorunlu mu | Not |
|---|---|---|
| `models/psikiyatri_model.pt` (~426 MB) | Evet | Eğitilmiş 9-sınıf model |
| `models/classes.json` | Evet | Sınıf sırası + kalibrasyon (T) + model kartı |
| `models/face_landmarker.task` | Hayır | Yoksa konteyner ilk açılışta otomatik indirir |
| BERTurk / Whisper | Hayır | İlk kullanımda otomatik iner, volume'da kalıcıdır |

> Model kopyalanmazsa servis yine açılır ama "eğitilmemiş model" uyarısıyla
> anlamsız skor üretir — üretimde mutlaka kopyalayın.

## 3. Üretim ayarlarını doldur

```bash
cp docker/.env.example docker/.env
nano docker/.env
```

Doldurulacaklar: `PAI_ADMIN_PASSWORD`, `PAI_SECRET_KEY`, `PAI_ANON_SALT`,
`DOMAIN`. Sır üretmek için:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

> Güvence: `PAI_ENV=production` iken sırlar varsayılan bırakılırsa uygulama
> **başlamayı reddeder** (bilinçli koruma). `PAI_ANON_SALT` bir kez üretilir
> ve **asla değiştirilmez** (değişirse takma-kimlik eşleşmeleri bozulur).

## 4. Yayına al

```bash
cd docker
docker compose up -d --build
```

İlk build ~5-10 dk sürer (CPU torch + bağımlılıklar). Ardından:

```bash
docker compose logs -f app       # "Uvicorn running" + "ffmpeg hazır" görün
curl -s https://psikiyatri.siteniz.com/health
```

`/health` çıktısında `"status":"ok"` ve `model_card.git_commit` görünmelidir.

## 5. İlk doğrulama (tarayıcıdan)

1. `https://psikiyatri.siteniz.com` → giriş ekranı (Eurotech logolu).
2. `admin` + `.env`'deki parola ile gir → **Yönetim** görünüyor olmalı.
3. Bir doktor hesabı "Kayıt olun" ile açılıp anında giriş yapabilmeli.
4. Hasta/Danışan oluştur → **Otomatik Görüşme** → kamera izni → kısa görüşme
   → rapor hasta geçmişine düşmeli. (HTTPS olduğundan kamera/mikrofon çalışır.)

## 6. İşletme

```bash
# Güncelleme (yeni sürüm yayınlandığında)
git pull && cd docker && docker compose up -d --build

# Durum / loglar
docker compose ps
docker compose logs -f app

# Durdurma
docker compose down          # volume'lar (DB, sertifika, HF cache) KORUNUR
```

### Yedekleme (önerilir — cron ile günlük)

Kalıcı veriler host'ta `data/` klasöründedir (SQLite DB + denetim günlüğü):

```bash
tar czf /root/yedek/psikiyatri-data-$(date +%F).tar.gz -C ~/psikiyatri-ai data
```

`models/` bir kez yedeklenmesi yeterlidir (değişmez).

## Sorun giderme

| Belirti | Neden / Çözüm |
|---|---|
| Uygulama açılmıyor, logda "güvensiz varsayılan" | `docker/.env` sırları doldurulmamış |
| Sertifika alınamıyor | DNS A kaydı henüz yayılmadı veya 80/443 kapalı |
| Kamera açılmıyor | Sayfa HTTP ile açılmış — mutlaka `https://` kullanın |
| İlk analiz çok yavaş | Normal: ilk istekte BERTurk yüklenir (~10-30 sn, CPU) |
| "eğitilmemiş model" uyarısı | `models/psikiyatri_model.pt` kopyalanmamış (Adım 2) |
| İlk sesli yanıt yavaş | Normal: Whisper modeli ilk kullanımda iner (volume'da kalır) |

## Kapsam hatırlatması

Bu dağıtım **kontrollü pilot** içindir: sistem klinik karar desteği prototipidir
(TRL6), kesin teşhis koymaz; gerçek hasta verisi toplanacaksa aydınlatma
metni/açık rıza ve etik kurul onayı yayın sahibinin sorumluluğundadır
(bkz. `docs/https-dagitim.md`, `docs/veri-yonetimi-ve-etik.md`).
