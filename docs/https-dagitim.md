# HTTPS Dağıtımı (kamera/mikrofon için)

Tarayıcının kamera ve mikrofon erişimi (`getUserMedia`) yalnızca
**`localhost`** veya **HTTPS** üzerinde çalışır. Otomatik görüşme başka bir
bilgisayardan (danışan bilgisayarı) ağ üzerinden açılacaksa, sunucu HTTPS
sunmalıdır; aksi hâlde **kamera hiç başlamaz** ve görüşme yalnızca yazılı
yanıtlarla sürer.

## 1. Kendinden-imzalı sertifika (iç ağ / pilot)

```bash
# Sunucunun ağdaki IP'siyle sertifika üret
python scripts/generate_cert.py --host 192.168.1.50

# HTTPS ile çalıştır
uvicorn psikiyatri_ai.api.main:app --host 0.0.0.0 --port 8443 \
    --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem
```

Danışan/klinik bilgisayarında `https://192.168.1.50:8443` açılır. Kendinden-
imzalı sertifika olduğundan tarayıcı bir kez güvenlik uyarısı gösterir;
kontrollü iç ağda "yine de devam et" ile geçilir. Daha temizi: `certs/cert.pem`
dosyasını cihazlara **güvenilir kök** olarak elle eklemek (uyarı kaybolur).

> `certs/` dizini git'e girmez (özel anahtar). Sertifikalar her ortamda
> yeniden üretilmelidir.

## 2. Ters vekil (reverse proxy) — önerilen üretim yolu

Gerçek dağıtımda uygulamayı düz HTTP'de tutup önüne bir ters vekil koymak
daha sağlıklıdır. Örnek **Caddy** (otomatik Let's Encrypt sertifikası):

```
psikiyatri.ornek-klinik.tr {
    reverse_proxy 127.0.0.1:8020
}
```

Caddy sertifikayı otomatik alır ve yeniler; uygulama `127.0.0.1:8020`'de
sade çalışır. nginx + certbot da eşdeğer bir seçenektir.

## Notlar

- Kendinden-imzalı sertifika **yalnızca** kontrollü iç ağ/pilot içindir;
  halka açık kullanımda gerçek bir CA sertifikası şarttır.
- HTTPS ayrıca konuşma tanıma (STT) ve service worker (PWA) için de
  gereklidir; birçok tarayıcı özelliği güvenli bağlam (secure context)
  ister.
- Docker dağıtımında SSL argümanları `docker/docker-compose.yml` komutuna
  eklenebilir ya da bir Caddy/nginx servisi tanımlanabilir.
