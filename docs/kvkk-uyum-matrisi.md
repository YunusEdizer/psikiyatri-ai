# KVKK / GDPR Uyum Matrisi

Bu belge, sistemin kişisel sağlık verisi işleme faaliyetlerinin
**KVKK (6698 sayılı Kanun)** ve **GDPR** ilkeleriyle eşleşmesini ve her
ilkenin depodaki **teknik karşılığını** gösterir. Pilot ve ürünleşme
aşamalarında kurum Veri Koruma Görevlisi (DPO) ile birlikte gözden
geçirilmelidir.

> Kapsam notu: Sistem *özel nitelikli kişisel veri* (sağlık verisi)
> işler; KVKK md. 6 uyarınca **açık rıza** veya kanuni istisna şarttır.
> Rıza formu şablonu ve veri toplama SOP'u pilot kurumla birlikte
> hazırlanır (bkz. `docs/veri-yonetimi-ve-etik.md`).

## İlke → Teknik Önlem Matrisi

| # | KVKK/GDPR ilkesi | Teknik / idari önlem | Depodaki karşılık |
|---|------------------|----------------------|-------------------|
| 1 | Hukuka uygunluk, açık rıza (KVKK md.5-6) | Açık rıza alanları manifest şemasında zorunlu; rızasız kayıt üretim modunda reddedilir | `data/schema.py`, `data/manifest_builder.py` |
| 2 | Veri minimizasyonu (amaçla sınırlılık) | Doğrudan tanımlayıcı (ad, TC, dosya no) **hiç saklanmaz**; hasta yalnızca takma-kimlikle temsil edilir | `data/anonymization.py` (`pseudonymize`, HMAC-SHA256 + tuz), `db/models.py` (`Patient`) |
| 3 | Takma-kimliklendirme (GDPR md.25 — privacy by design) | Kurum içi kimlik sunucuda tek yönlü özete çevrilir ve atılır; boylamsal eşleştirme kimliksiz yapılır | `routes_patients.create_patient` |
| 4 | Serbest metinde kişisel veri | TC, telefon, e-posta, unvanlı/hitaplı adlar otomatik maskelenir; transkript ve doktor notu **maskeli** yazılır | `mask_pii()`; `Interview.transcript_masked`, `DoctorNote.content_masked` |
| 5 | Ham medyanın saklanmaması | Video/ses/sensör kayıtları geçici dosyada işlenir, yanıt sonrası **silinir**; kalıcı depoya yazılmaz | `api/uploads.py` (`cleanup`) |
| 6 | Erişim kontrolü (KVKK md.12 — veri güvenliği) | JWT ile kimlik doğrulama; rol tabanlı yetki (yönetici / psikiyatrist / psikolog); hasta uç noktaları kimliksiz erişime kapalı | `api/security.py`, `routes_*` |
| 7 | Hesap verebilirlik / izlenebilirlik | Tüm erişim ve işleme olayları (giriş, hasta görüntüleme, analiz, silme) kullanıcı kimliğiyle denetim günlüğüne yazılır | `data/audit.py`; panelde "Denetim Günlüğü" ekranı (yalnızca yönetici) |
| 8 | Unutulma hakkı (KVKK md.7 / GDPR md.17) | `DELETE /patients/{id}` hastayı ve **tüm bağlı kayıtları** (görüşme, not, analiz) kalıcı siler; işlem denetim günlüğüne işlenir; yalnızca yönetici yetkilidir | `routes_patients.delete_patient` (kaskatlı silme) |
| 9 | Güvenli sır yönetimi | Anonimleştirme tuzu, JWT anahtarı ve yönetici parolası ortam değişkeniyle verilir; üretim modunda varsayılan değerler **reddedilir** (uygulama açılmaz) | `config.py` (`_enforce_production_secret`) |
| 10 | Parola güvenliği | PBKDF2-SHA256 (200.000 iterasyon, kayıt başına rastgele tuz), sabit-zamanlı karşılaştırma | `api/security.py` |
| 11 | Oturum güvenliği | JWT süreli (varsayılan 8 saat); süresi dolan belirteç reddedilir | `api/security.py` |
| 12 | Şeffaflık / açıklanabilirlik | Model çıktısı olasılık + modalite katkısı + uyumsuzluk skoru + kural tabanlı gerekçelerle sunulur; "kesin tanı değildir" uyarısı her raporda yer alır | `decision/engine.py`, rapor ekranı |
| 13 | İstemci tarafında veri sızıntısı önleme | PWA service worker **API yanıtlarını asla önbelleklemez**; oturum belirteci kalıcı değil oturumluk depoda tutulur | `static/sw.js`, `static/index.html` |

## Saklama Süreleri ve Silme Politikası

| Veri | Saklama | Silme yolu |
|------|---------|-----------|
| Hasta kaydı (takma-kimlik + klinik bağlam) | Tedavi ilişkisi + kurum politikası süresince | `DELETE /patients/{id}` (kaskatlı) |
| Görüşme transkripti (maskeli) | Hasta kaydına bağlı | Hasta silinince otomatik |
| Analiz sonuçları / karar profilleri | Hasta kaydına bağlı | Hasta silinince otomatik |
| Ham medya (video/ses/sensör) | **Saklanmaz** — yalnızca istek süresince geçici | Otomatik (istek sonu) |
| Denetim günlüğü | Mevzuat gereği min. saklama (öneri: 2 yıl) | Yıllık döngüsel arşiv/imha (kurum SOP'u) |
| Model eğitim verisi | Ayrı süreç — bkz. veri yönetimi belgesi | `docs/veri-yonetimi-ve-etik.md` |

Not: Denetim günlüğünde kişisel veri bulunmaz (yalnızca takma-kimlik ve
olay meta verisi); bu nedenle hasta silindiğinde günlük kayıtları
hesap verebilirlik amacıyla korunur.

## Üretim Dağıtımı Zorunlu Kontrol Listesi

- [ ] `PAI_ENV=production`
- [ ] `PAI_ANON_SALT` — güçlü rastgele değer, sır yöneticisinde
- [ ] `PAI_SECRET_KEY` — güçlü rastgele değer, sır yöneticisinde
- [ ] `PAI_ADMIN_PASSWORD` — ilk giriş sonrası değiştirilecek güçlü parola
- [ ] `PAI_DATABASE_URL` — TLS'li PostgreSQL bağlantısı
- [ ] HTTPS zorunlu (ters proxy / TLS sonlandırma)
- [ ] Veritabanı diskte şifreli (at-rest encryption)
- [ ] Yedekleme + felaket kurtarma planı (yedekler de şifreli)
- [ ] Kurum DPO onayı + işleme envanterine (VERBİS) kayıt
- [ ] Açık rıza formu ve aydınlatma metni kurum hukuk birimince onaylı
