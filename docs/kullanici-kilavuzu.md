# Kullanıcı Kılavuzu — Klinik Karar Destek Paneli

Bu kılavuz, **psikolog ve psikiyatristlerin** paneli günlük pratikte
kullanması içindir. Teknik kurulum için `docs/hizli-baslangic.md`'ye
bakın.

> ⚠️ **Önemli:** Sistem bir **karar destek** aracıdır. Ürettiği hiçbir
> çıktı kesin tanı değildir; nihai tanı ve tedavi kararı yetkili sağlık
> uzmanına aittir.

## 1. Giriş

Tarayıcıda `http://<sunucu>:8000/` adresini açın (mobil cihazda
"Ana ekrana ekle" ile uygulama gibi kurulabilir — PWA).

- **Kullanıcı adı / parola** ile giriş yapın. Hesabınız yoksa sistem
  yöneticinizden isteyin.
- **Girişsiz hızlı analiz**: kayıt tutmadan tek seanslık deneme analizi
  yapmak için giriş ekranındaki bağlantıyı kullanın (hasta kaydı
  oluşturmaz, hiçbir veri saklanmaz).

Roller:

| Rol | Yetkiler |
|-----|----------|
| Psikolog / Psikiyatrist | Hasta kaydı, görüşme, analiz, not ekleme, rapor |
| Yönetici | + kullanıcı yönetimi, denetim günlüğü, hasta silme (KVKK) |

## 2. Hasta Kaydı

**Hastalar → + Yeni Hasta**

- *Kurum içi kimlik*: dosya numarası girebilirsiniz; sistem bunu geri
  döndürülemez bir **takma-kimliğe** çevirir ve **asla saklamaz**. Aynı
  numara her zaman aynı takma-kimliği üretir (takip için yeterli).
- Doğum yılı, cinsiyet, kültürel bağlam alanları isteğe bağlıdır.
- Sevk/başvuru notundaki kişisel veriler (TC, telefon, ad) otomatik
  maskelenir.

## 3. Görüşme ve Analiz

Hasta profilinde **+ Yeni Görüşme**:

1. **Transkript**: seans konuşma metnini yapıştırın (veya yazın).
2. **Ses**: dosya yükleyin ya da **🎙 Kaydet** ile doğrudan mikrofon
   kaydı alın (mobilde de çalışır).
3. **Video / giyilebilir sensör CSV**: varsa ekleyin.
4. **Belirti süresi (hafta)** ve **vaka şiddeti**: karar motoru bu
   bilgileri tanı ölçütleriyle (örn. ≥2 hafta epizod ölçütü,
   kronikleşme eşiği) ilişkilendirir.
5. **Doktor/klinisyen notu**: gözlemleriniz maskelenerek kaydedilir ve
   metin analizine dâhil edilir.
6. **Analiz Et ve Kaydet** — sonuç sağdaki rapor panelinde görünür ve
   hastanın görüşme geçmişine kalıcı eklenir.

> Ham video/ses dosyaları analizden hemen sonra **silinir**; sadece
> analiz sonuçları ve maskeli metinler saklanır.

### 3.1 Otomatik Görüşme (yönlendirmeli tarama)

Kenar çubuğundaki **◉ Otomatik Görüşme**, 24 soruluk yarı yapılandırılmış
tarama protokolünü danışana bilgisayar başında otomatik uygular. Sorular
NLP korpusundaki `Q1–Q24` anket şemasıyla bire bir hizalıdır ve DSM-5 ana
belirti kümelerini tarar (duygudurum, kaygı, OKB, travma, mani, psikotik
belirtiler, dürtü/öfke; Q6 kendine-zarar risk taramasıdır).

Akış:

1. **KVKK onamı** (zorunlu): kamera+mikrofon kaydı ve işleme amacı
   danışana açıklanır, onay kutusu işaretlenir.
2. **Görüşme başlar**: kamera canlı önizlemede, video+ses kayıtta.
   Sistem her soruyu ekranda gösterir ve **sesli okur** (kapatılabilir).
3. **Danışan sesli yanıtlar**: konuşma tanıma yanıtı otomatik metne döker;
   metin elle düzeltilebilir. Sistem soruyu okurken tanıma duraklatılır
   (kendi sesini not almaz). Soru **Yanıtsız geç** ile atlanabilir.
4. **Bitişte** yanıtlar sunucuda `Görüşmeci:/Danışan:` transkriptine
   çevrilir; video + ses + metin birlikte füzyon analizinden geçer ve
   karar destek raporu sağ panelde görünür.

Notlar:

- Kamera/mikrofon erişimi yoksa görüşme **yalnızca yazılı yanıtlarla**
  sürer (özellik kaybıyla, kesintisiz).
- Konuşma tanıma tarayıcı hizmetini kullanır; bazı tarayıcılarda ses bu
  işlem için tarayıcı sağlayıcısının sunucusuna iletilir. Hassas
  ortamlarda bu seçeneği kapatıp yanıtları elle yazın.
- Kayıtlar analiz sonrası **silinir**; bu akışta kalıcı kayıt tutulmaz
  (kalıcı kayıt için hasta profilinden *Yeni Görüşme* kullanın).

## 4. Raporu Okuma

| Bölüm | Anlamı |
|-------|--------|
| **Risk bandı** | düşük / orta / yüksek — model + klinik girdilerin birleşimi. "Yüksek-risk ifadesi" uyarısı, konuşmada kendine zarar içeriği saptandığını gösterir. |
| **Olasılık dağılımı** | Modelin sınıf olasılıkları (kontrol + tanılar). |
| **Karar Destek Profili** | Tanı profili + her tanıya destek veren bulgular, güven düzeyi, risk etkenleri, klinik bulgular ve öneriler. |
| **Güven düzeyi uyarıları** | "Olasılıklar birbirine yakın", "tek veri kaynağı", "eğitilmemiş model" gibi düşük-güven durumlarında sonuç ihtiyatla yorumlanmalıdır. |
| **Modalite katkısı** | Modelin karara hangi kaynaktan (ses/yüz/metin/sensör) ne oranda beslendiği. |
| **Uyumsuzluk skoru** | Yüksek değer, söylenen içerik ile ses tonu/yüz ifadesi arasında çelişkiye işaret eder (bastırılmış duygulanım olasılığı) — görüşmede derinleştirin. |
| **Saptanan belirti ifadeleri** | Transkriptte eşleşen ifadeler — şeffaflık için gösterilir. |
| **"Sonuç belirsiz"** | Model güveni düşükse **veya** model güveni orta düzeydeyken sözlük analizi hiçbir belirti bulamamışsa sistem tanı öne sürmekten çekimser kalır. Mevcut model kontrol (sağlıklı) sınıfı içermediğinden "bulgu yok" kararını üretemez; bu durumda değerlendirme tümüyle klinisyene aittir. |
| **Ses gözlemleri (betimsel)** | Kaydedilen sesten doğrudan hesaplanan yorumlanabilir ölçütler: perde (F0), tonlama (monoton/dalgalı), ses şiddeti, konuşma etkinliği ve süre. Bunlar klinisyene **gözlemsel** destektir; tanı üretmez ve tanı füzyonuna katılmaz. Sebep: ses/görüntü kodlayıcıları henüz etiketli medya ile eğitilmediğinden — ampirik olarak eğitilmemiş ses kodlayıcısının metin tanısını bozduğu görülmüştür (ör. depresyon %78 → %43) — bu modaliteler tanıya karıştırılmaz. |

**🖨 Raporu yazdır** ile rapor yazdırılabilir/PDF alınabilir.

## 5. Görüşme Geçmişi ve Notlar

- Hasta profilindeki geçmiş tablosundan eski raporlara tıklayarak
  ulaşılır.
- Rapora sonradan not eklemek için API `POST /interviews/{id}/notes`
  kullanılabilir (panel sürümünde not alanı görüşme formundadır).

## 6. Veri Seti (Araştırma Korpusu)

**Veri Seti** sekmesi, modelin eğitiminde kullanılan anonimleştirilmiş
Türkçe NLP korpusunu (10.000 kayıt, 8 tanı sınıfı) inceleme imkânı verir
(giriş gerektirir):

- **Özet kartı** — toplam kayıt, sınıf sayısı ve tanı dağılımı çubukları.
- **Kayıt listesi** — tanı filtresi ve transkript içi arama ile sayfalı
  gezinme.
- **Kayıt detayı** — seçilen kaydın takma-kimliği ve tam (maskeli)
  transkripti. Tanı adı sızıntıları `[TANI]` olarak temizlenmiştir;
  her detay erişimi denetim günlüğüne işlenir (KVKK).

Korpus görünmüyorsa manifest üretilmemiş demektir:
`python scripts/ingest_nlp_dataset.py` (bkz. `docs/veri-formati.md`).

## 7. Yönetici İşlevleri

**Yönetim** sekmesi (yalnızca yönetici):

- **Kullanıcı oluşturma** — rol seçerek yeni klinisyen hesabı açın.
- **Denetim günlüğü** — kim, ne zaman, hangi hastaya erişti; KVKK
  hesap verebilirlik kaydı.
- **Hasta silme (KVKK)** — hasta profilindeki "Hastayı sil" düğmesi
  hastayı ve tüm bağlı kayıtları **geri alınamaz** biçimde siler
  (unutulma hakkı).

## 8. Sorun Bildirimi

Pilot sürecinde hata/öneri bildirimleri için proje ekibine e-posta ile
ulaşın veya kurumun belirlediği destek kanalını kullanın. Bildirimde
ekran görüntüsü ve (varsa) rapor **seans numarasını** paylaşın — hasta
kimliği paylaşmayın.
