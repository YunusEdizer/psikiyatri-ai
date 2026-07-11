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

**🖨 Raporu yazdır** ile rapor yazdırılabilir/PDF alınabilir.

## 5. Görüşme Geçmişi ve Notlar

- Hasta profilindeki geçmiş tablosundan eski raporlara tıklayarak
  ulaşılır.
- Rapora sonradan not eklemek için API `POST /interviews/{id}/notes`
  kullanılabilir (panel sürümünde not alanı görüşme formundadır).

## 6. Yönetici İşlevleri

**Yönetim** sekmesi (yalnızca yönetici):

- **Kullanıcı oluşturma** — rol seçerek yeni klinisyen hesabı açın.
- **Denetim günlüğü** — kim, ne zaman, hangi hastaya erişti; KVKK
  hesap verebilirlik kaydı.
- **Hasta silme (KVKK)** — hasta profilindeki "Hastayı sil" düğmesi
  hastayı ve tüm bağlı kayıtları **geri alınamaz** biçimde siler
  (unutulma hakkı).

## 7. Sorun Bildirimi

Pilot sürecinde hata/öneri bildirimleri için proje ekibine e-posta ile
ulaşın veya kurumun belirlediği destek kanalını kullanın. Bildirimde
ekran görüntüsü ve (varsa) rapor **seans numarasını** paylaşın — hasta
kimliği paylaşmayın.
