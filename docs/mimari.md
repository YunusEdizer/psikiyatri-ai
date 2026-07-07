# Mimari

## Genel Bakış

Sistem, üç modaliteyi bağımsız kodlayıp ortak bir uzayda birleştiren
**geç füzyon (late fusion)** mimarisini kullanır. Bu yaklaşım eksik
modaliteye dayanıklıdır ve her modalitenin ayrı ölçeklenmesine izin verir.

```
Ham veri            Ön işleme              Kodlayıcı            Füzyon
─────────           ─────────              ─────────           ──────
video      ─► kare örnekle + yüz    ─► VisionEncoder ─┐
                işaretçisi (MediaPipe)                 │
ses kaydı  ─► MFCC + eGeMAPS        ─► AudioEncoder  ─┼─► MultimodalFusion ─► karar
                (librosa, openSMILE)                   │      (dikkat +      + olasılık
konuşma    ─► normalize + maskele   ─► TextEncoder   ─┘      uyumsuzluk)    + açıklama
                (BERTurk tokenizer)
```

## Katmanlar

| Katman | Paket | Sorumluluk |
|--------|-------|------------|
| Veri | `data/` | Alma, **anonimleştirme**, veri sözleşmeleri |
| Modalite | `modalities/{vision,audio,text}/` | Ön işleme + kodlayıcı (embedding üretimi) |
| Füzyon | `fusion/` | Dikkat tabanlı birleştirme + uyumsuzluk skoru |
| Eğitim | `training/` | Uçtan uca model, eğitim/değerlendirme döngüleri |
| Servis | `api/` | FastAPI karar destek uç noktaları |
| Yardımcı | `utils/` | Metrikler |

## Modalite Sözleşmesi

Her modalite kodlayıcısı `nn.Module` olup `(B, ...) -> (B, embed_dim)`
üretir. Füzyon katmanı yalnızca gömme sözlüğü görür; modalitenin iç
detayından bağımsızdır. Bu, yeni modalite (ör. EEG, giyilebilir sensör)
eklemeyi kolaylaştırır — proje formunda belirtilen giyilebilir teknoloji
genişlemesi buraya yeni bir kodlayıcı olarak takılır.

## Ayırt Edici Bileşen: Uyumsuzluk (Incongruence)

`fusion/incongruence.py`, modalite gömmeleri arasındaki ikili kosinüs
mesafesini [0, 1] aralığında bir skora indirger. Bu skor:

1. Füzyon sınıflandırıcısına **ek öznitelik** olarak verilir.
2. API yanıtında uzmana **açıklanabilir sinyal** olarak sunulur.

Hipotez: ses tonu / yüz ifadesi / konuşma içeriği arasındaki yüksek
uyumsuzluk, bastırılmış duygu ve içsel çatışma göstergesi olabilir.

## Eksik Modaliteye Dayanıklılık

`MultimodalFusion.forward`, gömme sözlüğünün alt kümesini kabul eder.
Bir seansta yalnızca ses + metin varsa model yine çalışır; dikkat
ağırlıkları mevcut modaliteler üzerinden normalize edilir.

## Açıklanabilirlik

Her tahmin şunları döndürür:
- Sınıf olasılıkları
- Modalite katkı ağırlıkları (dikkat)
- Uyumsuzluk skoru

Bu, "kara kutu" olmayan, uzmanın gerekçe görebildiği bir karar desteği
sağlar — klinik kabul için kritiktir.
