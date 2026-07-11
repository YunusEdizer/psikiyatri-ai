"""Ekran görüntüleri için gerçekçi demo verisi yükler (canlı API üzerinden)."""

import csv
import io
import sys
import time

import httpx

BASE = "http://127.0.0.1:8010"

# Sunucunun açılmasını bekle
for _ in range(60):
    try:
        if httpx.get(BASE + "/health", timeout=2).status_code == 200:
            break
    except Exception:
        time.sleep(1)
else:
    sys.exit("Sunucu açılmadı")


def login(username, password):
    r = httpx.post(BASE + "/auth/login", data={"username": username, "password": password})
    r.raise_for_status()
    return {"Authorization": "Bearer " + r.json()["access_token"]}


admin = login("admin", "degistir123")

# Klinisyen hesabı
r = httpx.post(
    BASE + "/auth/users",
    json={
        "username": "dr.aylin",
        "password": "GucluParola-2026",
        "full_name": "Dr. Aylin Kaya",
        "role": "psikiyatrist",
    },
    headers=admin,
)
assert r.status_code in (201, 409), r.text
httpx.post(
    BASE + "/auth/users",
    json={
        "username": "psk.mert",
        "password": "GucluParola-2026",
        "full_name": "Psk. Mert Aydın",
        "role": "psikolog",
    },
    headers=admin,
)
dr = login("dr.aylin", "GucluParola-2026")


def new_patient(**kw):
    r = httpx.post(BASE + "/patients", json=kw, headers=dr)
    r.raise_for_status()
    return r.json()


def new_interview(pid, data, files=None):
    r = httpx.post(
        f"{BASE}/patients/{pid}/interviews", data=data, files=files, headers=dr, timeout=120
    )
    r.raise_for_status()
    return r.json()


# --- Hasta 1: depresyon profili + giyilebilir sensörlü ikinci seans ---
p1 = new_patient(
    identifier="dosya-1042",
    birth_year=1991,
    gender="kadın",
    cultural_context="TR / kentsel",
    referral_note=(
        "Hasta Elif Hanım, 12345678901 TC kimlik no ile Dr. Mehmet tarafından sevk edildi."
    ),
)
new_interview(
    p1["id"],
    {
        "transcript": (
            "Son iki aydır kendimi çok mutsuz ve değersiz hissediyorum. Eskiden keyif "
            "aldığım şeylerden zevk almıyorum, resim yapmayı bile bıraktım. Geceleri "
            "uyuyamıyorum, sabahları bitkin uyanıyorum. İş yerinde odaklanamıyorum, "
            "her şey anlamsız geliyor."
        ),
        "symptom_duration_weeks": "8",
        "severity": "orta",
        "doctor_note": (
            "Hasta görüşme boyunca göz teması kurmadı, psikomotor yavaşlama belirgin. "
            "Duygulanım kısıtlı, çökkün."
        ),
    },
)

# Giyilebilir sensör CSV'si (kalp/EEG/hareket) — gerçek füzyon modeline girer
sys.path.insert(0, "src")
from psikiyatri_ai.modalities.wearable.features import WEARABLE_CHANNELS  # noqa: E402

buf = io.StringIO()
w = csv.writer(buf)
w.writerow(WEARABLE_CHANNELS)
for i in range(120):
    w.writerow(
        [
            88 + (i % 7),          # nabız yüksek-normal
            0.35 + 0.02 * (i % 5), # HRV düşük
            4.2 + 0.1 * (i % 4),   # cilt iletkenliği
            0.6, 0.25, 0.4, 0.3, 0.45, 0.5,
        ][: len(WEARABLE_CHANNELS)]
    )
new_interview(
    p1["id"],
    {
        "transcript": "Bu hafta biraz daha iyiyim ama hâlâ çok yorgunum, uykum düzensiz.",
        "symptom_duration_weeks": "10",
        "severity": "orta",
        "doctor_note": "Kontrol seansı; giyilebilir sensör verisi eşliğinde değerlendirildi.",
    },
    files={"wearable": ("sensor-kaydi.csv", buf.getvalue().encode("utf-8"), "text/csv")},
)

# --- Hasta 2: anksiyete profili ---
p2 = new_patient(identifier="dosya-2087", birth_year=1985, gender="erkek",
                 cultural_context="TR / kırsal")
new_interview(
    p2["id"],
    {
        "transcript": (
            "Sürekli bir endişe hâlindeyim, sanki kötü bir şey olacakmış gibi. Kalabalık "
            "yerlerde panik oluyorum, kalbim hızla çarpıyor, nefes alamıyorum. Geceleri "
            "tedirgin uyanıyorum, kendimi hep gergin ve huzursuz hissediyorum."
        ),
        "symptom_duration_weeks": "3",
        "severity": "hafif",
        "doctor_note": "Yaygın anksiyete belirtileri; somatik yakınmalar ön planda.",
    },
)

# --- Hasta 3: yüksek risk ---
p3 = new_patient(identifier="dosya-3155", birth_year=1998, gender="kadın")
new_interview(
    p3["id"],
    {
        "transcript": (
            "Artık hiçbir şeyin düzeleceğine inanmıyorum, kendimi tamamen çaresiz ve "
            "umutsuz hissediyorum. Bazen yaşamak istemiyorum, her şey bitsin istiyorum. "
            "Kimseyle görüşmüyorum, günlerdir evden çıkmadım."
        ),
        "symptom_duration_weeks": "30",
        "severity": "şiddetli",
        "doctor_note": (
            "Hasta görüşmede intihar düşüncelerini doğruladı; güvenlik planı oluşturuldu. "
            "Yakını Ayşe Hanım bilgilendirildi, tel 0532 123 45 67."
        ),
    },
)

print("Demo verisi yüklendi:")
for p in (p1, p2, p3):
    print(" -", p["pseudonym"])
