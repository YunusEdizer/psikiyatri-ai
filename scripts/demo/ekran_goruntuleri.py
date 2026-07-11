"""Panelin tüm ekranlarının görüntülerini ve gezinti videosunu alır."""

import shutil
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8010"
OUT = Path("docs/ekran-goruntuleri")
OUT.mkdir(parents=True, exist_ok=True)
VIDEO_DIR = Path("docs/ekran-goruntuleri/_video_tmp")

DR = ("dr.aylin", "GucluParola-2026")
ADMIN = ("admin", "degistir123")


def snap(page, name, full=False, pause=0.9):
    time.sleep(pause)  # çubuk animasyonları / ağ istekleri otursun
    page.screenshot(path=str(OUT / name), full_page=full)
    print("✓", name)


def login(page, creds):
    page.goto(BASE + "/")
    page.wait_for_selector("#v-login:not(.hidden)", timeout=10000)
    page.fill("#l-user", creds[0])
    page.fill("#l-pass", creds[1])
    page.click("#l-go")
    page.wait_for_selector("#app:not(.hidden)", timeout=10000)


with sync_playwright() as pw:
    browser = pw.chromium.launch()

    # ---------- Masaüstü + video kaydı ----------
    ctx = browser.new_context(
        viewport={"width": 1440, "height": 900},
        locale="tr-TR",
        record_video_dir=str(VIDEO_DIR),
        record_video_size={"width": 1440, "height": 900},
    )
    page = ctx.new_page()

    # 05 — giriş ekranı
    page.goto(BASE + "/")
    page.wait_for_selector("#l-go")
    snap(page, "05-giris.png", pause=0.5)

    # 06 — hasta listesi (risk rozetleriyle)
    login(page, DR)
    page.wait_for_selector("#plist tr.rowlink")
    snap(page, "06-hasta-listesi.png")

    # 07 — yeni hasta formu
    page.click("#btn-newpat")
    page.fill("#np-ident", "dosya-4021")
    page.fill("#np-birth", "1994")
    page.select_option("#np-gender", "kadın")
    page.fill("#np-ctx", "TR / kentsel")
    page.fill("#np-note", "Hasta Zeynep Hanım aile hekimi tarafından yönlendirildi.")
    snap(page, "07-yeni-hasta.png")
    page.click("#np-cancel")

    # 08 — hasta profili (yüksek riskli hasta en üstte)
    page.click("#plist tr.rowlink >> nth=0")
    page.wait_for_selector("#v-patient:not(.hidden)")
    page.wait_for_selector("#ilist tr")
    snap(page, "08-hasta-profili.png")

    # 09 — yeni görüşme formu (doldurulmuş)
    page.click("#btn-newint")
    page.fill("#if-txt",
              "Bu hafta uykum biraz düzeldi ama hâlâ kendimi çok karamsar hissediyorum, "
              "hiçbir şey yapmak istemiyorum.")
    page.fill("#if-dur", "31")
    page.select_option("#if-sev", "şiddetli")
    page.fill("#if-note", "İzlem seansı; duygudurumda kısmi düzelme, riskli ifadeler azalmış.")
    snap(page, "09-yeni-gorusme-formu.png")

    # 10 — analiz sonucu: karar destek profili
    page.click("#if-go")
    page.wait_for_selector("#report:not(.hidden)", timeout=60000)
    page.wait_for_selector("text=Karar Destek Profili", timeout=15000)
    snap(page, "10-rapor-karar-destek.png", full=True, pause=4.5)  # bildirim kaybolsun

    # 11 — giyilebilir sensörlü seansın kalıcı raporu (derin öğrenme füzyonu)
    page.click("#back-patients")
    page.wait_for_selector("#plist tr.rowlink")
    # Hasta 1 (dosya-1042) — listenin en altındaki kayıt
    page.click("#plist tr.rowlink >> nth=-1")
    page.wait_for_selector("#v-patient:not(.hidden)")
    page.wait_for_selector("#ilist tr.rowlink")
    page.click("#ilist tr.rowlink >> nth=0")  # en yeni: sensörlü seans
    page.wait_for_selector("#report:not(.hidden)", timeout=30000)
    snap(page, "11-rapor-giyilebilir-fuzyon.png", full=True, pause=4.5)  # bildirim kaybolsun

    # 12 — hızlı analiz: sentetik demo (eğitilmiş 5 sınıflı model)
    page.click("#mainnav a[data-view='quick']")
    page.wait_for_selector("#v-quick:not(.hidden)")
    page.click(".tabs button[data-tab='demo']")
    page.select_option("#scn", "depresyon")
    page.click("#q-go")
    page.wait_for_selector("#q-report:not(.hidden)", timeout=30000)
    snap(page, "12-hizli-demo-model.png", full=True, pause=1.4)

    # 13 — yönetim: kullanıcılar + KVKK denetim günlüğü (yönetici)
    page.click("#logout")
    page.wait_for_selector("#v-login:not(.hidden)")
    login(page, ADMIN)
    page.click("#mainnav a[data-view='admin']")
    page.wait_for_selector("#ulist tr")
    page.wait_for_selector("#alist tr")
    snap(page, "13-yonetim-denetim-gunlugu.png", full=True)

    # 14 — API dokümanı (Swagger)
    page.goto(BASE + "/docs")
    page.wait_for_selector(".swagger-ui .opblock", timeout=20000)
    snap(page, "14-api-docs.png", full=True, pause=1.2)

    page.close()
    ctx.close()

    # videoyu adlandır
    videos = sorted(VIDEO_DIR.glob("*.webm"), key=lambda p: p.stat().st_size, reverse=True)
    if videos:
        shutil.move(str(videos[0]), str(OUT / "ekran-kaydi-panel.webm"))
        print("✓ ekran-kaydi-panel.webm")
    shutil.rmtree(VIDEO_DIR, ignore_errors=True)

    # ---------- Mobil (PWA) ----------
    mctx = browser.new_context(
        viewport={"width": 390, "height": 844},
        device_scale_factor=2,
        is_mobile=True,
        has_touch=True,
        locale="tr-TR",
    )
    mpage = mctx.new_page()
    login(mpage, DR)
    mpage.wait_for_selector("#plist tr.rowlink")
    snap(mpage, "15-mobil-hasta-listesi.png")
    mpage.click("#plist tr.rowlink >> nth=0")
    mpage.wait_for_selector("#v-patient:not(.hidden)")
    mpage.wait_for_selector("#ilist tr.rowlink")
    mpage.click("#ilist tr.rowlink >> nth=0")
    mpage.wait_for_selector("#report:not(.hidden)", timeout=30000)
    snap(mpage, "16-mobil-rapor.png", full=True, pause=1.4)
    mctx.close()

    browser.close()

print("Bitti →", OUT.resolve())
