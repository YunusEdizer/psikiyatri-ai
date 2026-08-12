/* Basit PWA service worker: uygulama kabuğunu önbelleğe alır.
   API istekleri (auth/patients/interviews/session/demo/predict/audit/
   dataset) her zaman ağa gider — klinik veri asla önbelleklenmez (KVKK).
   Uygulama kabuğu ("/") network-first sunulur: arayüz güncellemeleri
   anında ulaşır, ağ yokken önbellekten (çevrimdışı) devam edilir. */
const CACHE = "psikiyatri-ai-v3";
const SHELL = ["/", "/icon.svg", "/manifest.webmanifest", "/eurotech_logo.png"];
const API_PREFIXES = [
  "/auth", "/patients", "/interview", "/session",
  "/demo", "/predict", "/audit", "/dataset",
  "/health", "/docs", "/openapi.json",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET") return;
  if (API_PREFIXES.some((p) => url.pathname.startsWith(p))) return;

  // Uygulama kabuğu: önce ağ (taze arayüz), düşerse önbellek (çevrimdışı).
  if (url.pathname === "/" || e.request.mode === "navigate") {
    e.respondWith(
      fetch(e.request)
        .then((res) => {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(e.request, copy));
          return res;
        })
        .catch(() => caches.match(e.request))
    );
    return;
  }

  // Statik varlıklar: önbellek öncelikli.
  e.respondWith(
    caches.match(e.request).then((hit) => hit || fetch(e.request))
  );
});
