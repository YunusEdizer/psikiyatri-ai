/* Basit PWA service worker: uygulama kabuğunu önbelleğe alır.
   API istekleri (auth/patients/interviews/session/demo/predict/audit)
   her zaman ağa gider — klinik veri asla önbelleklenmez (KVKK). */
const CACHE = "psikiyatri-ai-v1";
const SHELL = ["/", "/icon.svg", "/manifest.webmanifest"];
const API_PREFIXES = [
  "/auth", "/patients", "/interviews", "/session",
  "/demo", "/predict", "/audit", "/health", "/docs", "/openapi.json",
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
  e.respondWith(
    caches.match(e.request).then((hit) => hit || fetch(e.request))
  );
});
