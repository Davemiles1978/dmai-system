/* Alex Riviera PWA service worker
 * Strategy:
 *  - Precache shell (HTML pages + icons + manifest)
 *  - Network-first for /api/* (always fresh)
 *  - Cache-first for /static/* (icons, css)
 *  - Stale-while-revalidate for HTML pages
 */
const VERSION = 'ar-pwa-v3';
const STATIC_CACHE = `static-${VERSION}`;
const RUNTIME_CACHE = `runtime-${VERSION}`;

const PRECACHE_URLS = [
  '/static/manifest.json',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(STATIC_CACHE).then((cache) => cache.addAll(PRECACHE_URLS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => ![STATIC_CACHE, RUNTIME_CACHE].includes(k))
        .map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;

  const url = new URL(req.url);

  // Never cache the service worker itself or auth endpoints
  if (url.pathname === '/service-worker.js' || url.pathname.startsWith('/api/auth')) {
    return;
  }

  // API: network-first, fallback to cache
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(
      fetch(req).then((resp) => {
        const copy = resp.clone();
        caches.open(RUNTIME_CACHE).then((c) => c.put(req, copy)).catch(() => {});
        return resp;
      }).catch(() => caches.match(req))
    );
    return;
  }

  // Static assets: cache-first
  if (url.pathname.startsWith('/static/')) {
    event.respondWith(
      caches.match(req).then((cached) => cached || fetch(req).then((resp) => {
        const copy = resp.clone();
        caches.open(STATIC_CACHE).then((c) => c.put(req, copy)).catch(() => {});
        return resp;
      }))
    );
    return;
  }

  // HTML / navigation: stale-while-revalidate
  if (req.mode === 'navigate' || (req.headers.get('accept') || '').includes('text/html')) {
    event.respondWith(
      caches.match(req).then((cached) => {
        const fetchPromise = fetch(req).then((resp) => {
          const copy = resp.clone();
          caches.open(RUNTIME_CACHE).then((c) => c.put(req, copy)).catch(() => {});
          return resp;
        }).catch(() => cached);
        return cached || fetchPromise;
      })
    );
  }
});

// Message channel: allow page to force-update SW
self.addEventListener('message', (event) => {
  if (event.data === 'SKIP_WAITING') self.skipWaiting();
});
