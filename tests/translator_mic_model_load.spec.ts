/**
 * Translator microphone-tab speech-runtime load smoke test (Bug B).
 *
 * Confirms the mic tab's onnxruntime-web boot path resolves through the
 * importmap in public/app.html and exposes a runtime reference on
 * `window` without surfacing a module-resolve error or a 404 against
 * the pinned ORT URLs. The actual transcription pipeline (transformers.js
 * + Whisper weights) is NOT exercised — the model alone is 50–200 MB
 * and bringing it up would dwarf the load-resolution check this spec
 * is for.
 *
 * Headless Chromium does not advertise `navigator.gpu` by default, so
 * the boot path takes the WASM branch and the assertion accepts either
 * `window.ort` (WebGPU branch) or `window.ortWasm` (WASM branch). The
 * Firefox path is verified by the deployed-site E2E + manual smoke.
 *
 * Test framework note: the repo pins `@playwright/test@1.59.1` (see
 * package.json and tests/translator_upload_smoke.spec.ts header). The
 * translator-fix prompt suggested 1.49.1 but existing-convention takes
 * precedence; sharing one Playwright version across all specs avoids
 * two divergent Chromium downloads.
 *
 * Run:
 *   npx playwright install chromium     # one-time
 *   npx playwright test tests/translator_mic_model_load.spec.ts --reporter=list
 */
import { test, expect } from '@playwright/test';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..');
const PUBLIC_DIR = path.join(REPO_ROOT, 'public');

const ORT_VERSION = '1.20.1';

const MIME: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.css':  'text/css; charset=utf-8',
  '.js':   'text/javascript; charset=utf-8',
  '.mjs':  'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg':  'image/svg+xml',
  '.sigml': 'application/xml; charset=utf-8',
  '.csv':  'text/csv; charset=utf-8',
  '.ttf':  'font/ttf',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ico':  'image/x-icon',
  '.txt':  'text/plain; charset=utf-8',
};

function startStaticServer(): Promise<http.Server> {
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      try {
        const url = new URL(req.url || '/', 'http://x');
        let rel = decodeURIComponent(url.pathname);
        if (rel === '/api/plan' && req.method === 'POST') {
          let body = '';
          req.on('data', (c) => { body += c; });
          req.on('end', () => {
            res.writeHead(200, { 'content-type': 'application/json' });
            res.end(JSON.stringify({ final: '', per_token_review: [] }));
          });
          return;
        }
        if (rel === '/') rel = '/index.html';
        const candidates = [path.join(PUBLIC_DIR, rel)];
        if (rel.startsWith('/data/')) candidates.push(path.join(REPO_ROOT, rel));
        for (const full of candidates) {
          if (!full.startsWith(REPO_ROOT)) continue;
          try {
            const stat = fs.statSync(full);
            if (stat.isFile()) {
              const ext = path.extname(full).toLowerCase();
              res.writeHead(200, { 'content-type': MIME[ext] || 'application/octet-stream' });
              fs.createReadStream(full).pipe(res);
              return;
            }
          } catch { /* keep trying */ }
        }
        res.writeHead(404, { 'content-type': 'text/plain' });
        res.end('not found');
      } catch (e) {
        res.writeHead(500, { 'content-type': 'text/plain' });
        res.end('internal error');
      }
    });
    server.on('error', reject);
    server.listen(0, '127.0.0.1', () => resolve(server));
  });
}

let server: http.Server;
let baseURL: string;

test.beforeAll(async () => {
  server = await startStaticServer();
  const addr = server.address();
  if (typeof addr === 'string' || addr === null) throw new Error('server address missing');
  baseURL = `http://127.0.0.1:${addr.port}`;
});

test.afterAll(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

test('mic tab boots an onnxruntime-web runtime reference', async ({ page }) => {
  test.setTimeout(60_000);

  const consoleErrors: string[] = [];
  const consoleAll: string[] = [];
  const failedRequests: string[] = [];

  page.on('console', (msg) => {
    const t = msg.type();
    consoleAll.push(`[${t}] ${msg.text()}`);
    if (t === 'error') consoleErrors.push(msg.text());
  });
  page.on('pageerror', (err) => {
    const text = `pageerror: ${String(err)}`;
    consoleAll.push(text);
    consoleErrors.push(text);
  });
  page.on('requestfailed', (req) => {
    const url = req.url();
    if (url.includes('onnxruntime-web')) {
      failedRequests.push(`${url} (${req.failure()?.errorText || 'unknown'})`);
    }
  });
  page.on('response', (resp) => {
    const url = resp.url();
    if (url.includes('onnxruntime-web') && resp.status() >= 400) {
      failedRequests.push(`${url} (HTTP ${resp.status()})`);
    }
  });

  await page.goto(`${baseURL}/app.html`, { waitUntil: 'domcontentloaded' });

  // Switch to the Microphone Input tab. The mic-boot module observes
  // #panel-microphone gaining the `active` class via the classic
  // switchPanel() handler bound to this button.
  await page.evaluate(() => {
    const btn = document.querySelector('.sidebar-item[data-panel="microphone"]') as HTMLElement | null;
    btn?.click();
  });
  await expect(page.locator('#panel-microphone')).toBeVisible();

  // Within 20 s, the boot path resolves the importmap and exposes a
  // runtime reference. WebGPU-capable browsers expose `window.ort`;
  // WASM-fallback browsers (including default headless Chromium)
  // expose `window.ortWasm`. Either signal counts.
  await expect.poll(
    async () => await page.evaluate(() => Boolean((window as any).ort) || Boolean((window as any).ortWasm)),
    { timeout: 20_000, intervals: [200, 400, 800] },
  ).toBe(true);

  // Console hygiene: a module-resolve error or a 4xx/network failure
  // against any onnxruntime-web URL would mean the importmap pin is
  // wrong. Filter out unrelated noise (font preload warnings, the
  // pre-existing security workflow's favicon chatter, etc.).
  const moduleResolveLike = (s: string) => /\b(?:Failed to resolve module specifier|Importing a module script failed|Module specifier .* does not start|Failed to load module script|TypeError: Failed to fetch dynamically imported module)\b/i.test(s);
  const ortUrlLike = (s: string) => /onnxruntime-web/i.test(s) && /Failed to load resource|net::ERR_|HTTP \d{3}/i.test(s);

  const realErrors = consoleErrors.filter(
    (e) => moduleResolveLike(e) || ortUrlLike(e),
  );
  expect(
    realErrors,
    `unexpected ORT module-resolve / load errors:\n${realErrors.join('\n')}\n\nfull console tail:\n${consoleAll.slice(-30).join('\n')}`,
  ).toEqual([]);

  expect(
    failedRequests,
    `network requests to onnxruntime-web URLs failed:\n${failedRequests.join('\n')}`,
  ).toEqual([]);

  // The pinned URLs must contain the version we documented. This
  // catches an accidental @latest or version drift in app.html's
  // importmap.
  const importmapHasPin = await page.evaluate((v) => {
    const im = document.querySelector('script[type="importmap"]');
    if (!im) return false;
    return im.textContent?.includes(`onnxruntime-web@${v}`) ?? false;
  }, ORT_VERSION);
  expect(importmapHasPin, `importmap should pin onnxruntime-web@${ORT_VERSION}`).toBe(true);

  // The fallback note is hidden in WebGPU mode and visible in WASM
  // mode; both are valid. Just confirm the element exists so a future
  // refactor doesn't silently delete it.
  await expect(page.locator('#micFallbackNote')).toHaveCount(1);

  // If the boot took the WASM branch, the hint must be visible (the
  // user needs to know transcription will be slower). If it took the
  // WebGPU branch, the hint must stay hidden (it's not relevant).
  const took = await page.evaluate(() => ({
    hasOrt: Boolean((window as any).ort),
    hasOrtWasm: Boolean((window as any).ortWasm),
  }));
  if (took.hasOrtWasm && !took.hasOrt) {
    await expect(page.locator('#micFallbackNote')).toBeVisible();
  } else {
    await expect(page.locator('#micFallbackNote')).toBeHidden();
  }
});
