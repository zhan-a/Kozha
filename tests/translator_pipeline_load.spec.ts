/**
 * Translator end-to-end pipeline load — proves the actual Whisper ASR
 * pipeline can be instantiated against jsDelivr-hosted weights from
 * inside app.html, with the avatar's WASM2JS polyfill in flight.
 *
 * The bug this guards: /cwa/allcsa.js is an Emscripten WASM2JS build
 * that overwrites the global WebAssembly with a partial polyfill
 * (Memory/Module/Instance only — no validate, compile, or
 * instantiateStreaming). Without restoring the native WebAssembly
 * before the transformers.js bundle's static `import * as ort from
 * "onnxruntime-web"` evaluates, ORT's SIMD probe throws "WebAssembly
 * SIMD is not supported" and the user sees "We couldn't load the
 * speech model" / "Unsupported model type: whisper".
 *
 * The other smoke specs stub the pipeline (50–200 MB of weights would
 * dominate CI runtime); this one runs the real instantiation but
 * stops short of inference. That's enough to lock in the loader fix
 * without paying the full transcription cost.
 *
 * Run:
 *   npx playwright test tests/translator_pipeline_load.spec.ts --reporter=list
 */
import { test, expect } from '@playwright/test';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..');
const PUBLIC_DIR = path.join(REPO_ROOT, 'public');

const MIME: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.css':  'text/css; charset=utf-8',
  '.js':   'text/javascript; charset=utf-8',
  '.mjs':  'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg':  'image/svg+xml',
  '.sigml': 'application/xml; charset=utf-8',
  '.csv':  'text/csv; charset=utf-8',
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

test('app.html loads the whisper ASR pipeline against jsDelivr weights', async ({ page }) => {
  // Cold-load: bundle (~3 MB) + ORT WASM helpers (~40 MB) + Whisper-tiny.en
  // weights (~40 MB) over jsDelivr. Allow a generous ceiling.
  test.setTimeout(180_000);

  const consoleAll: string[] = [];
  page.on('console', (msg) => consoleAll.push(`[${msg.type()}] ${msg.text()}`));
  page.on('pageerror', (err) => consoleAll.push(`pageerror: ${String(err)}`));

  await page.goto(`${baseURL}/app.html`, { waitUntil: 'domcontentloaded' });

  // Mirror the production code path: restore the native WebAssembly
  // (clobbered by /cwa/allcsa.js) before importing transformers.js,
  // then load the bundle with the same env config app.html uses.
  const result = await page.evaluate(async () => {
    try {
      if ((window as any).__nativeWebAssembly) {
        (window as any).WebAssembly = (window as any).__nativeWebAssembly;
      }
      const url = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.web.min.js';
      const mod: any = await import(/* @vite-ignore */ url);
      if (!mod || typeof mod.pipeline !== 'function') {
        return { ok: false, message: 'pipeline export missing' };
      }
      if (mod.env) {
        mod.env.allowRemoteModels = true;
        mod.env.allowLocalModels = false;
        if (mod.env.backends?.onnx?.wasm) {
          mod.env.backends.onnx.wasm.numThreads = 1;
          mod.env.backends.onnx.wasm.wasmPaths =
            'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
        }
      }
      const asr = await mod.pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');
      return { ok: true, type: typeof asr };
    } catch (err: any) {
      return { ok: false, message: err?.message || String(err) };
    }
  });

  expect(
    result.ok,
    `pipeline load failed: ${result.message}\nconsole tail:\n${consoleAll.slice(-30).join('\n')}`,
  ).toBe(true);
  expect(result.type).toBe('function');
});
