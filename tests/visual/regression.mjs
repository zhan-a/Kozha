#!/usr/bin/env node
/*
 * Visual regression runner for the main-page surface (index.html + app.html).
 *
 * Captures pixel-level screenshots of /, /app (and their mobile variants)
 * through Puppeteer against a minimal static HTTP server serving ./public.
 * Compares the result against a checked-in baseline under
 * tests/visual/baseline/; writes the live capture to tests/visual/current/
 * and a diff image to tests/visual/diff/<name>.diff.png.
 *
 * Modes:
 *   node tests/visual/regression.mjs --baseline    # overwrite the baseline
 *   node tests/visual/regression.mjs               # compare current against baseline
 *
 * Exit: non-zero if any scenario's pixel-diff ratio exceeds MAX_DIFF (0.5%).
 *
 * Determinism: we emulate prefers-reduced-motion, await document.fonts.ready
 * and networkidle, inject a stop-all-animations stylesheet (the app page does
 * not honor reduced-motion), and hide CWASA avatar surfaces which load
 * asynchronously and would otherwise introduce flake.
 */

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import puppeteer from 'puppeteer';
import { PNG } from 'pngjs';
import pixelmatch from 'pixelmatch';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..', '..');
const PUBLIC_DIR = path.join(REPO_ROOT, 'public');
const BASELINE_DIR = path.join(__dirname, 'baseline');
const CURRENT_DIR = path.join(__dirname, 'current');
const DIFF_DIR = path.join(__dirname, 'diff');

const ARGS = new Set(process.argv.slice(2));
const UPDATE_BASELINE = ARGS.has('--baseline');
const MAX_DIFF_RATIO = 0.005; // 0.5%

for (const d of [BASELINE_DIR, CURRENT_DIR, DIFF_DIR]) {
  fs.mkdirSync(d, { recursive: true });
}

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.ttf': 'font/ttf',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ico': 'image/x-icon',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.webp': 'image/webp',
  '.txt': 'text/plain; charset=utf-8',
};

function safeStatic(req, res) {
  const parsed = new URL(req.url, 'http://localhost');
  let pathname = decodeURIComponent(parsed.pathname);
  if (pathname === '/' || pathname === '') pathname = '/index.html';
  if (!path.extname(pathname)) {
    const candidate = path.join(PUBLIC_DIR, pathname + '.html');
    if (fs.existsSync(candidate)) {
      return serveFile(res, candidate);
    }
  }
  const filePath = path.join(PUBLIC_DIR, pathname);
  if (!filePath.startsWith(PUBLIC_DIR)) {
    res.writeHead(403).end('forbidden');
    return;
  }
  if (!fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
    res.writeHead(404).end('not found');
    return;
  }
  serveFile(res, filePath);
}

function serveFile(res, filePath) {
  const ext = path.extname(filePath).toLowerCase();
  const mime = MIME[ext] || 'application/octet-stream';
  res.writeHead(200, { 'Content-Type': mime, 'Cache-Control': 'no-store' });
  fs.createReadStream(filePath).pipe(res);
}

async function startServer() {
  return await new Promise((resolve, reject) => {
    const server = http.createServer(safeStatic);
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const addr = server.address();
      resolve({ server, port: addr.port });
    });
  });
}

// Scenarios. Each scenario renders one page at one viewport.
const SCENARIOS = [
  { name: 'landing-desktop', url: '/',        viewport: { width: 1440, height: 900 } },
  { name: 'landing-mobile',  url: '/',        viewport: { width: 390,  height: 844 } },
  { name: 'app-desktop',     url: '/app.html', viewport: { width: 1440, height: 900 } },
  { name: 'app-mobile',      url: '/app.html', viewport: { width: 390,  height: 844 } },
];

// Injected to freeze animations, disable caret blinking, and hide the
// asynchronously-loaded avatar regions that introduce non-determinism.
const STABILIZE_CSS = `
*, *::before, *::after {
  animation-duration: 0ms !important;
  animation-delay: 0ms !important;
  animation-iteration-count: 1 !important;
  animation-play-state: paused !important;
  transition-duration: 0ms !important;
  transition-delay: 0ms !important;
  caret-color: transparent !important;
}
/* Hide CWASA avatar surfaces — they load WASM/JS asynchronously and their
   pixel output varies between runs even when our CSS hasn't changed. */
.CWASAAvatar, .CWASAGUI, .demo-avatar-area, .avatar-wrapper, .avatar-stage,
.avatar-loading, .spinner, #heroAvatar {
  visibility: hidden !important;
}
`;

async function capture(browser, server, scenario) {
  const page = await browser.newPage();
  try {
    await page.setViewport({
      width: scenario.viewport.width,
      height: scenario.viewport.height,
      deviceScaleFactor: 1,
    });
    await page.emulateMediaFeatures([
      { name: 'prefers-reduced-motion', value: 'reduce' },
    ]);
    const url = `http://127.0.0.1:${server.port}${scenario.url}`;
    await page.goto(url, { waitUntil: 'networkidle0', timeout: 30000 });
    // Inject stabilizer after initial render so any computed-style-dependent
    // JS has already settled.
    await page.addStyleTag({ content: STABILIZE_CSS });
    // Wait for fonts (Instrument Serif + DM Sans come from Google Fonts).
    await page.evaluate(() => document.fonts?.ready);
    // Give the browser one paint frame to apply stabilizer + font swap.
    await new Promise((r) => setTimeout(r, 250));
    const buf = await page.screenshot({ fullPage: false, type: 'png' });
    // Puppeteer 23 returns Uint8Array; pngjs expects a Node Buffer.
    return Buffer.isBuffer(buf) ? buf : Buffer.from(buf);
  } finally {
    await page.close();
  }
}

function readPNG(filePath) {
  return PNG.sync.read(fs.readFileSync(filePath));
}

function compareAgainstBaseline(name, pngBuf) {
  const currentPath = path.join(CURRENT_DIR, `${name}.png`);
  fs.writeFileSync(currentPath, pngBuf);
  const baselinePath = path.join(BASELINE_DIR, `${name}.png`);
  if (!fs.existsSync(baselinePath)) {
    return { ok: false, reason: 'no baseline on disk; run with --baseline first' };
  }
  const cur = PNG.sync.read(pngBuf);
  const base = readPNG(baselinePath);
  if (cur.width !== base.width || cur.height !== base.height) {
    return {
      ok: false,
      reason: `size mismatch (baseline ${base.width}x${base.height}, current ${cur.width}x${cur.height})`,
    };
  }
  const diff = new PNG({ width: cur.width, height: cur.height });
  const diffPixels = pixelmatch(base.data, cur.data, diff.data, cur.width, cur.height, {
    threshold: 0.1, // per-pixel threshold; lower = more sensitive
    includeAA: false,
  });
  const total = cur.width * cur.height;
  const ratio = diffPixels / total;
  fs.writeFileSync(path.join(DIFF_DIR, `${name}.diff.png`), PNG.sync.write(diff));
  return { ok: ratio <= MAX_DIFF_RATIO, ratio, diffPixels, total };
}

async function main() {
  const { server, port } = await startServer();
  // eslint-disable-next-line no-console
  console.log(`[visual] static server listening on 127.0.0.1:${port}`);
  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });
  const failures = [];
  try {
    for (const scenario of SCENARIOS) {
      // eslint-disable-next-line no-console
      process.stdout.write(`[visual] ${scenario.name} ... `);
      const pngBuf = await capture(browser, { port }, scenario);
      if (UPDATE_BASELINE) {
        fs.writeFileSync(path.join(BASELINE_DIR, `${scenario.name}.png`), pngBuf);
        // eslint-disable-next-line no-console
        console.log('baseline written');
        continue;
      }
      const result = compareAgainstBaseline(scenario.name, pngBuf);
      if (result.ok) {
        // eslint-disable-next-line no-console
        console.log(
          `ok  (${result.ratio !== undefined ? (result.ratio * 100).toFixed(3) + '%' : 'n/a'})`,
        );
      } else {
        failures.push({ name: scenario.name, ...result });
        // eslint-disable-next-line no-console
        console.log(
          `FAIL (${
            result.ratio !== undefined
              ? (result.ratio * 100).toFixed(3) + '% > ' + (MAX_DIFF_RATIO * 100).toFixed(2) + '%'
              : result.reason
          })`,
        );
      }
    }
  } finally {
    await browser.close();
    server.close();
  }
  if (failures.length) {
    // eslint-disable-next-line no-console
    console.error(`[visual] ${failures.length} regression(s) above ${MAX_DIFF_RATIO * 100}% threshold`);
    process.exit(1);
  } else {
    // eslint-disable-next-line no-console
    console.log(`[visual] all ${SCENARIOS.length} scenarios within ${MAX_DIFF_RATIO * 100}% threshold`);
  }
}

main().catch((err) => {
  // eslint-disable-next-line no-console
  console.error(err);
  process.exit(2);
});
