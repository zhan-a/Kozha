/**
 * Step 5 ("Submit for Deaf review") timeline-overflow regression.
 *
 * Catches the failure mode that motivated prompt 06: the horizontal
 * timeline (`draft → pending_review → validated → exported`) used
 * `grid-template-columns: repeat(5, 1fr)`. Because `1fr` is
 * `minmax(auto, 1fr)`, a label longer than its 1fr share would push
 * the column wider than the assigned share, and the last node (and
 * trailing label, e.g. `Live`) ended up sitting past the card's right
 * edge at every viewport ≥ 320 px.
 *
 * What this guards:
 *
 *   1. The timeline's painted rect stays inside the card's painted
 *      rect at six viewport widths (the prompt's spec list).
 *   2. The status labels match the backend `SignStatus` enum +
 *      post-validation `exported` event verbatim — the timeline must
 *      not drift if the backend renames a status.
 *   3. The current step carries `aria-current` so SR users (and the
 *      surrounding copy "we are here") can identify it; the connector
 *      lines stay decorative (CSS pseudo-elements; not in the a11y
 *      tree).
 *   4. At narrow viewports the timeline switches to a vertical layout
 *      with no zero-height row and no overlapping nodes.
 *
 * Pure local: spins up a one-off static HTTP server pointed at
 * `public/`. No deployed site, no LLM key.
 *
 * Run:  npx playwright test tests/contrib_step5_timeline.spec.ts --reporter=list
 */
import { test, expect } from '@playwright/test';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { AddressInfo } from 'node:net';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..');
const PUBLIC_DIR = path.join(REPO_ROOT, 'public');

const MIME: Record<string, string> = {
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
};

type Server = { server: http.Server; port: number };

function startServer(): Promise<Server> {
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      try {
        const parsed = new URL(req.url || '/', 'http://localhost');
        let pathname = decodeURIComponent(parsed.pathname);
        if (pathname === '/' || pathname === '') pathname = '/contribute.html';
        const filePath = path.join(PUBLIC_DIR, pathname);
        if (!filePath.startsWith(PUBLIC_DIR)) {
          res.writeHead(403).end('forbidden');
          return;
        }
        if (!fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
          res.writeHead(404).end('not found');
          return;
        }
        const ext = path.extname(filePath).toLowerCase();
        const mime = MIME[ext] || 'application/octet-stream';
        res.writeHead(200, { 'Content-Type': mime, 'Cache-Control': 'no-store' });
        fs.createReadStream(filePath).pipe(res);
      } catch (e) {
        res.writeHead(500).end(String(e));
      }
    });
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const addr = server.address() as AddressInfo;
      resolve({ server, port: addr.port });
    });
  });
}

let serverHandle: Server;

test.beforeAll(async () => {
  serverHandle = await startServer();
});

test.afterAll(async () => {
  await new Promise<void>((resolve) => serverHandle.server.close(() => resolve()));
});

// Six widths from the prompt's spec list. 1440 / 1280 / 1024 are the
// desktop range where the panel's 2-col grid forces the card narrow;
// 768 is the single-col panel breakpoint; 414 / 360 are the mobile
// extremes that previously hard-clipped the trailing node.
const VIEWPORTS = [
  { label: '1440', width: 1440, height: 900 },
  { label: '1280', width: 1280, height: 900 },
  { label: '1024', width: 1024, height: 768 },
  { label: '768',  width: 768,  height: 1024 },
  { label: '414',  width: 414,  height: 896 },
  { label: '360',  width: 360,  height: 740 },
] as const;

// Status names mirror backend/chat2hamnosys/models.py SignStatus + the
// post-validation `exported` event in review/router.py. If the backend
// renames a status, this assertion fails — that is the desired signal
// (the UI label must move with the source).
const EXPECTED_STATUSES = ['draft', 'pending_review', 'validated', 'exported'];

test.describe('contribute.html step 5 review timeline', () => {
  for (const vp of VIEWPORTS) {
    test(`timeline rect stays inside card rect at ${vp.label}px`, async ({ page }) => {
      await page.setViewportSize({ width: vp.width, height: vp.height });
      await page.goto(`http://127.0.0.1:${serverHandle.port}/contribute.html`, {
        waitUntil: 'domcontentloaded',
      });
      await page.click('#c2-tab-5');
      await page.waitForSelector('#c2-panel-5.is-active', { state: 'visible' });

      const cardBox = await page.locator('#c2-panel-5 .c2-viz-5__card').boundingBox();
      const tlBox = await page.locator('#c2-panel-5 .c2-viz-5__timeline').boundingBox();
      expect(cardBox, 'card bounding box').not.toBeNull();
      expect(tlBox, 'timeline bounding box').not.toBeNull();
      const cardR = cardBox!.x + cardBox!.width;
      const cardL = cardBox!.x;
      const tlR = tlBox!.x + tlBox!.width;
      const tlL = tlBox!.x;

      // Sub-pixel slack of 1 px so a fractional grid track size doesn't
      // fail the assertion on its own. Anything beyond that is a real
      // overflow — the failure mode the prompt is fixing.
      expect(tlR, `timeline right edge ≤ card right edge at ${vp.label}px`).toBeLessThanOrEqual(cardR + 1);
      expect(tlL, `timeline left edge ≥ card left edge at ${vp.label}px`).toBeGreaterThanOrEqual(cardL - 1);

      // Per-li bounds — catches the case where the OL sits inside the
      // card but a child column escapes (would happen if a future
      // change removes `minmax(0, 1fr)`).
      const liRects = await page
        .locator('#c2-panel-5 .c2-viz-5__timeline li')
        .evaluateAll((els) => els.map((el) => el.getBoundingClientRect()));
      for (let i = 0; i < liRects.length; i++) {
        const r = liRects[i];
        expect(r.right, `li[${i}] right edge ≤ card right edge at ${vp.label}px`).toBeLessThanOrEqual(cardR + 1);
        expect(r.left, `li[${i}] left edge ≥ card left edge at ${vp.label}px`).toBeGreaterThanOrEqual(cardL - 1);
        expect(r.height, `li[${i}] non-zero height at ${vp.label}px`).toBeGreaterThan(0);
      }
    });
  }

  test('vertical layout at ≤ 600 px viewport stacks li nodes in one column', async ({ page }) => {
    // Per the prompt, ≤ 600 px viewport must use a vertical timeline.
    // We confirm by asserting all li elements share the same x position
    // (single column) and step downward in y.
    await page.setViewportSize({ width: 360, height: 740 });
    await page.goto(`http://127.0.0.1:${serverHandle.port}/contribute.html`, {
      waitUntil: 'domcontentloaded',
    });
    await page.click('#c2-tab-5');
    const liRects = await page
      .locator('#c2-panel-5 .c2-viz-5__timeline li')
      .evaluateAll((els) => els.map((el) => el.getBoundingClientRect()));
    expect(liRects.length).toBe(EXPECTED_STATUSES.length);
    const firstX = liRects[0].x;
    let prevY = -Infinity;
    for (let i = 0; i < liRects.length; i++) {
      expect(liRects[i].x, `li[${i}] starts at the same x as li[0] (single column)`).toBeCloseTo(firstX, 0);
      expect(liRects[i].y, `li[${i}] sits below li[${i - 1}]`).toBeGreaterThan(prevY);
      // No zero-height row, no overlapping nodes — both failure modes
      // the prompt explicitly calls out for the vertical layout.
      expect(liRects[i].height, `li[${i}] non-zero height`).toBeGreaterThan(0);
      prevY = liRects[i].y + liRects[i].height;
    }
  });

  test('status labels match the backend SignStatus + exported event verbatim', async ({ page }) => {
    await page.setViewportSize({ width: 1024, height: 768 });
    await page.goto(`http://127.0.0.1:${serverHandle.port}/contribute.html`, {
      waitUntil: 'domcontentloaded',
    });
    await page.click('#c2-tab-5');
    const labels = await page
      .locator('#c2-panel-5 .c2-viz-5__timeline li')
      .evaluateAll((els) => els.map((el) => (el.textContent || '').trim()));
    expect(labels).toEqual(EXPECTED_STATUSES);
    const dataStatus = await page
      .locator('#c2-panel-5 .c2-viz-5__timeline li')
      .evaluateAll((els) => els.map((el) => el.getAttribute('data-status')));
    expect(dataStatus).toEqual(EXPECTED_STATUSES);
  });

  test('exactly one stage carries aria-current="step"', async ({ page }) => {
    await page.setViewportSize({ width: 1024, height: 768 });
    await page.goto(`http://127.0.0.1:${serverHandle.port}/contribute.html`, {
      waitUntil: 'domcontentloaded',
    });
    await page.click('#c2-tab-5');
    const current = await page
      .locator('#c2-panel-5 .c2-viz-5__timeline li[aria-current="step"]')
      .evaluateAll((els) => els.map((el) => el.getAttribute('data-status')));
    expect(current).toEqual(['pending_review']);
  });

  test('timeline is an ordered list with an accessible name', async ({ page }) => {
    await page.setViewportSize({ width: 1024, height: 768 });
    await page.goto(`http://127.0.0.1:${serverHandle.port}/contribute.html`, {
      waitUntil: 'domcontentloaded',
    });
    await page.click('#c2-tab-5');
    const tag = await page.locator('#c2-panel-5 .c2-viz-5__timeline').evaluate((el) => el.tagName);
    expect(tag).toBe('OL');
    const ariaLabel = await page.locator('#c2-panel-5 .c2-viz-5__timeline').getAttribute('aria-label');
    expect(ariaLabel).toBeTruthy();
  });
});
