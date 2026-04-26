/**
 * Authoring-page (chat2hamnosys/index.html) avatar visibility regression.
 *
 * Catches the failure mode that motivated prompt 07: the CWASA canvas
 * mounted but reported zero bounding rect because the [hidden] attribute
 * was overridden by an author `display: flex` rule, and the
 * percentage-only sizing chain through .CWASAAvatar → .divAv → .canvasAv
 * collapsed inside a centered flex container.
 *
 * What this guards:
 *   1. The CWASA-mount slot has a non-zero rendered rect once a
 *      session reports SiGML, at viewport widths 1440, 1024, 768, 414.
 *   2. The slot's bounding rect intersects the viewport — i.e. the
 *      canvas is on-screen, not pushed below the fold or off-page.
 *   3. The fallback snapshot card takes over (with a non-zero rect)
 *      if the CWASA bundle fails to mount within the 6 s deadline,
 *      instead of leaving an empty box.
 *
 * Pure local: spins up the same one-off static HTTP server pattern as
 * tests/contrib_step4_layout.spec.ts, plus an in-test API stub via
 * `page.route()` so the test does not require an LLM key or backend.
 *
 * Run:  npx playwright test tests/c2h_avatar_visible.spec.ts --reporter=list
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
  '.css':  'text/css; charset=utf-8',
  '.js':   'text/javascript; charset=utf-8',
  '.mjs':  'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg':  'image/svg+xml',
  '.ttf':  'font/ttf',
  '.woff': 'font/woff',
  '.woff2':'font/woff2',
  '.ico':  'image/x-icon',
  '.png':  'image/png',
  '.jpg':  'image/jpeg',
  '.webp': 'image/webp',
  '.jar':  'application/java-archive',
  '.sigml':'application/xml; charset=utf-8',
  '.csv':  'text/csv; charset=utf-8',
};

type Server = { server: http.Server; port: number };

function startServer(): Promise<Server> {
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      try {
        const parsed = new URL(req.url || '/', 'http://localhost');
        let pathname = decodeURIComponent(parsed.pathname);
        if (pathname === '/' || pathname === '') pathname = '/chat2hamnosys/index.html';
        const filePath = path.join(PUBLIC_DIR, pathname);
        if (!filePath.startsWith(PUBLIC_DIR)) { res.writeHead(403).end('forbidden'); return; }
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

const VIEWPORTS = [
  { label: '1440', width: 1440, height: 900 },
  { label: '1024', width: 1024, height: 768 },
  { label: '768',  width: 768,  height: 1024 },
  { label: '414',  width: 414,  height: 896 },
] as const;

// Minimal SiGML payload from the BSL "HELLO" entry — short enough that
// CWASA can compile it on the test runner without exercising the
// avatar's heavier animation paths.
const STUB_SIGML = `<sigml>
  <hns_sign gloss="HELLO">
    <hamnosys_manual>
      <hamflathand/><hamthumboutmod/>
      <hamextfingeru/><hampalml/>
      <hamforehead/><hamlrat/>
      <hammover/>
    </hamnosys_manual>
  </hns_sign>
</sigml>`;

function stubEnvelope(extra: Partial<Record<string, unknown>> = {}): unknown {
  return {
    session_id: '00000000-0000-0000-0000-000000000001',
    sign_language: 'bsl',
    state: 'rendered',
    gloss: 'HELLO',
    description_prose: 'wave a flat hand near the temple',
    history: [],
    pending_questions: [],
    clarifications: [],
    gaps: [],
    parameters: {},
    preview: { sigml: STUB_SIGML, message: 'stubbed preview for layout test' },
    ...extra,
  };
}

async function installApiStubs(page: import('@playwright/test').Page) {
  // Match every chat2hamnosys API path. The stub returns the same
  // envelope for both POST /sessions (create) and the SSE/GET refresh
  // path the app uses for re-fetches.
  await page.route('**/api/chat2hamnosys/**', async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname.endsWith('/sessions') && route.request().method() === 'POST') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          session_id: '00000000-0000-0000-0000-000000000001',
          session_token: 'stub-token',
          session: stubEnvelope(),
        }),
      });
      return;
    }
    if (url.pathname.includes('/events')) {
      // Long-poll/SSE — close immediately so the app falls back to
      // poll-on-demand. Empty body means no events.
      await route.fulfill({ status: 204, body: '' });
      return;
    }
    if (url.pathname.endsWith('/session') || url.pathname.match(/\/sessions\/[^/]+$/)) {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ session: stubEnvelope() }),
      });
      return;
    }
    await route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
  });
}

async function waitForCwasaSlot(page: import('@playwright/test').Page) {
  // The CWASA mount becomes visible as soon as renderPreview() unhides
  // it. The canvas itself appears later (after the bundle downloads
  // and CWASA.init runs). Both the mount and the canvas (when present)
  // must have non-zero dimensions.
  await page.waitForFunction(() => {
    const m = document.getElementById('cwasaMount');
    const f = document.getElementById('snapshotFallback');
    const visible = (el: HTMLElement | null) =>
      !!el && !el.hidden && el.getBoundingClientRect().width > 0;
    return visible(m as HTMLElement | null) || visible(f as HTMLElement | null);
  }, { timeout: 10_000 });
}

test.describe('chat2hamnosys/index.html avatar mount visibility', () => {
  for (const vp of VIEWPORTS) {
    test(`avatar slot has non-zero rect that intersects viewport at ${vp.label}px`, async ({ page }) => {
      await page.setViewportSize({ width: vp.width, height: vp.height });
      await installApiStubs(page);
      // localStorage tokens — the gate will redirect to /contribute.html
      // without them. We seed before navigation so the very first GET
      // sees a logged-in contributor.
      await page.addInitScript(() => {
        try {
          localStorage.setItem('bridgn.contributor_token', 'stub');
          localStorage.setItem('bridgn.contributor_expires_at', String(Math.floor(Date.now() / 1000) + 3600));
        } catch (_e) { /* ignore */ }
      });
      await page.goto(`http://127.0.0.1:${serverHandle.port}/chat2hamnosys/index.html`, {
        waitUntil: 'domcontentloaded',
      });

      // The mobile-tabs (≤ 880 px viewport) switch the preview panel to
      // display:none until its tab is selected. Activate the Preview
      // tab so the avatar slot is laid out at all.
      if (vp.width <= 880) {
        await page.locator('.mobile-tabs .tab[data-tab="preview"]').click();
        await page.waitForFunction(() => {
          const p = document.querySelector('.panel-preview') as HTMLElement | null;
          return !!p && getComputedStyle(p).display !== 'none';
        }, { timeout: 5_000 });
      }

      // Wait for either the CWASA slot or the snapshot fallback to be
      // visible — both are valid outcomes per the acceptance criteria.
      await waitForCwasaSlot(page);

      // Pick whichever slot is currently the visible one — the canvas
      // (if CWASA mounted) takes priority, the fallback otherwise.
      const visibleSlot = await page.evaluate(() => {
        const cwasaMount = document.getElementById('cwasaMount');
        const fallback = document.getElementById('snapshotFallback');
        const isVisible = (el: HTMLElement | null) =>
          !!el && !el.hidden && el.getBoundingClientRect().width > 0;
        if (isVisible(cwasaMount as HTMLElement | null)) return 'cwasa';
        if (isVisible(fallback as HTMLElement | null))   return 'fallback';
        return null;
      });
      expect(visibleSlot, 'one of the preview slots is visible').toBeTruthy();

      const slotId = visibleSlot === 'cwasa' ? '#cwasaMount' : '#snapshotFallback';
      const slot = page.locator(slotId);
      const slotBox = await slot.boundingBox();

      // 1. Non-zero rendered rect — the headline guarantee from prompt
      //    07. A zero-size mount is the failure mode we are guarding.
      expect(slotBox?.width  ?? 0, `${slotId} width`).toBeGreaterThan(0);
      expect(slotBox?.height ?? 0, `${slotId} height`).toBeGreaterThan(0);

      // 2. Slot intersects the viewport. A canvas pushed below the fold
      //    or pushed off-page (the original symptom) would technically
      //    have a non-zero rect but not be visible.
      const intersects = await page.evaluate((id) => {
        const el = document.querySelector(id) as HTMLElement | null;
        if (!el) return false;
        const r = el.getBoundingClientRect();
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        const off = r.right <= 0 || r.bottom <= 0 || r.left >= vw || r.top >= vh;
        return !off;
      }, slotId);
      expect(intersects, `${slotId} bounding rect intersects the viewport`).toBeTruthy();

      // 3. Slot meets the prompt's 320 px minimum-height floor.
      expect(slotBox?.height ?? 0, `${slotId} hits the 320px min-height floor`)
        .toBeGreaterThanOrEqual(320 - 2); // 1px tolerance on borders/sub-pixel rounding

      // 4. If the canvas mounted, it must inherit the slot's size — i.e.
      //    the percentage chain resolved against a definite parent.
      if (visibleSlot === 'cwasa') {
        const canvasBox = await page.locator('#cwasaMount canvas.canvasAv').boundingBox().catch(() => null);
        if (canvasBox) {
          expect(canvasBox.width,  'canvas inherits slot width').toBeGreaterThan(0);
          expect(canvasBox.height, 'canvas inherits slot height').toBeGreaterThan(0);
        }
      }
    });
  }

  test('status chip reflects loading → playing transitions', async ({ page }) => {
    await page.setViewportSize({ width: 1024, height: 768 });
    await installApiStubs(page);
    await page.addInitScript(() => {
      try {
        localStorage.setItem('bridgn.contributor_token', 'stub');
        localStorage.setItem('bridgn.contributor_expires_at', String(Math.floor(Date.now() / 1000) + 3600));
      } catch (_e) { /* ignore */ }
    });
    await page.goto(`http://127.0.0.1:${serverHandle.port}/chat2hamnosys/index.html`, {
      waitUntil: 'domcontentloaded',
    });

    // Chip exists and starts visible. data-state moves to "loading"
    // once the session is created and the warmup event fires.
    const chip = page.locator('#previewStatus');
    await expect(chip).toBeVisible();

    // Wait for it to leave the initial 'idle' state. Either it lands on
    // 'loading' (still warming) or jumps straight to 'playing' /
    // 'unavailable'. Anything other than 'idle' proves the chip is
    // wired to the bridge.
    await page.waitForFunction(() => {
      const el = document.getElementById('previewStatus');
      return !!el && el.dataset.state !== 'idle';
    }, { timeout: 10_000 });

    const finalState = await chip.getAttribute('data-state');
    expect(finalState, 'status chip moves off idle once the session boots').not.toBe('idle');
  });
});
