/**
 * Authoring-page (chat2hamnosys/index.html) layout regression — prompt 08.
 *
 * Catches the failure mode that motivated the fix: a large empty band
 * opened up below the chat panel because the layout used
 * `min-height: 100vh` on the outer container while the chat had
 * `min-height: 60vh` and the right column collapsed to its content.
 * The page outer-scrolled even though every panel was hollow.
 *
 * What this guards (per prompt 08 acceptance):
 *   1. At every supported viewport the page's outer scrollHeight is at
 *      most 1.05× the viewport height. (Inner columns may scroll; the
 *      page itself does not.)
 *   2. The grid resolves to three columns at ≥ 1024 px, two columns at
 *      768–1023 px, and one column at < 768 px — verified by counting
 *      distinct grid-column-start values across the visible panels.
 *   3. The chat composer (.chat-form) sits at the bottom of the chat
 *      panel: form bottom is within 2 px of panel bottom (so it stays
 *      reachable without scrolling the chat column).
 *   4. The legacy `.mobile-tabs` element no longer exists in the DOM
 *      (prompt 08 collapsed it into the single responsive grid).
 *
 * Pure local: same one-off static HTTP server pattern as
 * `tests/c2h_avatar_visible.spec.ts` and `tests/contrib_step4_layout.spec.ts`.
 *
 * Run:  npx playwright test tests/c2h_layout_no_outer_scroll.spec.ts --reporter=list
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

function stubEnvelope(): unknown {
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
  };
}

async function installApiStubs(page: import('@playwright/test').Page) {
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

async function gotoAuthoring(page: import('@playwright/test').Page, port: number) {
  await page.addInitScript(() => {
    try {
      localStorage.setItem('bridgn.contributor_token', 'stub');
      localStorage.setItem('bridgn.contributor_expires_at', String(Math.floor(Date.now() / 1000) + 3600));
    } catch (_e) { /* ignore */ }
  });
  await page.goto(`http://127.0.0.1:${port}/chat2hamnosys/index.html`, {
    waitUntil: 'domcontentloaded',
  });
  // Let layout settle — fonts, the language picker fetch, and any
  // post-boot reflow.
  await page.waitForLoadState('networkidle').catch(() => { /* okay if it never goes idle */ });
  await page.waitForTimeout(150);
}

const VIEWPORTS = [
  { label: '1440', width: 1440, height: 900,  expectedColumns: 3 },
  { label: '1280', width: 1280, height: 800,  expectedColumns: 3 },
  { label: '1024', width: 1024, height: 768,  expectedColumns: 3 },
  { label: '768',  width: 768,  height: 1024, expectedColumns: 2 },
  { label: '414',  width: 414,  height: 896,  expectedColumns: 1 },
] as const;

test.describe('chat2hamnosys/index.html layout — no outer scroll, responsive columns', () => {
  for (const vp of VIEWPORTS) {
    test(`outer page does not scroll at ${vp.label}px`, async ({ page }) => {
      await page.setViewportSize({ width: vp.width, height: vp.height });
      await installApiStubs(page);
      await gotoAuthoring(page, serverHandle.port);

      // The headline guarantee. document.scrollingElement is <html>
      // when body has overflow:hidden; .scrollHeight is the content
      // height (clamped to box height when overflow:hidden takes effect).
      const measurement = await page.evaluate(() => {
        const root = document.scrollingElement || document.documentElement;
        return {
          scrollHeight: root.scrollHeight,
          clientHeight: root.clientHeight,
          innerHeight:  window.innerHeight,
          bodyHeight:   document.body.getBoundingClientRect().height,
        };
      });

      // 1.05× tolerance: 5 % covers sub-pixel rounding plus rare
      // single-px reflows from font-loading races.
      const ceiling = vp.height * 1.05;
      expect(measurement.scrollHeight, `outer scrollHeight at ${vp.label}px`)
        .toBeLessThanOrEqual(ceiling);
      expect(measurement.bodyHeight, `body box height at ${vp.label}px`)
        .toBeLessThanOrEqual(ceiling);
    });

    test(`grid resolves to ${vp.expectedColumns} column(s) at ${vp.label}px`, async ({ page }) => {
      await page.setViewportSize({ width: vp.width, height: vp.height });
      await installApiStubs(page);
      await gotoAuthoring(page, serverHandle.port);

      // Count distinct rendered left edges across the three panels.
      // `getComputedStyle().gridColumnStart` is unreliable when items
      // are placed via `grid-area: <name>` (some engines return the
      // area name, others 'auto'); rounding the rendered .left coalesces
      // sub-pixel jitter into a single bucket per visible column.
      const distinctColumns = await page.evaluate(() => {
        const panels = Array.from(document.querySelectorAll('.layout > .panel')) as HTMLElement[];
        const lefts = panels
          .filter((p) => getComputedStyle(p).display !== 'none')
          .map((p) => Math.round(p.getBoundingClientRect().left));
        return new Set(lefts).size;
      });

      expect(distinctColumns, `column count at ${vp.label}px`).toBe(vp.expectedColumns);
    });
  }

  test('chat composer sticks to the bottom of the chat panel at 1280px', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 800 });
    await installApiStubs(page);
    await gotoAuthoring(page, serverHandle.port);

    const offset = await page.evaluate(() => {
      const panel = document.querySelector('.panel-chat') as HTMLElement | null;
      const form  = document.getElementById('chatForm') as HTMLElement | null;
      if (!panel || !form) return null;
      const pr = panel.getBoundingClientRect();
      const fr = form.getBoundingClientRect();
      return { panelBottom: pr.bottom, formBottom: fr.bottom };
    });

    expect(offset, 'chat panel and form measurable').not.toBeNull();
    // Within 2 px — accounts for sub-pixel rounding and the panel border.
    expect(Math.abs((offset!.panelBottom) - (offset!.formBottom)), 'composer pinned to panel bottom')
      .toBeLessThanOrEqual(2);
  });

  test('chat composer remains reachable on mobile (414px)', async ({ page }) => {
    await page.setViewportSize({ width: 414, height: 896 });
    await installApiStubs(page);
    await gotoAuthoring(page, serverHandle.port);

    const reachable = await page.evaluate(() => {
      const form = document.getElementById('chatForm') as HTMLElement | null;
      if (!form) return false;
      const r = form.getBoundingClientRect();
      // Form's bottom edge sits within the viewport — the user
      // doesn't need to outer-scroll to find Send.
      return r.top < window.innerHeight && r.bottom <= window.innerHeight + 2;
    });

    expect(reachable, 'composer is within the viewport on mobile').toBe(true);
  });

  test('legacy .mobile-tabs element is gone (collapsed into responsive grid)', async ({ page }) => {
    await page.setViewportSize({ width: 414, height: 896 });
    await installApiStubs(page);
    await gotoAuthoring(page, serverHandle.port);

    const tabs = await page.locator('.mobile-tabs').count();
    expect(tabs, 'no .mobile-tabs in the DOM').toBe(0);
  });

  test('tab order on desktop is chat → preview → inspect', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await installApiStubs(page);
    await gotoAuthoring(page, serverHandle.port);

    // The DOM source order maps to tab order when no element has an
    // explicit positive tabindex. Verify the section ordering.
    const order = await page.evaluate(() => {
      const sections = Array.from(document.querySelectorAll('main.layout > section.panel')) as HTMLElement[];
      return sections.map((s) => s.dataset.panel);
    });

    expect(order, 'panel DOM order drives tab order').toEqual(['chat', 'preview', 'inspect']);
  });
});
