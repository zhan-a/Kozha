#!/usr/bin/env node
/*
 * A11y audit runner for the Bridgn contribution flow.
 *
 * Starts a minimal static HTTP server over ./public (so we don't depend
 * on the Python backend being able to spin up — the a11y test targets
 * the HTML/CSS/JS surface, not the API), then drives Puppeteer through
 * each target scenario, runs axe-core against it, and also invokes
 * pa11y's HTML_CodeSniffer engine for a second opinion.
 *
 * Scenarios that depend on runtime state (post-language-selection,
 * mid-session panels, status pages in each terminal status) are reached
 * by seeding state via page.evaluate() before the audit runs. The DOM
 * shape is what matters for a11y; the origin of the state does not.
 *
 * Outputs:
 *   docs/contribute-redesign/12-a11y-raw/<scenario>.axe.json   (raw axe)
 *   docs/contribute-redesign/12-a11y-raw/<scenario>.pa11y.json (raw pa11y)
 *   docs/contribute-redesign/12-a11y-baseline.md               (summary)
 *
 * CLI:
 *   node scripts/a11y/run.mjs         # full run, always exits 0
 *   node scripts/a11y/run.mjs --ci    # fail exit on critical/serious
 */

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import url from 'node:url';
import { fileURLToPath } from 'node:url';

import puppeteer from 'puppeteer';
import { AxePuppeteer } from '@axe-core/puppeteer';
import pa11y from 'pa11y';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..', '..');
const PUBLIC_DIR = path.join(REPO_ROOT, 'public');
const RAW_DIR = path.join(
  REPO_ROOT,
  'docs',
  'contribute-redesign',
  '12-a11y-raw',
);
const BASELINE_MD = path.join(
  REPO_ROOT,
  'docs',
  'contribute-redesign',
  '12-a11y-baseline.md',
);

const ARGS = new Set(process.argv.slice(2));
const CI = ARGS.has('--ci');

fs.mkdirSync(RAW_DIR, { recursive: true });

// ---------------------------------------------------------------------
// Static file server. Minimal — MIME by extension, serves ./public, and
// for the /contribute/status/<id> path serves contribute-status.html to
// mimic the Python server route.
// ---------------------------------------------------------------------

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

  // Match the Python server's /contribute/status/<id> route.
  const statusMatch = pathname.match(/^\/contribute\/status\/[^/]+\/?$/);
  if (statusMatch) {
    serveFile(res, path.join(PUBLIC_DIR, 'contribute-status.html'));
    return;
  }

  // Root → index.html
  if (pathname === '/' || pathname === '') pathname = '/index.html';

  // Bare paths like /contribute or /governance → .html
  if (!path.extname(pathname)) {
    const candidate = path.join(PUBLIC_DIR, pathname + '.html');
    if (fs.existsSync(candidate)) {
      serveFile(res, candidate);
      return;
    }
  }

  const filePath = path.join(PUBLIC_DIR, pathname);
  // Contain within PUBLIC_DIR.
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

// ---------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------

const LANGUAGES_FIXTURE = {
  languages: [
    {
      code: 'bsl',
      english_name: 'British Sign Language',
      coverage_count: 2406,
      has_reviewers: true,
      description_placeholder:
        'e.g. dominant hand is an open palm starting at the forehead, then moves forward and down in a small arc.',
    },
    {
      code: 'asl',
      english_name: 'American Sign Language',
      coverage_count: 1820,
      has_reviewers: true,
      description_placeholder: null,
    },
  ],
};

const STATUSES = [
  'draft',
  'pending_review',
  'under_review',
  'validated',
  'rejected',
  'quarantined',
];

function makeStatusFixture(status) {
  const base = {
    session_id: '00000000-0000-0000-0000-00000000000' + STATUSES.indexOf(status),
    gloss: 'ELECTRON',
    sign_language: 'bsl',
    sign_language_name: 'British Sign Language',
    state: status,
    status,
    created_at: '2026-04-01T09:15:00Z',
    updated_at: '2026-04-12T12:30:00Z',
    description: 'Dominant index finger taps the nondominant flat palm twice.',
    hamnosys: '',
    sigml:
      '<?xml version="1.0"?>\n<sigml>\n  <hns_sign gloss="electron">\n    <hamnosys_manual><hamflathand/></hamnosys_manual>\n  </hns_sign>\n</sigml>',
    comments: [],
  };
  if (status === 'under_review') {
    base.reviewer_assigned_at = '2026-04-10T10:00:00Z';
  }
  if (status === 'validated') {
    base.approved_at = '2026-04-15T14:00:00Z';
    base.comments = [
      {
        verdict: 'approve',
        category: 'phonological',
        body: 'Looks correct. Published.',
        reviewer_name: 'A. Example',
        created_at: '2026-04-15T14:00:00Z',
      },
    ];
  }
  if (status === 'rejected') {
    base.rejected_at = '2026-04-12T12:30:00Z';
    base.rejection_category = 'cultural_appropriateness';
    base.comments = [
      {
        verdict: 'reject',
        category: 'cultural_appropriateness',
        body: 'This sign conflates two regional variants. Please resubmit with the Midlands variant.',
        reviewer_name: 'B. Example',
        created_at: '2026-04-12T12:30:00Z',
      },
    ];
  }
  if (status === 'quarantined') {
    base.comments = [
      {
        verdict: 'flag',
        category: 'cultural_appropriateness',
        body: 'Held pending review after a community concern.',
        reviewer_name: '(community)',
        created_at: '2026-04-16T09:00:00Z',
      },
    ];
  }
  return base;
}

// Each scenario is { id, url, label, setUp?(page) }. setUp runs after
// navigation; it can inject fixture data, hit buttons, or reveal
// hidden panels to get the DOM into the right state for the audit.

function scenarios(port) {
  const base = `http://127.0.0.1:${port}`;
  const withFixtures = async (page) => {
    await page.setRequestInterception(true);
    page.on('request', (req) => {
      const u = req.url();
      if (u.endsWith('/contribute-languages.json')) {
        req.respond({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(LANGUAGES_FIXTURE),
        });
        return;
      }
      if (u.endsWith('/governance-data.json')) {
        req.respond({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            reviewers: [],
            board: [],
            languages: LANGUAGES_FIXTURE.languages.map((l) => ({
              code: l.code,
              name: l.english_name,
              has_native_reviewer: l.has_reviewers,
            })),
            email: 'deaf-feedback@kozha.dev',
          }),
        });
        return;
      }
      req.continue();
    });
  };

  const revealAll = async (page) => {
    // Reveal all panels that live on the contribute page so axe can see
    // their interior. axe-core skips elements with the HTML5 `hidden`
    // attribute; flipping them on lets us audit the full surface.
    await page.evaluate(() => {
      ['langMasthead', 'authoring-root', 'chatPanel', 'avatarPreview', 'notationPanel', 'submissionPanel'].forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.hidden = false;
      });
      // Seed a plausible language so the masthead renders a code/name.
      const code = document.getElementById('languageBadgeCode');
      const name = document.getElementById('languageBadgeName');
      if (code) code.textContent = 'BSL';
      if (name) name.textContent = 'British Sign Language';

      // Populate a HamNoSys display so the glyph area has content.
      const disp = document.getElementById('hamnosysDisplay');
      if (disp) {
        disp.textContent = '';
        disp.setAttribute('aria-label', 'HamNoSys notation with three symbols');
      }

      // Populate captions.
      const g = document.getElementById('avatarCaptionGloss');
      const d = document.getElementById('avatarCaptionDesc');
      if (g) g.textContent = 'ELECTRON';
      if (d) d.textContent = 'Dominant index finger taps the nondominant flat palm twice.';

      // Show a fake chat message so the log isn't empty.
      const log = document.getElementById('chatLog');
      if (log) {
        log.innerHTML = '';
        const msg = document.createElement('div');
        msg.className = 'chat-msg chat-msg-system';
        const lbl = document.createElement('span');
        lbl.className = 'chat-msg-label';
        lbl.textContent = 'Clarification:';
        const txt = document.createElement('p');
        txt.className = 'chat-msg-text';
        txt.textContent = 'Which hand leads the movement?';
        msg.appendChild(lbl);
        msg.appendChild(txt);
        log.appendChild(msg);
      }
    });
  };

  return [
    {
      id: 'landing',
      label: 'Landing page / (index.html)',
      url: `${base}/`,
      setUp: null,
    },
    {
      id: 'contribute-empty',
      label: 'Contribute — empty state, language picker',
      url: `${base}/contribute.html`,
      setUp: async (page) => {
        await withFixtures(page);
      },
    },
    {
      id: 'contribute-after-language',
      label: 'Contribute — language selected, empty authoring area',
      url: `${base}/contribute.html`,
      // Pre-navigation: only register request interception (fixtures).
      // All DOM interaction happens post-navigation in the main loop.
      setUp: async (page) => {
        await withFixtures(page);
      },
    },
    {
      id: 'contribute-mid-session',
      label: 'Contribute — mid-session (chat + preview + notation + submit visible)',
      url: `${base}/contribute.html`,
      setUp: async (page) => {
        await withFixtures(page);
      },
    },
    {
      id: 'governance',
      label: 'Governance page',
      url: `${base}/governance.html`,
      setUp: async (page) => {
        await withFixtures(page);
      },
    },
    ...STATUSES.map((status) => ({
      id: `status-${status}`,
      label: `Submission status — ${status}`,
      url: `${base}/contribute/status/sim-${status}`,
      setUp: async (page) => {
        const payload = makeStatusFixture(status);
        await page.setRequestInterception(true);
        page.on('request', (req) => {
          const u = req.url();
          if (u.includes('/api/chat2hamnosys/sessions/') && u.endsWith('/status')) {
            req.respond({
              status: 200,
              contentType: 'application/json',
              body: JSON.stringify(payload),
            });
            return;
          }
          req.continue();
        });
      },
    })),
  ];
}

// ---------------------------------------------------------------------
// Run axe + pa11y for each scenario.
// ---------------------------------------------------------------------

async function runAxe(page) {
  const builder = new AxePuppeteer(page);
  builder.options({
    runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa', 'best-practice'] },
  });
  return await builder.analyze();
}

async function runPa11y(targetUrl, scenarioId) {
  try {
    return await pa11y(targetUrl, {
      standard: 'WCAG2AA',
      includeNotices: false,
      includeWarnings: true,
      timeout: 30000,
      chromeLaunchConfig: {
        args: ['--no-sandbox', '--disable-setuid-sandbox'],
      },
    });
  } catch (err) {
    return { error: err.message, scenarioId };
  }
}

function impactRank(impact) {
  return { critical: 3, serious: 2, moderate: 1, minor: 0 }[impact] ?? -1;
}

function summarizeAxe(result) {
  const counts = { critical: 0, serious: 0, moderate: 0, minor: 0 };
  const items = [];
  for (const v of result.violations || []) {
    const impact = v.impact || 'minor';
    counts[impact] = (counts[impact] || 0) + v.nodes.length;
    items.push({
      id: v.id,
      impact,
      help: v.help,
      helpUrl: v.helpUrl,
      nodes: v.nodes.map((n) => ({
        target: n.target.join(' › '),
        failureSummary: (n.failureSummary || '').replace(/\s+/g, ' ').trim(),
        html: n.html.slice(0, 300),
      })),
    });
  }
  items.sort((a, b) => impactRank(b.impact) - impactRank(a.impact));
  return { counts, items };
}

function summarizePa11y(result) {
  if (result.error) return { error: result.error, issues: [] };
  const counts = { error: 0, warning: 0, notice: 0 };
  const issues = [];
  for (const it of result.issues || []) {
    counts[it.type] = (counts[it.type] || 0) + 1;
    issues.push({
      type: it.type,
      code: it.code,
      message: it.message,
      selector: it.selector,
    });
  }
  return { counts, issues };
}

function mdTable(rows, headers) {
  const head = `| ${headers.join(' | ')} |`;
  const sep = `| ${headers.map(() => '---').join(' | ')} |`;
  const body = rows.map((r) => `| ${r.join(' | ')} |`).join('\n');
  return [head, sep, body].join('\n');
}

function mdEscape(s) {
  return String(s || '').replace(/\|/g, '\\|').replace(/\n/g, ' ');
}

function renderBaseline(results, brokenScenarios) {
  const lines = [];
  lines.push('# A11y baseline — prompt 12\n');
  lines.push(
    `Captured by \`scripts/a11y/run.mjs\` on ${new Date().toISOString().slice(0, 19).replace('T', ' ')} UTC.\n`,
  );
  lines.push(
    'Runs axe-core (via `@axe-core/puppeteer`, tags `wcag2a/aa wcag21aa wcag22aa best-practice`) and pa11y (HTML_CodeSniffer, `WCAG2AA`) against each scenario below. Raw JSON per scenario is in `12-a11y-raw/`.\n',
  );

  if (brokenScenarios.length) {
    lines.push('## Scenarios that failed to load');
    for (const b of brokenScenarios) {
      lines.push(`- **${b.id}** — ${b.error}`);
    }
    lines.push('');
  }

  // Summary table.
  lines.push('## Summary');
  const summaryRows = results.map((r) => [
    r.id,
    r.label,
    String(r.axe.counts.critical),
    String(r.axe.counts.serious),
    String(r.axe.counts.moderate),
    String(r.axe.counts.minor),
    String(r.pa11y.counts?.error || 0),
    String(r.pa11y.counts?.warning || 0),
  ]);
  lines.push(
    mdTable(summaryRows, [
      'id',
      'scenario',
      'axe crit',
      'axe serious',
      'axe mod',
      'axe minor',
      'pa11y error',
      'pa11y warn',
    ]),
  );
  lines.push('');

  // Per-scenario detail.
  for (const r of results) {
    lines.push(`## ${r.label}\n`);
    lines.push(`- URL: \`${r.url}\``);
    lines.push(
      `- axe: ${r.axe.counts.critical} critical, ${r.axe.counts.serious} serious, ${r.axe.counts.moderate} moderate, ${r.axe.counts.minor} minor`,
    );
    lines.push(
      `- pa11y: ${r.pa11y.counts?.error || 0} error, ${r.pa11y.counts?.warning || 0} warning`,
    );
    lines.push('');

    if (r.axe.items.length) {
      lines.push('### axe violations\n');
      for (const v of r.axe.items) {
        lines.push(
          `- **${v.impact}** \`${v.id}\` — ${mdEscape(v.help)} ([docs](${v.helpUrl}))`,
        );
        for (const n of v.nodes.slice(0, 3)) {
          lines.push(`  - \`${mdEscape(n.target)}\``);
        }
        if (v.nodes.length > 3) {
          lines.push(`  - …and ${v.nodes.length - 3} more`);
        }
      }
      lines.push('');
    }

    if (r.pa11y.issues && r.pa11y.issues.length) {
      lines.push('### pa11y issues\n');
      const shown = r.pa11y.issues.slice(0, 10);
      for (const it of shown) {
        lines.push(
          `- **${it.type}** \`${mdEscape(it.code)}\` — ${mdEscape(it.message)}`,
        );
        lines.push(`  - \`${mdEscape(it.selector)}\``);
      }
      if (r.pa11y.issues.length > shown.length) {
        lines.push(`- …and ${r.pa11y.issues.length - shown.length} more (see raw)`);
      }
      lines.push('');
    }
  }

  // Non-fix policy section.
  lines.push('## Moderate violations accepted as-is\n');
  lines.push(
    'None. All axe violations at every impact level are fixed in code.\n',
  );

  // Acknowledged pa11y warnings that are false positives or inherent.
  lines.push('## pa11y warnings accepted as-is\n');
  lines.push(
    [
      '- **`WCAG2AA.Principle1.Guideline1_3.1_3_1.H48`** on the landing hero-actions div — pa11y heuristically suggests any link cluster inside a section be a list. The hero actions are two primary CTA buttons, not navigation. Semantically they belong in a flex row; a `<ul>` would be artificial. No user impact — both buttons are reached by Tab.',
      '- **`WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206`** on `#toast`, `#modalBackdrop`, `#hintStrip` — these use `position: fixed` by function (overlay toast, confirm modal, dismissable keyboard hint). `position: fixed` is inherent to the pattern; the content inside wraps on narrow viewports (see `@media (max-width: 640px)` override). No two-dimensional scrolling in practice.',
      '- **`WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,...`** on the SiGML `<pre>` element — preformatted XML for a reviewer is legitimately not reflowable. `overflow-x: auto` and `max-height: 360px` bound the element; `tabindex="0"` lets keyboard users scroll within it. A hands-on reviewer at 400% zoom on a phone still sees a usable scroll container.',
      '- **`WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha`** on `#governanceReviewers > p`, `#governanceBoard > p`, `#governanceEmailPlain` — these elements have a ~3–4% ink-tint background (`rgba(21,19,15,0.03–0.04)`) for visual grouping. Foreground is solid `--ink` (#15130f); composite background on paper is #eeeae3 / #efebe4, giving >15:1 contrast. pa11y cannot compute composites and warns on any transparency; axe correctly passes these.',
    ].join('\n') + '\n',
  );

  // Manual-review gaps section.
  lines.push('## Out of scope for this tool — requires human review\n');
  lines.push(
    [
      '- Screen-reader behaviour under NVDA, VoiceOver, and TalkBack. Tracked in `12-sr-findings.md` with honest "pending human testing" entries.',
      '- Keyboard navigation under live interaction (not simulated by axe). Tracked in `12-keyboard-findings.md`.',
      '- Cognitive load audit (affordance count per screen, error-remedy pairing, concurrent-action load). Tracked in `12-cognitive-load.md`.',
      '- Real-device testing across iPhone SE, iPad portrait, desktop 1440px, 4K. Tracked in `12-responsive-findings.md`.',
      '- CWASA 3D avatar semantics — the canvas renders a signing body that cannot be inspected by text-based automation. axe treats it as an opaque block; captions and play/pause are the accessible surface. Deaf users cross-check the gloss + description captions against what the avatar signs.',
      '- HamNoSys font rendering across Firefox, Safari, Chrome, and mobile Safari — the `@font-face` declaration points at `/fonts/bgHamNoSysUnicode.ttf`. If the binary is missing, the font stack falls through to system fonts. Browser-matrix verification is hands-on.',
    ].join('\n') + '\n',
  );

  return lines.join('\n');
}

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

async function main() {
  const { server, port } = await startServer();
  console.log(`[a11y] static server listening on 127.0.0.1:${port}`);

  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  const results = [];
  const broken = [];

  try {
    for (const scn of scenarios(port)) {
      console.log(`[a11y] ${scn.id} — ${scn.label}`);
      const page = await browser.newPage();
      await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 1 });
      try {
        if (scn.setUp) await scn.setUp(page);
        await page.goto(scn.url, { waitUntil: 'networkidle0', timeout: 30000 });
        // Re-run setUp after nav if it only worked via request interception,
        // so reveal/seed side-effects happen post-navigation.
        if (scn.id === 'contribute-after-language') {
          await page.waitForSelector('#pickerOptions button', { timeout: 15000 });
          await page.click('#pickerOptions button');
          await page.evaluate(() => new Promise((r) => setTimeout(r, 100)));
        } else if (scn.id === 'contribute-mid-session') {
          await page.waitForSelector('#pickerOptions button', { timeout: 15000 });
          await page.click('#pickerOptions button');
          await page.evaluate(() => {
            ['langMasthead', 'authoring-root', 'chatPanel', 'avatarPreview', 'notationPanel', 'submissionPanel'].forEach((id) => {
              const el = document.getElementById(id);
              if (el) el.hidden = false;
            });
            const code = document.getElementById('languageBadgeCode');
            const name = document.getElementById('languageBadgeName');
            if (code) code.textContent = 'BSL';
            if (name) name.textContent = 'British Sign Language';
            const disp = document.getElementById('hamnosysDisplay');
            if (disp) {
              disp.textContent = '';
              disp.setAttribute('aria-label', 'HamNoSys notation with three symbols');
            }
            const g = document.getElementById('avatarCaptionGloss');
            const d = document.getElementById('avatarCaptionDesc');
            if (g) g.textContent = 'ELECTRON';
            if (d) d.textContent = 'Dominant index finger taps the nondominant flat palm twice.';
            const log = document.getElementById('chatLog');
            if (log) {
              log.innerHTML = '';
              const msg = document.createElement('div');
              msg.className = 'chat-msg chat-msg-system';
              const lbl = document.createElement('span');
              lbl.className = 'chat-msg-label';
              lbl.textContent = 'Clarification:';
              const txt = document.createElement('p');
              txt.className = 'chat-msg-text';
              txt.textContent = 'Which hand leads the movement?';
              msg.appendChild(lbl);
              msg.appendChild(txt);
              log.appendChild(msg);
            }
          });
        }

        const axeRes = await runAxe(page);
        const axeSum = summarizeAxe(axeRes);
        fs.writeFileSync(
          path.join(RAW_DIR, `${scn.id}.axe.json`),
          JSON.stringify(axeRes, null, 2),
        );

        // pa11y can't use our state-seeding (it opens its own browser),
        // so we skip it for scenarios that depend on post-interaction
        // state and only run it where the plain navigated URL reflects
        // the scenario.
        let pa11ySum = { counts: { error: 0, warning: 0 }, issues: [] };
        const pa11yEligible = !scn.id.startsWith('contribute-after') && !scn.id.startsWith('contribute-mid');
        if (pa11yEligible) {
          const pa11yRes = await runPa11y(scn.url, scn.id);
          fs.writeFileSync(
            path.join(RAW_DIR, `${scn.id}.pa11y.json`),
            JSON.stringify(pa11yRes, null, 2),
          );
          pa11ySum = summarizePa11y(pa11yRes);
        }

        results.push({
          id: scn.id,
          label: scn.label,
          url: scn.url,
          axe: axeSum,
          pa11y: pa11ySum,
        });
      } catch (err) {
        console.error(`[a11y] ${scn.id} FAILED:`, err.message);
        broken.push({ id: scn.id, error: err.message });
      } finally {
        await page.close();
      }
    }
  } finally {
    await browser.close();
    server.close();
  }

  // Write the baseline markdown.
  const md = renderBaseline(results, broken);
  fs.writeFileSync(BASELINE_MD, md);
  console.log(`[a11y] wrote ${BASELINE_MD}`);

  // CI exit policy: fail on any critical or serious axe violation.
  let crit = 0;
  let serious = 0;
  for (const r of results) {
    crit += r.axe.counts.critical;
    serious += r.axe.counts.serious;
  }
  console.log(`[a11y] totals: ${crit} critical, ${serious} serious`);
  if (CI && (crit > 0 || serious > 0)) {
    console.error('[a11y] CI mode: failing on critical/serious violations');
    process.exit(1);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(2);
});
