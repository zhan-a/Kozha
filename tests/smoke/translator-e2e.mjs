#!/usr/bin/env node
/*
 * Polish-14 end-to-end translator smoke.
 *
 * Complements tests/smoke/translator-sigml.mjs (which focuses on the
 * `[object Object]` regression) by walking through the prompt-14 §2
 * matrix:
 *
 *   - English → BSL, common word ("hello")
 *   - English → BSL, fingerspell candidate ("pneumonia")
 *   - English → LSF, "fruit" (original reported bug)
 *   - English → ASL, a sentence ("good morning friend")
 *   - English → DGS, basic word ("water")
 *   - English → PJM, basic word ("thank you")
 *
 * For each case we assert:
 *   1. buildSigml returned a non-empty string with no `[object Object]`
 *   2. either glossHit or fingerspellCount > 0 (path is not silent)
 *   3. console has no errors matching /mismatched input|object Object/
 *
 * Running the spacy-driven *cross-lingual* pairs (French → LSF,
 * Spanish → BSL via argos, Polish → PJM) requires the full server
 * with the argostranslate runtime — those are covered by
 * server/tests/test_translation_regression.py (pytest), not here.
 *
 * Exit non-zero on any assertion failure. Writes a summary to
 * docs/polish/14-e2e-smoke.md.
 */

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import puppeteer from 'puppeteer';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..', '..');
const PUBLIC_DIR = path.join(REPO_ROOT, 'public');
const DATA_DIR = path.join(REPO_ROOT, 'data');
const REPORT_PATH = path.join(REPO_ROOT, 'docs', 'polish', '14-e2e-smoke.md');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.sigml': 'application/xml; charset=utf-8',
  '.csv': 'text/csv; charset=utf-8',
  '.ttf': 'font/ttf',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ico': 'image/x-icon',
  '.txt': 'text/plain; charset=utf-8',
};

function serveFile(req, res, root) {
  const url = new URL(req.url, 'http://x');
  let rel = decodeURIComponent(url.pathname);
  if (rel === '/') rel = '/index.html';
  const candidates = [path.join(root, rel)];
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
    } catch (_) {}
  }
  res.writeHead(404, { 'content-type': 'text/plain' });
  res.end('not found');
}

function startServer(root) {
  return new Promise((resolve, reject) => {
    const s = http.createServer((req, res) => serveFile(req, res, root));
    s.on('error', reject);
    s.listen(0, '127.0.0.1', () => resolve(s));
  });
}

async function waitForReady(page) {
  await page.waitForFunction(() => typeof window.getCWAEnv === 'function', { timeout: 30000 });
  await page.waitForFunction(
    () => {
      try {
        const defs = window.getCWAEnv().get('HNSDefs');
        return !!(defs && defs.hamMap && Object.keys(defs.hamMap).length > 100);
      } catch (_) { return false; }
    },
    { timeout: 30000 },
  );
}

async function loadLanguage(page, lang) {
  await page.evaluate(async (code) => {
    const sel = document.getElementById('signLangSelect');
    if (!sel) throw new Error('signLangSelect missing');
    sel.value = code;
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    await new Promise((r) => setTimeout(r, 50));
    await new Promise((resolve) => {
      const check = () => {
        if (typeof glossToSign !== 'undefined' && glossToSign && (glossToSign.size > 0 || letterToSign?.size > 0)) resolve();
        else setTimeout(check, 50);
      };
      check();
    });
  }, lang);
}

async function translate(page, text) {
  return await page.evaluate((raw) => {
    const words = raw.toLowerCase().split(/\s+/).filter(Boolean);
    const tokens = [];
    let fingerspellCount = 0;
    let glossHitCount = 0;
    for (const w of words) {
      if (typeof glossToSign !== 'undefined' && glossToSign.has(w)) {
        tokens.push(w);
        glossHitCount += 1;
        continue;
      }
      if (typeof conceptToGloss !== 'undefined' && conceptToGloss.has(w)) {
        tokens.push(conceptToGloss.get(w));
        glossHitCount += 1;
        continue;
      }
      // Fall through: fingerspell characters. letterToSign uses
      // uppercase keys (A..Z) per the alphabet-DB loader.
      for (const ch of w.toUpperCase()) {
        if (typeof letterToSign !== 'undefined' && letterToSign.has(ch)) {
          tokens.push(ch);
          fingerspellCount += 1;
        }
      }
    }
    const sigml = (typeof buildSigml === 'function') ? buildSigml(tokens) : null;
    const malformed = window.__lastMalformedSigns || window.__heroLastMalformedSigns || [];
    return {
      text: raw,
      tokenCount: tokens.length,
      glossHitCount,
      fingerspellCount,
      sigmlLen: sigml ? sigml.length : 0,
      sigmlHasObjectLiteral: !!(sigml && sigml.includes('[object Object]')),
      malformedCount: malformed.length,
    };
  }, text);
}

const CASES = [
  { label: 'English → BSL, common word',         lang: 'bsl', text: 'hello' },
  { label: 'English → BSL, fingerspell candidate', lang: 'bsl', text: 'pneumonia' },
  { label: 'English → LSF, original bug word',   lang: 'lsf', text: 'fruit' },
  { label: 'English → ASL, short sentence',      lang: 'asl', text: 'good morning friend' },
  { label: 'English → DGS, basic word',          lang: 'dgs', text: 'water' },
  { label: 'English → PJM, basic word',          lang: 'pjm', text: 'thank you' },
];

async function main() {
  const server = await startServer(PUBLIC_DIR);
  const port = server.address().port;
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-dev-shm-usage'],
  });
  const failures = [];
  const results = [];
  try {
    const page = await browser.newPage();
    const consoleErrors = [];
    page.on('console', (msg) => {
      const t = msg.type();
      if (t === 'error' || t === 'warning') {
        consoleErrors.push({ type: t, text: msg.text() });
      }
    });
    page.on('pageerror', (err) => consoleErrors.push({ type: 'pageerror', text: String(err) }));

    await page.setViewport({ width: 1280, height: 800 });
    await page.goto(`http://127.0.0.1:${port}/app.html`, { waitUntil: 'networkidle0', timeout: 30000 });
    await waitForReady(page);

    for (const c of CASES) {
      consoleErrors.length = 0;
      await loadLanguage(page, c.lang);
      const res = await translate(page, c.text);
      const badConsole = consoleErrors.filter((e) => /mismatched input|object Object/i.test(e.text));
      const ok =
        !res.sigmlHasObjectLiteral &&
        res.sigmlLen > 0 &&
        badConsole.length === 0 &&
        (res.glossHitCount + res.fingerspellCount) > 0;
      results.push({ ...c, ...res, badConsole: badConsole.length, ok });
      console.log(`[${c.lang}] "${c.text}" → ${JSON.stringify({ ...res, badConsole: badConsole.length, ok })}`);
      if (res.sigmlHasObjectLiteral) failures.push(`${c.label}: SiGML contains "[object Object]"`);
      if (res.sigmlLen === 0)        failures.push(`${c.label}: buildSigml returned empty`);
      if (badConsole.length)         failures.push(`${c.label}: bad console ${JSON.stringify(badConsole)}`);
      if ((res.glossHitCount + res.fingerspellCount) === 0) {
        failures.push(`${c.label}: path silent — no gloss hits and no fingerspell fallback`);
      }
    }
  } finally {
    await browser.close();
    server.close();
  }

  // Emit a machine-readable + human-readable report.
  const now = new Date().toISOString().replace('T', ' ').slice(0, 19) + ' UTC';
  const lines = [
    '# Polish 14 — End-to-end translator smoke',
    '',
    `Run at: ${now}`,
    'Harness: headless Chromium via puppeteer against a local static server for `public/`.',
    '',
    '| Case | Lang | Input | Tokens | Gloss hits | Fingerspell | SiGML bytes | Console err | OK |',
    '| ---- | ---- | ----- | ------ | ---------- | ----------- | ----------- | ----------- | -- |',
  ];
  for (const r of results) {
    lines.push(
      `| ${r.label} | ${r.lang} | \`${r.text}\` | ${r.tokenCount} | ${r.glossHitCount} | ${r.fingerspellCount} | ${r.sigmlLen} | ${r.badConsole} | ${r.ok ? 'yes' : 'NO'} |`
    );
  }
  lines.push('');
  lines.push('## Cross-lingual pairs (spacy + argostranslate)');
  lines.push('');
  lines.push('The following cases need the full Python server (argostranslate models live there, not in the static-only smoke server):');
  lines.push('');
  lines.push('- French → LSF, sentence — covered by `server/tests/test_translation_regression.py::test_plan_returns_string_final[fruit-fr-lsf]`');
  lines.push('- Spanish → BSL, fallback — covered by the same parametrised suite on `(es, en, *)` sources (argos es→en, then en→bsl gloss)');
  lines.push('- Polish → PJM — covered by `test_translate_text_returns_string[en-pl-fruit]` (Polish translation path)');
  lines.push('');
  if (failures.length) {
    lines.push('## Failures');
    lines.push('');
    for (const f of failures) lines.push(`- ${f}`);
    lines.push('');
  } else {
    lines.push('## Result');
    lines.push('');
    lines.push('All cases asserted:');
    lines.push('- buildSigml emitted non-empty bytes with no `[object Object]`');
    lines.push('- path was not silent (at least one gloss hit or fingerspell letter)');
    lines.push('- console had no errors matching `/mismatched input|object Object/`');
    lines.push('');
    lines.push('**OK** — smoke passes.');
    lines.push('');
  }
  fs.writeFileSync(REPORT_PATH, lines.join('\n'));
  console.log(`\nwrote ${REPORT_PATH}`);

  if (failures.length) {
    console.error('\nFAIL:');
    for (const f of failures) console.error('  - ' + f);
    process.exit(1);
  }
  console.log('\nOK: polish-14 e2e smoke passed.');
}

main().catch((err) => {
  console.error(err);
  process.exit(2);
});
