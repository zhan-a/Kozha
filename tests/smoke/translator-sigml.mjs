#!/usr/bin/env node
/*
 * Smoke test for the `[object Object]` HamNoSys regression (prompt 3 fix).
 *
 * Spins up a static server for ./public, loads /app.html in headless Chrome,
 * waits for CWASA + LSF database to load, then exercises buildSigml(['fruit'])
 * in-page and asserts:
 *   1. the composed SiGML string does NOT contain '[object Object]'
 *   2. the validator flagged at least one malformed sign (FRUIT uses
 *      <hampalmud/> which CWASA cannot parse)
 *   3. no console error matches /mismatched input|object Object/
 *
 * Also takes a screenshot per language pair and drops it in
 * tests/visual/translation-sanity/ (not a pixel-diff — captured for
 * eyeballing in the PR).
 *
 * Run:
 *   node tests/smoke/translator-sigml.mjs
 *
 * Exit non-zero on any assertion failure.
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
const SANITY_DIR = path.join(REPO_ROOT, 'tests', 'visual', 'translation-sanity');

fs.mkdirSync(SANITY_DIR, { recursive: true });

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
  '.png': 'image/png',
  '.wasm': 'application/wasm',
};

function mimeFor(p) {
  return MIME[path.extname(p).toLowerCase()] || 'application/octet-stream';
}

function serveFile(res, filePath) {
  res.writeHead(200, { 'Content-Type': mimeFor(filePath), 'Cache-Control': 'no-store' });
  fs.createReadStream(filePath).pipe(res);
}

function stubJson(res, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(body);
}

function handler(req, res) {
  const parsed = new URL(req.url, 'http://localhost');
  const p = decodeURIComponent(parsed.pathname);

  if (p === '/api/translate-text' && req.method === 'POST') {
    let body = '';
    req.on('data', (c) => { body += c; });
    req.on('end', () => {
      try {
        const { text } = JSON.parse(body || '{}');
        stubJson(res, { translated: text });
      } catch {
        stubJson(res, { translated: '' });
      }
    });
    return;
  }
  if (p === '/api/plan' && req.method === 'POST') {
    let body = '';
    req.on('data', (c) => { body += c; });
    req.on('end', () => {
      try {
        const { text } = JSON.parse(body || '{}');
        stubJson(res, { final: (text || '').trim() + ' .', raw: text, language: 'en', allowed: [] });
      } catch {
        stubJson(res, { final: '.', raw: '', language: 'en', allowed: [] });
      }
    });
    return;
  }
  if (p.startsWith('/data/')) {
    const fp = path.join(DATA_DIR, p.slice('/data/'.length));
    if (!fp.startsWith(DATA_DIR) || !fs.existsSync(fp)) { res.writeHead(404).end('not found'); return; }
    return serveFile(res, fp);
  }
  const fp = path.join(PUBLIC_DIR, p === '/' ? '/index.html' : p);
  if (!fp.startsWith(PUBLIC_DIR) || !fs.existsSync(fp) || fs.statSync(fp).isDirectory()) {
    res.writeHead(404).end('not found');
    return;
  }
  serveFile(res, fp);
}

async function startServer() {
  return new Promise((resolve, reject) => {
    const server = http.createServer(handler);
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port }));
  });
}

async function waitForReady(page) {
  await page.waitForFunction(
    () => !!window.SigmlValidator && typeof window.SigmlValidator.validateHnsSignXml === 'function',
    { timeout: 15000 },
  );
  await page.waitForFunction(
    () => !!window.CWASA && typeof window.getCWAEnv === 'function',
    { timeout: 30000 },
  );
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
    // switchSignLanguage is awaited by the change handler; wait for glossToSign
    await new Promise((resolve) => {
      const check = () => {
        if (typeof glossToSign !== 'undefined' && glossToSign && (glossToSign.size > 0 || letterToSign?.size > 0)) resolve();
        else setTimeout(check, 50);
      };
      check();
    });
  }, lang);
}

async function exerciseFruit(page, lang) {
  return await page.evaluate((langCode) => {
    const t = 'fruit';
    const gloss = (typeof glossToSign !== 'undefined' && glossToSign.has(t))
      ? t
      : (typeof conceptToGloss !== 'undefined' && conceptToGloss.has(t))
        ? conceptToGloss.get(t)
        : t;
    const tokens = [gloss];
    const sigml = (typeof buildSigml === 'function') ? buildSigml(tokens) : null;
    const malformed = window.__lastMalformedSigns || window.__heroLastMalformedSigns || [];
    return {
      lang: langCode,
      gloss,
      glossHit: typeof glossToSign !== 'undefined' && glossToSign.has(gloss),
      sigmlLen: sigml ? sigml.length : 0,
      sigmlHasObjectLiteral: !!(sigml && sigml.includes('[object Object]')),
      malformedCount: malformed.length,
      malformedFirst: malformed[0] || null,
      glossDbSize: typeof glossToSign !== 'undefined' ? glossToSign.size : 0,
      letterDbSize: typeof letterToSign !== 'undefined' ? letterToSign.size : 0,
    };
  }, lang);
}

const CASES = [
  { lang: 'bsl', expectMalformed: false, expectSigml: true, label: 'English fruit → BSL' },
  { lang: 'lsf', expectMalformed: true,  expectSigml: true, label: 'English fruit → LSF (reported bug)' },
  { lang: 'asl', expectMalformed: false, expectSigml: true, label: 'English fruit → ASL (BSL alias)' },
  { lang: 'dgs', expectMalformed: false, expectSigml: false, label: 'English fruit → DGS (no FRUIT entry)' },
  { lang: 'pjm', expectMalformed: null,  expectSigml: null,  label: 'English fruit → PJM (whatever the DB has)' },
];

async function main() {
  const { server, port } = await startServer();
  const failures = [];
  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });
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
      const res = await exerciseFruit(page, c.lang);
      const shotPath = path.join(SANITY_DIR, `fruit-${c.lang}.png`);
      await page.screenshot({ path: shotPath, fullPage: false });
      const badConsole = consoleErrors.filter((e) => /mismatched input|object Object/i.test(e.text));
      console.log(`[${c.lang}]`, JSON.stringify({ ...res, badConsole: badConsole.length }));
      if (res.sigmlHasObjectLiteral) {
        failures.push(`${c.label}: composed SiGML contains "[object Object]"`);
      }
      if (badConsole.length) {
        failures.push(`${c.label}: console has ${badConsole.length} error(s) matching /mismatched input|object Object/: ${JSON.stringify(badConsole)}`);
      }
      if (c.expectMalformed === true && res.malformedCount === 0) {
        failures.push(`${c.label}: expected validator to flag malformed entry, but got 0`);
      }
      if (c.expectMalformed === false && res.malformedCount > 0) {
        failures.push(`${c.label}: unexpected malformed entry flagged: ${JSON.stringify(res.malformedFirst)}`);
      }
    }
    await page.close();
  } finally {
    await browser.close();
    server.close();
  }

  if (failures.length) {
    console.error('\nFAIL:');
    for (const f of failures) console.error('  - ' + f);
    process.exit(1);
  }
  console.log('\nOK: translator SiGML smoke test passed.');
}

main().catch((err) => {
  console.error(err);
  process.exit(2);
});
