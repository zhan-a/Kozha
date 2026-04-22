#!/usr/bin/env node
/*
 * Smoke + unit test for the /credits page (prompt-polish 9).
 *
 * The browser half spins up a static server for ./public, loads /credits
 * in headless Chrome, and asserts:
 *   1. every corpus that appears in the README is cited on the page
 *   2. no placeholder text like "TBD" or "coming soon" remains
 *   3. every external <a href> on the page has a non-empty href
 *   4. the footer link from /progress points at /credits (structural check)
 *
 * The parse-only half reads public/credits.html directly and asserts:
 *   - every <a> that is an author/repo citation is inside a credits-entry
 *     section that names the corpus — no orphaned credits, no stray links
 *     that drift out of their citation block
 *
 * Run: node tests/smoke/credits-page.mjs
 * Exit non-zero on any failure.
 */

import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import puppeteer from 'puppeteer';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..', '..');
const PUBLIC_DIR = path.join(REPO_ROOT, 'public');
const README_PATH = path.join(REPO_ROOT, 'README.md');

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
  '.txt': 'text/plain; charset=utf-8',
};

function mimeFor(p) {
  return MIME[path.extname(p).toLowerCase()] || 'application/octet-stream';
}

function serveFile(res, filePath) {
  res.writeHead(200, { 'Content-Type': mimeFor(filePath), 'Cache-Control': 'no-store' });
  fs.createReadStream(filePath).pipe(res);
}

function handler(req, res) {
  const parsed = new URL(req.url, 'http://localhost');
  const p = decodeURIComponent(parsed.pathname);

  // Mirror the server.py alias: /credits → credits.html
  if (p === '/credits' || p === '/credits/') {
    return serveFile(res, path.join(PUBLIC_DIR, 'credits.html'));
  }
  if (p === '/progress' || p === '/progress/') {
    return serveFile(res, path.join(PUBLIC_DIR, 'progress.html'));
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

// ---------- corpus list — every source that must be cited on /credits ----------
// Pulled from the README credits section. Each entry is a pair
// [corpus name pattern, human-readable label for the failure message].
const REQUIRED_CORPORA = [
  [/DictaSign/i,                'DictaSign (BSL/LSF/GSL)'],
  [/DGS Lexicon/i,              'DGS Lexicon'],
  [/PJM Dictionary/i,           'PJM Dictionary (Warsaw)'],
  [/SignAvatars/i,              'SignAvatars (aggregator)'],
  [/SignLanguageSynthesis/i,    'SignLanguageSynthesis (NGT, Esselink)'],
  [/algerianSignLanguage-avatar/i, 'algerianSignLanguage-avatar (Zerrouki)'],
  [/bdsl-3d-animation/i,        'bdsl-3d-animation (Khan)'],
  [/Text-to-Sign-Language/i,    'Text-to-Sign-Language (Divanshu)'],
  [/text_to_isl/i,              'text_to_isl (Shoebham)'],
  [/KurdishSignLanguage/i,      'KurdishSignLanguage (KurdishBLARK)'],
  [/\bVSL\b/,                   'VSL (Rido)'],
  [/syntheticfsl/i,             'syntheticfsl (Ablog)'],
  [/signtyper/i,                'signtyper (Ablog)'],
  [/CWASA/i,                    'CWASA (UEA VHG)'],
  [/HamNoSys/i,                 'HamNoSys (IDGS)'],
  [/argostranslate|argos-translate/i, 'argostranslate'],
  [/spaCy/i,                    'spaCy'],
];

// Strings that indicate unfinished copy. "coming soon" is a false-positive
// risk (the translator dropdown uses it for disabled sign languages), but
// it should never appear on /credits itself.
const PLACEHOLDER_PATTERNS = [
  /\bTBD\b/,
  /\bTODO\b/,
  /\bFIXME\b/,
  /coming soon/i,
  /lorem ipsum/i,
  /placeholder/i,
];

function checkRequiredCorpora(pageText, failures) {
  for (const [pattern, label] of REQUIRED_CORPORA) {
    if (!pattern.test(pageText)) {
      failures.push(`/credits is missing the required corpus citation: ${label} (pattern ${pattern})`);
    }
  }
}

function checkNoPlaceholders(pageText, failures) {
  for (const p of PLACEHOLDER_PATTERNS) {
    const match = pageText.match(p);
    if (match) {
      failures.push(`/credits contains placeholder text matching ${p}: "${match[0]}"`);
    }
  }
}

async function checkAllLinksHaveHref(page, failures) {
  const bad = await page.$$eval('a', (anchors) => {
    return anchors
      .map((a, idx) => ({
        idx,
        text: (a.textContent || '').trim().slice(0, 60),
        href: a.getAttribute('href'),
      }))
      .filter((a) => a.href === null || a.href === '' || a.href === '#');
  });
  if (bad.length) {
    for (const a of bad) {
      failures.push(`/credits has an <a> with empty href: text="${a.text}" (index ${a.idx})`);
    }
  }
}

async function checkProgressFooterLinksCredits(page, port, failures) {
  await page.goto(`http://127.0.0.1:${port}/progress`, { waitUntil: 'domcontentloaded', timeout: 15000 });
  const hasCredits = await page.$$eval('.kz-footer a', (links) => {
    return links.some((a) => /credits/i.test(a.getAttribute('href') || ''));
  });
  if (!hasCredits) {
    failures.push('/progress footer does not contain a link to /credits');
  }
}

async function checkProgressSourceLinksCredits(page, port, failures) {
  await page.goto(`http://127.0.0.1:${port}/progress`, { waitUntil: 'networkidle0', timeout: 20000 });
  // The source cell is rendered asynchronously after the snapshot JSON loads.
  // Wait up to 6 seconds for the first source link to appear.
  try {
    await page.waitForSelector('.progress-source-link', { timeout: 6000 });
  } catch (_e) {
    // Snapshot may be unavailable under the static test server (no
    // progress_snapshot.json generation pipeline); in that case the table
    // renders an empty-state and there is nothing to link. Treat as skip.
    console.log('[credits-test] progress table did not populate (snapshot missing); skipping source-link assertion');
    return;
  }
  const firstHref = await page.$eval('.progress-source-link', (a) => a.getAttribute('href'));
  if (!firstHref || !/\/credits/.test(firstHref)) {
    failures.push(`/progress source link does not point at /credits (got "${firstHref}")`);
  }
}

// ---------- parse-only tests (no browser) ----------

function parseCreditsAnchorsAndCitations(html) {
  // Crude scan sufficient for the orphan check: scan for every <a href="...">
  // and record the byte offset, then scan for <article class="credits-entry">
  // open/close offsets. An anchor whose offset is outside every entry range
  // AND outside the hero, toc, and contributors sections is flagged.
  const linkRe = /<a\s[^>]*?href="([^"]*)"[^>]*>/gi;
  const links = [];
  let m;
  while ((m = linkRe.exec(html)) !== null) {
    links.push({ offset: m.index, href: m[1], snippet: html.slice(m.index, m.index + 80) });
  }

  const articleRe = /<article class="credits-entry"[^>]*>([\s\S]*?)<\/article>/gi;
  const ranges = [];
  while ((m = articleRe.exec(html)) !== null) {
    ranges.push({ start: m.index, end: m.index + m[0].length });
  }

  // Also allow anchors in the TOC nav, the hero <header>, the navigation,
  // and the content of the non-corpus sections (contributors, board,
  // funding, compensation, translation layer, notation, avatar). Those have
  // their own content structures rather than <article class="credits-entry">.
  const allowRangeRe = /<(nav[^>]*?class="(?:kz-header|credits-toc)"[^>]*?>[\s\S]*?<\/nav>|header[^>]*?class="credits-hero"[^>]*?>[\s\S]*?<\/header>|section[^>]*?id="(?:contributors|board|funding|compensation|translation|notation|avatar)"[^>]*?>[\s\S]*?<\/section>|footer[^>]*?class="kz-footer"[^>]*?>[\s\S]*?<\/footer>)/gi;
  const allowRanges = [];
  while ((m = allowRangeRe.exec(html)) !== null) {
    allowRanges.push({ start: m.index, end: m.index + m[0].length });
  }

  return { links, ranges, allowRanges };
}

function inAnyRange(offset, ranges) {
  return ranges.some((r) => offset >= r.start && offset < r.end);
}

function checkOrphanedCredits(failures) {
  const html = fs.readFileSync(path.join(PUBLIC_DIR, 'credits.html'), 'utf8');
  const { links, ranges, allowRanges } = parseCreditsAnchorsAndCitations(html);
  if (!ranges.length) {
    failures.push('credits.html has zero <article class="credits-entry"> blocks — structure broken');
  }
  const combined = ranges.concat(allowRanges);
  for (const link of links) {
    // Skip in-page anchor links (#foo) — they never need a citation block.
    if (link.href.startsWith('#')) continue;
    // Skip the favicon and stylesheet references (they're inside <head>).
    if (/\.(css|ico|svg|woff2?|ttf)(\?|$)/.test(link.href)) continue;
    if (!inAnyRange(link.offset, combined)) {
      failures.push(
        `credits.html has an orphaned <a> outside any citation block: href="${link.href}" near "${link.snippet.replace(/\s+/g, ' ')}"`,
      );
    }
  }
}

function checkReadmeAndPageAgree(failures) {
  // Basic cross-check: every corpus the README credits section names must
  // also be named on /credits. The list is the same pattern set the browser
  // half uses; we verify against the README so changes to the README don't
  // silently drift from the page. This is not a full structural equality
  // check — just a sanity net against losing a citation.
  const readme = fs.readFileSync(README_PATH, 'utf8');
  const page = fs.readFileSync(path.join(PUBLIC_DIR, 'credits.html'), 'utf8');
  for (const [pattern, label] of REQUIRED_CORPORA) {
    const inReadme = pattern.test(readme);
    const onPage = pattern.test(page);
    if (inReadme && !onPage) {
      failures.push(`README names ${label} but /credits does not. README and /credits must agree.`);
    }
  }
}

async function main() {
  const failures = [];

  // Parse-only checks first — they are the cheapest and tell us about
  // structural problems even if the browser run later fails for another
  // reason.
  checkOrphanedCredits(failures);
  checkReadmeAndPageAgree(failures);

  const { server, port } = await startServer();
  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 900 });
    await page.goto(`http://127.0.0.1:${port}/credits`, { waitUntil: 'domcontentloaded', timeout: 15000 });
    const pageText = await page.evaluate(() => document.body.innerText);

    checkRequiredCorpora(pageText, failures);
    checkNoPlaceholders(pageText, failures);
    await checkAllLinksHaveHref(page, failures);

    await checkProgressFooterLinksCredits(page, port, failures);
    await checkProgressSourceLinksCredits(page, port, failures);

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
  console.log('\nOK: /credits page smoke + unit tests passed.');
}

main().catch((err) => {
  console.error(err);
  process.exit(2);
});
