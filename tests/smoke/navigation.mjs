#!/usr/bin/env node
/*
 * Navigation smoke test for the prompt-11 unified chrome.
 *
 * Spins up a static server over ./public, then walks every top-level
 * public route in headless Chrome. On each page it asserts:
 *   1. the shared .kz-header is present and contains exactly the four
 *      canonical nav links (Translate, Contribute, Progress, About);
 *   2. the shared .kz-footer is present and contains the three column
 *      sections (Use, Project, Community) plus the GitHub link and the
 *      governance mailto address;
 *   3. the nav link matching the current page carries the .is-active
 *      class and aria-current="page" (except on / where no top-level
 *      link matches);
 *   4. the page has a <title>, a <meta name="description">, and a
 *      <link rel="canonical">;
 *   5. every in-page <a> has a non-empty href and every target="_blank"
 *      anchor carries rel containing noopener and noreferrer.
 * It also loads /404.html and asserts the same chrome is present, and
 * loads /sitemap.xml + /robots.txt and asserts they parse sensibly.
 *
 * Run: node tests/smoke/navigation.mjs
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
  '.xml': 'application/xml; charset=utf-8',
};

function mimeFor(p) {
  return MIME[path.extname(p).toLowerCase()] || 'application/octet-stream';
}

function serveFile(res, filePath, status = 200) {
  res.writeHead(status, { 'Content-Type': mimeFor(filePath), 'Cache-Control': 'no-store' });
  fs.createReadStream(filePath).pipe(res);
}

function handler(req, res) {
  const parsed = new URL(req.url, 'http://localhost');
  const p = decodeURIComponent(parsed.pathname);

  // Mirror the server.py aliases for extensionless routes.
  if (p === '/progress' || p === '/progress/') {
    return serveFile(res, path.join(PUBLIC_DIR, 'progress.html'));
  }
  if (p === '/credits' || p === '/credits/') {
    return serveFile(res, path.join(PUBLIC_DIR, 'credits.html'));
  }
  if (p === '/contribute/me' || p === '/contribute/me/') {
    return serveFile(res, path.join(PUBLIC_DIR, 'contribute-me.html'));
  }

  const fp = path.join(PUBLIC_DIR, p === '/' ? '/index.html' : p);
  if (!fp.startsWith(PUBLIC_DIR) || !fs.existsSync(fp) || fs.statSync(fp).isDirectory()) {
    // Mirror the server.py custom 404 behavior for HTML navigations.
    const accept = req.headers.accept || '';
    if (accept.includes('text/html')) {
      return serveFile(res, path.join(PUBLIC_DIR, '404.html'), 404);
    }
    res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' }).end('not found');
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

// The canonical four nav link hrefs, in order. Each page must present
// exactly these four links on its .kz-header.
const CANONICAL_NAV = [
  { href: '/app.html',        label: /^Translate$/i },
  { href: '/contribute.html', label: /^Contribute$/i },
  { href: '/progress',        label: /^Progress$/i },
  { href: '/credits',         label: /^About$/i },
];

// Pages that must show the shared chrome. activePathHref is the nav
// href that should carry .is-active on that page; null when no top-
// level link matches (landing, sub-pages that have breadcrumbs instead).
const PAGES = [
  { url: '/',                  activeHref: null,              label: 'landing' },
  { url: '/app.html',          activeHref: '/app.html',       label: 'translator' },
  { url: '/contribute.html',   activeHref: '/contribute.html', label: 'contribute' },
  { url: '/progress',          activeHref: '/progress',       label: 'progress' },
  { url: '/credits',           activeHref: '/credits',        label: 'credits' },
  { url: '/governance.html',   activeHref: null,              label: 'governance' },
  { url: '/contribute/me',     activeHref: null,              label: 'contribute-me' },
  { url: '/404.html',          activeHref: null,              label: '404' },
];

const FOOTER_COL_TITLES = ['Use', 'Project', 'Community'];

async function checkPage(page, base, target, failures) {
  const url = base + target.url;
  const resp = await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 15000 });
  if (!resp) {
    failures.push(`${target.label} (${target.url}): no response from server`);
    return;
  }

  // Header: the canonical four links, in order, inside .kz-header.
  const header = await page.$('.kz-header');
  if (!header) {
    failures.push(`${target.label} (${target.url}): .kz-header is missing`);
    return;
  }
  const navLinks = await page.$$eval('.kz-header .kz-header__link', (as) => as.map((a) => ({
    href: a.getAttribute('href'),
    text: (a.textContent || '').trim(),
    active: a.classList.contains('is-active'),
    ariaCurrent: a.getAttribute('aria-current'),
  })));
  if (navLinks.length !== CANONICAL_NAV.length) {
    failures.push(`${target.label} (${target.url}): expected ${CANONICAL_NAV.length} nav links, got ${navLinks.length}`);
  }
  for (let i = 0; i < CANONICAL_NAV.length; i++) {
    const want = CANONICAL_NAV[i];
    const got = navLinks[i];
    if (!got) continue;
    if (got.href !== want.href) {
      failures.push(`${target.label}: nav link ${i} href is "${got.href}" — expected "${want.href}"`);
    }
    if (!want.label.test(got.text)) {
      failures.push(`${target.label}: nav link ${i} label is "${got.text}" — expected match ${want.label}`);
    }
  }

  // Active state: exactly zero or one link is active; if active, it
  // matches target.activeHref.
  const activeLinks = navLinks.filter((l) => l.active);
  if (target.activeHref === null) {
    if (activeLinks.length !== 0) {
      failures.push(`${target.label}: expected no active nav link, got ${activeLinks.length}`);
    }
  } else {
    if (activeLinks.length !== 1) {
      failures.push(`${target.label}: expected exactly one active nav link, got ${activeLinks.length}`);
    } else {
      if (activeLinks[0].href !== target.activeHref) {
        failures.push(`${target.label}: active nav link is "${activeLinks[0].href}" — expected "${target.activeHref}"`);
      }
      if (activeLinks[0].ariaCurrent !== 'page') {
        failures.push(`${target.label}: active nav link missing aria-current="page"`);
      }
    }
  }

  // Hamburger exists and has aria attributes.
  const hamburgerState = await page.$eval('.kz-header .kz-header__hamburger', (b) => ({
    hasLabel: !!b.getAttribute('aria-label'),
    expanded: b.getAttribute('aria-expanded'),
  })).catch(() => null);
  if (!hamburgerState) {
    failures.push(`${target.label}: .kz-header__hamburger is missing`);
  } else {
    if (!hamburgerState.hasLabel) failures.push(`${target.label}: hamburger missing aria-label`);
    if (hamburgerState.expanded !== 'false' && hamburgerState.expanded !== 'true') {
      failures.push(`${target.label}: hamburger aria-expanded must be "true" or "false"`);
    }
  }

  // Footer: present, contains the three columns, and has both GitHub
  // and governance mailto links.
  const footer = await page.$('.kz-footer');
  if (!footer) {
    failures.push(`${target.label}: .kz-footer is missing`);
    return;
  }
  const colTitles = await page.$$eval('.kz-footer .kz-footer__col-title', (hs) => hs.map((h) => h.textContent.trim()));
  for (const wanted of FOOTER_COL_TITLES) {
    if (!colTitles.some((t) => t.toLowerCase() === wanted.toLowerCase())) {
      failures.push(`${target.label}: footer missing column "${wanted}" (got ${JSON.stringify(colTitles)})`);
    }
  }
  const footerHrefs = await page.$$eval('.kz-footer a', (as) => as.map((a) => a.getAttribute('href')));
  if (!footerHrefs.some((h) => h && /github\.com\//i.test(h))) {
    failures.push(`${target.label}: footer missing a GitHub link`);
  }
  if (!footerHrefs.some((h) => h && /^mailto:deaf-feedback@kozha\.dev$/i.test(h))) {
    failures.push(`${target.label}: footer missing the governance mailto link`);
  }

  // Meta: title, description, canonical.
  const meta = await page.evaluate(() => {
    var desc = document.querySelector('meta[name="description"]');
    var canonical = document.querySelector('link[rel="canonical"]');
    return {
      title: document.title,
      description: desc ? desc.getAttribute('content') : null,
      canonical: canonical ? canonical.getAttribute('href') : null,
    };
  });
  if (!meta.title) failures.push(`${target.label}: missing <title>`);
  if (!meta.description) failures.push(`${target.label}: missing <meta name="description">`);
  if (!meta.canonical) failures.push(`${target.label}: missing <link rel="canonical">`);

  // Link hygiene: every <a href> is non-empty; every target=_blank has
  // both noopener and noreferrer in rel.
  const anchorIssues = await page.$$eval('a', (anchors) => anchors.map((a, idx) => ({
    idx,
    href: a.getAttribute('href'),
    target: a.getAttribute('target'),
    rel: a.getAttribute('rel') || '',
    text: (a.textContent || '').trim().slice(0, 60),
  })));
  for (const a of anchorIssues) {
    if (a.href === null || a.href === '' || a.href === '#') {
      failures.push(`${target.label}: <a> index ${a.idx} has empty href (text="${a.text}")`);
    }
    if (a.target === '_blank') {
      if (!/\bnoopener\b/.test(a.rel) || !/\bnoreferrer\b/.test(a.rel)) {
        failures.push(`${target.label}: external <a> (text="${a.text}") needs rel="noopener noreferrer"`);
      }
    }
  }
}

async function checkSitemapAndRobots(base, failures) {
  const sitemap = await httpGet(base + '/sitemap.xml');
  if (!sitemap.ok) {
    failures.push(`/sitemap.xml returned ${sitemap.status}`);
  } else if (!/<urlset[\s>]/.test(sitemap.body)) {
    failures.push('/sitemap.xml does not contain a <urlset> element');
  } else {
    for (const required of ['/app.html', '/contribute.html', '/progress', '/credits', '/governance.html']) {
      if (!sitemap.body.includes(required)) {
        failures.push(`/sitemap.xml is missing ${required}`);
      }
    }
  }
  const robots = await httpGet(base + '/robots.txt');
  if (!robots.ok) {
    failures.push(`/robots.txt returned ${robots.status}`);
  } else {
    if (!/Sitemap:\s*https?:\/\/\S+\/sitemap\.xml/i.test(robots.body)) {
      failures.push('/robots.txt does not reference /sitemap.xml');
    }
  }
}

function httpGet(u) {
  return new Promise((resolve) => {
    http.get(u, { headers: { Accept: 'text/html' } }, (res) => {
      const chunks = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => resolve({
        status: res.statusCode,
        ok: res.statusCode >= 200 && res.statusCode < 400,
        body: Buffer.concat(chunks).toString('utf8'),
      }));
    }).on('error', () => resolve({ status: 0, ok: false, body: '' }));
  });
}

async function main() {
  const failures = [];

  const { server, port } = await startServer();
  const base = `http://127.0.0.1:${port}`;

  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 900 });
    for (const target of PAGES) {
      await checkPage(page, base, target, failures);
    }
    await page.close();
    await checkSitemapAndRobots(base, failures);
  } finally {
    await browser.close();
    server.close();
  }

  if (failures.length) {
    console.error('\nFAIL:');
    for (const f of failures) console.error('  - ' + f);
    process.exit(1);
  }
  console.log('\nOK: navigation smoke test passed across ' + PAGES.length + ' pages.');
}

main().catch((err) => {
  console.error(err);
  process.exit(2);
});
