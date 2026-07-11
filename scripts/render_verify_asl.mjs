#!/usr/bin/env node
/*
 * Headless render-verification harness for the ASL seed lexicon (proposal 59).
 *
 * Turns "does this sign actually render?" into an automatic, measurable check
 * instead of per-sign human review. For every <hns_sign> in
 * data/American_SL_ASL.sigml it:
 *   - serves public/ + data/ and loads a minimal CWASA/JASigning page headless
 *     (reusing the tests/smoke/translator-sigml.mjs serving scaffold),
 *   - plays the single sign on the avatar,
 *   - captures a representative (peak-pose) frame to
 *     proposals/reports/asl-render-verify/<gloss>.png,
 *   - records: rendered? (no console/parse error), non-degenerate? (the avatar
 *     actually moved — playback motion vs a static idle baseline), and any
 *     dropped-tag warnings.
 *
 * The avatar runs with ambIdle:false, so an idle avatar is pixel-stable
 * (measured ~0.0000 frame delta) while a sign that animates moves a large
 * fraction of the canvas. A sign that leaves the avatar frozen therefore failed
 * to animate -> status "static". Parse/console errors -> "error". Otherwise
 * "ok".
 *
 * Outputs:
 *   proposals/reports/asl-render-verify/<gloss>.png   per-sign thumbnails
 *   proposals/reports/asl-render-verify/results.json  machine verdict
 *   proposals/reports/asl-render-verify/index.md       contact sheet
 *   proposals/reports/asl-render-verify.md             summary + routing lists
 *
 * Deterministic + headless (--no-sandbox) so it can run in CI. Zero LLM.
 *
 * If puppeteer/Chromium can't launch, it falls back to the strongest static
 * check available (XML well-formed + every tag in CWASA's tokenNameMap +
 * handshape-led slot order) and clearly documents that visual verification was
 * not possible, so a human runs it locally.
 *
 * Run:
 *   node scripts/render_verify_asl.mjs        (or: npm run verify:asl)
 */

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { PNG } from 'pngjs';
import pixelmatch from 'pixelmatch';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..');
const PUBLIC_DIR = path.join(REPO_ROOT, 'public');
const DATA_DIR = path.join(REPO_ROOT, 'data');
const ALLCSA = path.join(PUBLIC_DIR, 'cwa', 'allcsa.js');
const SIGML_FILE = process.env.SIGML_FILE
  ? path.resolve(process.env.SIGML_FILE)
  : path.join(DATA_DIR, 'American_SL_ASL.sigml');

const OUT_DIR = process.env.OUT_DIR
  ? path.resolve(process.env.OUT_DIR)
  : path.join(REPO_ROOT, 'proposals', 'reports', 'asl-render-verify');
const SUMMARY_MD = process.env.SUMMARY_MD
  ? path.resolve(process.env.SUMMARY_MD)
  : path.join(REPO_ROOT, 'proposals', 'reports', 'asl-render-verify.md');

// --- tuning -----------------------------------------------------------------
const AVATAR = 'anna';
const CANVAS_PX = 512;
const SETTLE_MS = 700;        // let the avatar return to rest before a sign
const SAMPLE_MS = 160;        // gap between playback frame captures
const SAMPLE_FRAMES = 18;     // ~2.9s playback window
const PIXELMATCH_THRESHOLD = 0.1; // per-pixel colour tolerance
// A sign is "static" (degenerate) if the peak fraction of changed pixels vs the
// rest pose never clears the motion threshold. With ambIdle:false the idle
// avatar is pixel-stable (measured ~0%), while a sign that animates raises a
// hand to the body — a small but unambiguous fraction of a mostly-white 512²
// frame (~1-4%). The threshold is max(this floor, measured idle noise * k) so
// it adapts if a build ever introduces idle jitter.
const MOTION_FLOOR = 0.003;
const IDLE_NOISE_K = 4;
const PLAY_TIMEOUT_MS = 12000;   // a healthy playSiGMLText returns in <1s
// A screenshot of a healthy canvas returns in well under a second. A sign whose
// animation generator runs away wedges the renderer so the *screenshot* (not the
// async play call) never returns; bound it so such a sign fails in seconds
// instead of stalling on the 180s protocolTimeout.
const SHOT_TIMEOUT_MS = 8000;

const CONTAINER_TAGS = new Set([
  'hamnosys_manual', 'hamnosys_nonmanual', 'hamgestural_sign', 'hns_sign', 'sigml',
]);

// Tags that may legitimately precede the handshape in HamNoSys slot order
// (symmetry operators). Used only by the static "handshape-led" fallback check.
const SYMMETRY_PREFIXES = ['hamsymm', 'hamnonsymm', 'hamparbegin', 'hamparend', 'hamplus'];
// Prefixes that are NOT handshapes (orientation / location / movement / mods).
// The first manual tag after any symmetry operator should be none of these.
const NON_HANDSHAPE_PREFIXES = [
  'hamextfinger', 'hampalm',                                   // orientation
  'hammov', 'hamarc', 'hamcircle', 'hamwave', 'hamzigzag',     // movement
  'hamnomotion', 'hamrepeat', 'hamreplace', 'hamseq', 'hampar', // movement/seq
  'hamlrat', 'hamfingertip', 'hampalm', 'hamtouch',            // contact/loc-mod
];

// --- static helpers (also the fallback path) --------------------------------

function extractKnownTags() {
  const src = fs.readFileSync(ALLCSA, 'utf8');
  const start = src.indexOf('HNSDefs.tokenNameMap = [');
  if (start < 0) throw new Error('tokenNameMap not found in allcsa.js');
  const end = src.indexOf('];', start);
  const section = src.slice(start, end);
  const known = new Set();
  for (const m of section.matchAll(/"([^"]*)"/g)) {
    for (const name of m[1].split(/\s+/)) {
      if (name) known.add(name);
    }
  }
  return known;
}

// Pull each <hns_sign>...</hns_sign> block, its gloss and its ham* tag list.
function parseSigns(sigmlText) {
  const signs = [];
  const re = /<hns_sign\b([^>]*)>([\s\S]*?)<\/hns_sign>/g;
  let m;
  while ((m = re.exec(sigmlText)) !== null) {
    const attrs = m[1];
    const inner = m[2];
    const glossMatch = attrs.match(/gloss\s*=\s*"([^"]*)"/);
    const gloss = glossMatch ? glossMatch[1] : '';
    const manualMatch = inner.match(/<hamnosys_manual>([\s\S]*?)<\/hamnosys_manual>/);
    const manualInner = manualMatch ? manualMatch[1] : inner;
    const manualTags = [...manualInner.matchAll(/<(ham[A-Za-z][\w]*)/g)].map((x) => x[1]);
    const allTags = [...inner.matchAll(/<((?:ham|hnm)[A-Za-z][\w]*)/g)].map((x) => x[1]);
    signs.push({ gloss, block: m[0], inner, manualTags, allTags });
  }
  return signs;
}

// Minimal well-formedness scanner for this flat schema (open/close/self-close
// tags, attributes, comments, xml decl). Dependency-free so the fallback can
// claim "XML well-formed" without a DOM parser.
function isWellFormed(xml) {
  const stack = [];
  const tagRe = /<(\/?)([A-Za-z_][\w.-]*)((?:[^<>"']|"[^"]*"|'[^']*')*?)(\/?)>|<\?[\s\S]*?\?>|<!--[\s\S]*?-->/g;
  let m;
  let lastIndex = 0;
  while ((m = tagRe.exec(xml)) !== null) {
    // reject stray '<' between tags (malformed)
    const between = xml.slice(lastIndex, m.index);
    if (between.includes('<')) return false;
    lastIndex = tagRe.lastIndex;
    if (m[0].startsWith('<?') || m[0].startsWith('<!--')) continue;
    const closing = m[1] === '/';
    const name = m[2];
    const selfClose = m[4] === '/';
    if (closing) {
      if (stack.pop() !== name) return false;
    } else if (!selfClose) {
      stack.push(name);
    }
  }
  if (xml.slice(lastIndex).includes('<')) return false;
  return stack.length === 0;
}

function unknownTagsFor(sign, known) {
  return sign.allTags.filter((t) => !CONTAINER_TAGS.has(t) && !known.has(t));
}

// Static slot-order check: first manual tag after any symmetry operator must
// look like a handshape (i.e. not an orientation/location/movement tag).
function isHandshapeLed(sign) {
  let i = 0;
  while (i < sign.manualTags.length && SYMMETRY_PREFIXES.some((p) => sign.manualTags[i].startsWith(p))) i++;
  if (i >= sign.manualTags.length) return false;
  const first = sign.manualTags[i];
  return !NON_HANDSHAPE_PREFIXES.some((p) => first.startsWith(p));
}

function safeName(gloss) {
  return (gloss || 'UNKNOWN').replace(/[^A-Za-z0-9._-]/g, '_');
}

// --- static server (serves public/ + data/ + the harness page) --------------

const MIME = {
  '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8', '.svg': 'image/svg+xml',
  '.sigml': 'application/xml; charset=utf-8', '.csv': 'text/csv; charset=utf-8',
  '.ttf': 'font/ttf', '.woff': 'font/woff', '.woff2': 'font/woff2',
  '.ico': 'image/x-icon', '.png': 'image/png', '.wasm': 'application/wasm',
  '.jar': 'application/java-archive', '.txt': 'text/plain; charset=utf-8',
};
const mimeFor = (p) => MIME[path.extname(p).toLowerCase()] || 'application/octet-stream';

function harnessHtml() {
  return `<!doctype html><html lang="en"><head><meta charset="utf-8">
<link rel="stylesheet" href="/cwa/cwasa.css">
<script defer src="/cwa/allcsa.js"></script>
<style>
  html,body{margin:0;background:#fff}
  .stage{width:${CANVAS_PX}px;height:${CANVAS_PX}px;background:#fff}
  .CWASAAvatar.av0 canvas{width:${CANVAS_PX}px;height:${CANVAS_PX}px}
</style></head><body>
<div class="stage"><div class="CWASAAvatar av0"></div>
<div class="CWASAGUI av0" style="display:none" aria-hidden="true"></div></div>
<script>
  window.__cwasaReady=false;window.__cwasaErr=null;
  window.addEventListener('load',function(){setTimeout(function(){try{
    CWASA.init({useClientConfig:false,useCwaConfig:true,avSettings:[{
      width:${CANVAS_PX},height:${CANVAS_PX},avList:'avs',initAv:'${AVATAR}',
      ambIdle:false,allowFrameSteps:false,allowSiGMLText:true
    }]});
    CWASA.addHook('avatarready',function(){window.__cwasaReady=true;},0);
  }catch(e){window.__cwasaErr=String(e);}},50);});
</script></body></html>`;
}

function makeHandler() {
  const html = harnessHtml();
  return function handler(req, res) {
    const p = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
    if (p === '/__asl_render_harness.html') {
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store' });
      res.end(html);
      return;
    }
    if (p === '/favicon.ico') { res.writeHead(204).end(); return; }
    if (p.startsWith('/data/')) {
      const fp = path.join(DATA_DIR, p.slice('/data/'.length));
      if (!fp.startsWith(DATA_DIR) || !fs.existsSync(fp)) { res.writeHead(404).end('not found'); return; }
      res.writeHead(200, { 'Content-Type': mimeFor(fp), 'Cache-Control': 'no-store' });
      fs.createReadStream(fp).pipe(res);
      return;
    }
    const fp = path.join(PUBLIC_DIR, p === '/' ? '/index.html' : p);
    if (!fp.startsWith(PUBLIC_DIR) || !fs.existsSync(fp) || fs.statSync(fp).isDirectory()) {
      res.writeHead(404).end('not found');
      return;
    }
    res.writeHead(200, { 'Content-Type': mimeFor(fp), 'Cache-Control': 'no-store' });
    fs.createReadStream(fp).pipe(res);
  };
}

function startServer() {
  return new Promise((resolve, reject) => {
    const server = http.createServer(makeHandler());
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port }));
  });
}

// --- pixel diff -------------------------------------------------------------

function decode(buf) {
  return PNG.sync.read(Buffer.from(buf));
}

// Fraction of pixels that changed between two same-size PNG buffers.
function changedFraction(restPng, framePng) {
  if (restPng.width !== framePng.width || restPng.height !== framePng.height) return 1;
  const { width, height } = restPng;
  const diff = pixelmatch(restPng.data, framePng.data, null, width, height, {
    threshold: PIXELMATCH_THRESHOLD,
  });
  return diff / (width * height);
}

// --- error/warning classification -------------------------------------------

const PARSE_ERR_RE = /mismatched input|\[object Object\]|parse ?error|cannot parse|unrecognis|unrecognized|SyntaxError|is not a function|cannot read/i;
const DROPPED_TAG_RE = /unknown|not recognis|dropped|ignoring|ignored|skip/i;

// --- browser verification ---------------------------------------------------

async function runBrowserVerify(puppeteer, signs, known) {
  const { server, port } = await startServer();
  // NB: do NOT pass --disable-gpu — the CWASA avatar renders via WebGL and that
  // flag kills the GL context headless (avatarready then never fires). These are
  // the same flags the proven tests/smoke/translator-sigml.mjs uses.
  const browser = await puppeteer.launch({
    headless: true,
    protocolTimeout: 180000,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--hide-scrollbars'],
  });
  const results = [];
  try {
    let page;
    let bucket = [];

    // A fresh page per sign: a long-lived page wedges the renderer after ~7
    // signs of play+screenshot (Runtime.callFunctionOn / captureScreenshot time
    // out as CWASA animation state accumulates). Recreating the page on a thrown
    // infra error (cold-start nav timeout, renderer stall) and retrying makes
    // the run deterministic; real render failures don't throw — they're detected
    // via playThrew / console / motion and produce a verdict without retrying.
    async function newPage() {
      if (page) { try { await page.close(); } catch (_e) {} }
      page = await browser.newPage();
      page.on('console', (msg) => {
        const t = msg.type();
        if (t === 'error' || t === 'warning') bucket.push({ type: t, text: msg.text() });
      });
      page.on('pageerror', (err) => bucket.push({ type: 'pageerror', text: String(err) }));
      await page.setViewport({ width: CANVAS_PX + 40, height: CANVAS_PX + 40 });
    }

    async function loadAvatar() {
      await page.goto(`http://127.0.0.1:${port}/__asl_render_harness.html`, {
        waitUntil: 'networkidle0', timeout: 60000,
      });
      await page.waitForFunction('window.__cwasaReady===true || window.__cwasaErr', { timeout: 60000 });
      const initErr = await page.evaluate(() => window.__cwasaErr);
      if (initErr) throw new Error('CWASA init failed: ' + initErr);
      await page.waitForSelector('.CWASAAvatar.av0 canvas', { timeout: 15000 });
    }

    async function shoot() {
      const canvas = await page.$('.CWASAAvatar.av0 canvas');
      if (!canvas) return null;
      try {
        return await canvas.screenshot();
      } finally {
        await canvas.dispose();
      }
    }

    // Bound a screenshot so a wedged renderer is abandoned in seconds. The
    // underlying call may still reject later (at protocolTimeout) once we've moved
    // on — swallow that so it isn't an unhandled rejection.
    async function shootGuarded() {
      const p = shoot();
      p.catch(() => {});
      return withTimeout(p, SHOT_TIMEOUT_MS,
        `screenshot hung — renderer wedged (no frame within ${SHOT_TIMEOUT_MS}ms)`);
    }

    // Run one sign on a fresh avatar. Returns a verdict rec, or THROWS on infra
    // failure (so the caller can recreate the page and retry).
    async function verifySign(sign, unknown, doc) {
      await loadAvatar();
      await sleep(SETTLE_MS);
      const restBuf = await shootGuarded();
      const restPng = restBuf ? decode(restBuf) : null;
      await sleep(SAMPLE_MS);
      const idleBuf = await shootGuarded();
      const idleNoise = (restPng && idleBuf) ? changedFraction(restPng, decode(idleBuf)) : 0;

      // scope console/error capture to the play window (exclude page-load noise)
      bucket = [];
      // Normally playSiGMLText returns near-instantly (animation runs async via
      // requestAnimationFrame). Some encodings make CWASA's animation generator
      // hang the JS thread synchronously — bound the call so such a sign fails
      // fast and deterministically instead of stalling on protocolTimeout.
      const playPromise = page.evaluate((s) => {
        window.__playErr = null;
        try { CWASA.playSiGMLText(s, 0); } catch (e) { window.__playErr = String(e); }
      }, doc);
      playPromise.catch(() => {}); // swallow late rejection if we abandon a hung call
      await withTimeout(playPromise, PLAY_TIMEOUT_MS,
        `playSiGMLText hung — sign wedged the renderer (no return within ${PLAY_TIMEOUT_MS}ms)`);

      let peakFrac = 0;
      let peakBuf = restBuf;
      let prevPng = restPng;
      let totalMotion = 0;
      for (let i = 0; i < SAMPLE_FRAMES; i++) {
        await sleep(SAMPLE_MS);
        const buf = await shootGuarded();
        if (!buf || !restPng) continue;
        const png = decode(buf);
        const fracVsRest = changedFraction(restPng, png);
        if (prevPng) totalMotion += changedFraction(prevPng, png);
        prevPng = png;
        if (fracVsRest > peakFrac) { peakFrac = fracVsRest; peakBuf = buf; }
      }
      const playThrew = await page.evaluate(() => window.__playErr);
      if (!restPng) throw new Error('could not capture avatar canvas (screenshot stalled)');

      const consoleErrs = bucket.filter((e) => e.type === 'pageerror'
        || (e.type === 'error' && PARSE_ERR_RE.test(e.text)));
      const droppedWarnings = [...new Set(bucket
        .filter((e) => e.type === 'warning' && DROPPED_TAG_RE.test(e.text))
        .map((e) => e.text))];

      if (peakBuf) fs.writeFileSync(path.join(OUT_DIR, `${safeName(sign.gloss)}.png`), Buffer.from(peakBuf));

      const threshold = Math.max(MOTION_FLOOR, idleNoise * IDLE_NOISE_K);
      let status;
      const notesParts = [];
      if (playThrew || consoleErrs.length || unknown.length) {
        status = 'error';
        if (playThrew) notesParts.push(`play threw: ${playThrew}`);
        if (consoleErrs.length) notesParts.push(`${consoleErrs.length} parse/render error(s): ${consoleErrs.map((e) => e.text).join(' | ').slice(0, 240)}`);
        if (unknown.length) notesParts.push(`unknown tag(s): ${unknown.join(', ')}`);
      } else if (peakFrac < threshold) {
        status = 'static';
        notesParts.push(`avatar did not move (peak ${pct(peakFrac)} of pixels vs rest; threshold ${pct(threshold)}) — sign failed to animate`);
      } else {
        status = 'ok';
        notesParts.push(`rendered + animated (peak ${pct(peakFrac)} of pixels moved vs rest)`);
      }
      if (droppedWarnings.length) {
        notesParts.push(`dropped-tag warning(s): ${droppedWarnings.join(' | ').slice(0, 240)}`);
      }

      return {
        gloss: sign.gloss,
        status,
        notes: notesParts.join('; '),
        metrics: {
          peak_changed_fraction: round(peakFrac),
          total_motion: round(totalMotion),
          idle_noise_fraction: round(idleNoise),
          motion_threshold: round(threshold),
          dropped_tag_warnings: droppedWarnings.length,
          unknown_tags: unknown,
          thumbnail: peakBuf ? `${safeName(sign.gloss)}.png` : null,
        },
      };
    }

    const MAX_ATTEMPTS = 3;
    await newPage();
    // Warm up the browser/GPU/WASM so the first real sign isn't penalised by a
    // cold-start navigation timeout.
    try { await loadAvatar(); } catch (_e) { /* ignore warmup failure */ }

    for (const sign of signs) {
      const unknown = [...new Set(unknownTagsFor(sign, known))];
      const doc = `<?xml version="1.0" encoding="utf-8"?>\n<sigml>\n${sign.block}\n</sigml>`;
      let rec = null;
      let lastErr = null;
      for (let attempt = 1; attempt <= MAX_ATTEMPTS && !rec; attempt++) {
        try {
          rec = await verifySign(sign, unknown, doc);
        } catch (e) {
          lastErr = e;
          if (attempt < MAX_ATTEMPTS) {
            console.warn(`  retry ${sign.gloss} (attempt ${attempt} failed: ${e.message.split('\n')[0]})`);
            await newPage(); // clear a wedged renderer before the next try
          }
        }
      }
      if (!rec) {
        const msg = lastErr ? lastErr.message.split('\n')[0] : 'unknown';
        const hung = /hung|callFunctionOn timed out|captureScreenshot/i.test(msg);
        rec = {
          gloss: sign.gloss,
          status: 'error',
          notes: hung
            ? `does not render — the avatar hung on this sign across ${MAX_ATTEMPTS} attempts (${msg})`
            : `harness exception after ${MAX_ATTEMPTS} attempts: ${msg}`,
          metrics: { unknown_tags: unknown, thumbnail: null, render_hang: hung },
        };
      }
      results.push(rec);
      const pk = rec.metrics && rec.metrics.peak_changed_fraction != null ? pct(rec.metrics.peak_changed_fraction) : 'n/a';
      const warn = rec.metrics && rec.metrics.dropped_tag_warnings ? '(dropped-tag warn) ' : '';
      console.log(`[${rec.status}] ${sign.gloss}  peak=${pk}  ${warn}`);
    }
    if (page) await page.close();
  } finally {
    await browser.close();
    server.close();
  }
  return { mode: 'browser', results };
}

// --- static fallback (no browser) -------------------------------------------

function runStaticFallback(signs, known, reason) {
  const fileText = fs.readFileSync(SIGML_FILE, 'utf8');
  const fileWellFormed = isWellFormed(fileText);
  const results = signs.map((sign) => {
    const unknown = [...new Set(unknownTagsFor(sign, known))];
    const wellFormed = fileWellFormed && isWellFormed(`<root>${sign.inner}</root>`);
    const handshapeLed = isHandshapeLed(sign);
    const notesParts = ['VISUAL VERIFICATION NOT POSSIBLE (no Chromium) — static checks only'];
    let status = 'ok';
    if (!wellFormed) { status = 'error'; notesParts.push('not well-formed XML'); }
    if (unknown.length) { status = 'error'; notesParts.push(`unknown tag(s): ${unknown.join(', ')}`); }
    if (!handshapeLed) { status = 'error'; notesParts.push('not handshape-led (slot order)'); }
    if (status === 'ok') notesParts.push('well-formed, all tags in CWASA tokenNameMap, handshape-led — run locally to confirm it animates');
    return {
      gloss: sign.gloss,
      status,
      notes: notesParts.join('; '),
      metrics: { well_formed: wellFormed, handshape_led: handshapeLed, unknown_tags: unknown, thumbnail: null },
    };
  });
  return { mode: 'static-fallback', reason, results };
}

// --- reporting --------------------------------------------------------------

function writeOutputs(run) {
  const counts = { ok: 0, static: 0, error: 0 };
  for (const r of run.results) counts[r.status] = (counts[r.status] || 0) + 1;
  const generatedAt = new Date().toISOString();

  const resultsJson = {
    generated_at: generatedAt,
    mode: run.mode,
    avatar: AVATAR,
    sigml_file: path.relative(REPO_ROOT, SIGML_FILE),
    motion_floor: MOTION_FLOOR,
    counts,
    visual_verification: run.mode === 'browser',
    note: run.mode === 'browser'
      ? 'Each sign was played on a headless CWASA avatar (ambIdle:false); status reflects render success + measured motion.'
      : `Visual verification was NOT possible (${run.reason}); status reflects static checks only. Run locally with Chromium to verify motion.`,
    signs: run.results,
  };
  fs.writeFileSync(path.join(OUT_DIR, 'results.json'), JSON.stringify(resultsJson, null, 2) + '\n');

  // contact sheet
  const idx = [];
  idx.push('# ASL render-verification — contact sheet');
  idx.push('');
  idx.push(`Generated ${generatedAt} · mode \`${run.mode}\` · avatar \`${AVATAR}\``);
  idx.push('');
  idx.push(`**ok ${counts.ok} · static ${counts.static || 0} · error ${counts.error || 0}** of ${run.results.length} signs.`);
  idx.push('');
  if (run.mode !== 'browser') {
    idx.push(`> Visual verification was not possible (${run.reason}). No thumbnails; verdicts are static-only.`);
    idx.push('');
  }
  idx.push('| sign | status | thumbnail | notes |');
  idx.push('|---|---|---|---|');
  for (const r of run.results) {
    const thumb = r.metrics && r.metrics.thumbnail ? `![${r.gloss}](./${r.metrics.thumbnail})` : '_n/a_';
    const badge = r.status === 'ok' ? 'ok ✅' : r.status === 'static' ? 'static ⚠️' : 'error ❌';
    idx.push(`| \`${r.gloss}\` | ${badge} | ${thumb} | ${r.notes.replace(/\|/g, '\\|')} |`);
  }
  idx.push('');
  fs.writeFileSync(path.join(OUT_DIR, 'index.md'), idx.join('\n') + '\n');

  // summary + routing
  const errors = run.results.filter((r) => r.status === 'error');
  const statics = run.results.filter((r) => r.status === 'static');
  const oks = run.results.filter((r) => r.status === 'ok');
  const md = [];
  md.push('# ASL render-verification — summary (proposal 59)');
  md.push('');
  md.push(`Generated ${generatedAt} · mode \`${run.mode}\` · avatar \`${AVATAR}\` · source \`${path.relative(REPO_ROOT, SIGML_FILE)}\``);
  md.push('');
  if (run.mode === 'browser') {
    md.push('Each active ASL `<hns_sign>` was loaded into a headless CWASA/JASigning avatar');
    md.push('(`ambIdle:false`), played, and scored on two axes: **rendered?** (no console/parse');
    md.push('error and no unknown tags) and **non-degenerate?** (the avatar actually moved —');
    md.push(`peak changed-pixel fraction vs the static rest pose exceeded the ${pct(MOTION_FLOOR)} floor;`);
    md.push('an idle avatar measures ~0%). Thumbnails (peak pose) are in');
    md.push('`asl-render-verify/` — see `index.md` for the contact sheet.');
  } else {
    md.push(`**Visual verification was NOT possible** (${run.reason}). Chromium/puppeteer could not`);
    md.push('launch, so each sign was scored with the strongest static check available:');
    md.push('XML well-formed + every tag ∈ CWASA `tokenNameMap` + handshape-led slot order.');
    md.push('**A human must run `npm run verify:asl` locally with Chromium to confirm motion.**');
  }
  md.push('');
  md.push('## Counts');
  md.push('');
  md.push('| status | count | meaning |');
  md.push('|---|---|---|');
  md.push(`| ok | ${counts.ok} | rendered and animated (or, in fallback, passed all static checks) |`);
  md.push(`| static | ${counts.static || 0} | rendered without error but the avatar never moved — sign failed to animate |`);
  md.push(`| error | ${counts.error || 0} | console/parse error, play threw, or an unknown tag |`);
  md.push(`| **total** | **${run.results.length}** | |`);
  md.push('');
  md.push('## Routing');
  md.push('');
  md.push('- **Quarantine** (move to `data/American_SL_ASL_quarantine.sigml`, not served) — signs that');
  md.push('  error out (unknown tag / parse failure / play threw). These do not render at all.');
  if (errors.length) {
    for (const r of errors) md.push(`  - \`${r.gloss}\` — ${r.notes}`);
  } else {
    md.push('  - _none._');
  }
  md.push('');
  md.push('- **Route to stage 60 (gated LLM repair)** — signs that render but are degenerate (static),');
  md.push('  i.e. structurally valid SiGML the avatar will not animate, so the heuristic encoding needs');
  md.push('  a smarter pass.');
  if (statics.length) {
    for (const r of statics) md.push(`  - \`${r.gloss}\` — ${r.notes}`);
  } else {
    md.push('  - _none._');
  }
  md.push('');
  md.push('## Passing signs');
  md.push('');
  md.push(`${oks.length} sign(s) render and animate cleanly: ${oks.map((r) => '`' + r.gloss + '`').join(', ') || '_none_'}.`);
  md.push('');
  md.push('> Note: "ok" here means **renderable + non-degenerate**, not **correct**. Per proposal 55,');
  md.push('> palm/finger orientation and movement direction are heuristic defaults that ASL-LEX does not');
  md.push('> encode, so every sign still ships as `seed` / needs Deaf-native review regardless of this');
  md.push('> gate. This harness measures *renderability*, not lexical accuracy.');
  md.push('');
  fs.writeFileSync(SUMMARY_MD, md.join('\n') + '\n');

  return { counts, resultsJson };
}

// --- utils ------------------------------------------------------------------

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const round = (x) => Math.round(x * 1e4) / 1e4;
const pct = (x) => `${(x * 100).toFixed(2)}%`;

// Reject if `promise` doesn't settle within `ms`. Used to bound a play call that
// can hang the renderer; the caller recreates the page to abandon the hung call.
function withTimeout(promise, ms, label) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(label)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

// --- main -------------------------------------------------------------------

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const known = extractKnownTags();
  const sigmlText = fs.readFileSync(SIGML_FILE, 'utf8');
  const signs = parseSigns(sigmlText);
  if (!signs.length) throw new Error(`no <hns_sign> entries found in ${SIGML_FILE}`);
  console.log(`Loaded ${signs.length} ASL signs; CWASA knows ${known.size} tags.`);

  let run;
  let puppeteer = null;
  try {
    puppeteer = (await import('puppeteer')).default;
  } catch (e) {
    console.warn('puppeteer not installed — static fallback. (' + e.message + ')');
  }

  if (puppeteer) {
    try {
      run = await runBrowserVerify(puppeteer, signs, known);
    } catch (e) {
      console.warn('Browser verification failed (' + e.message + ') — falling back to static checks.');
      run = runStaticFallback(signs, known, 'Chromium launch/verify failed: ' + e.message);
    }
  } else {
    run = runStaticFallback(signs, known, 'puppeteer not installed');
  }

  const { counts } = writeOutputs(run);
  console.log(`\nmode=${run.mode}  ok=${counts.ok} static=${counts.static || 0} error=${counts.error || 0}`);
  console.log(`Wrote ${path.relative(REPO_ROOT, OUT_DIR)}/{results.json,index.md,*.png} and ${path.relative(REPO_ROOT, SUMMARY_MD)}`);
  // Exit non-zero only on hard errors so CI can gate; "static" is informational.
  if (counts.error > 0) process.exitCode = 1;
}

main().catch((err) => {
  console.error(err);
  process.exit(2);
});
