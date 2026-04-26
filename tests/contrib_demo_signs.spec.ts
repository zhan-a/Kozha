/**
 * Demo-card gloss / SiGML alignment.
 *
 * Catches the failure mode that motivated the fake-avatar audit
 * (docs/contrib-fix/prompts/03-fake-avatar-audit.md): the hero card
 * used to caption "BSL · ELECTRON" with a generic SVG silhouette,
 * even though `data/hamnosys_bsl_version1.sigml` had no `electron`
 * entry. Whatever a demo card says it plays, this spec asserts that
 *
 *   1. The `data-gloss` attribute names a real entry in the named
 *      `data-corpus` file. Two corpus shapes are supported:
 *
 *        a. A SiGML corpus file under `data/` (e.g. the walkthrough
 *           card's `hamnosys_bsl_version1.sigml`). The expected tag
 *           list is parsed straight out of the matching `<hns_sign>`.
 *        b. A chat2hamnosys fixture under
 *           `backend/chat2hamnosys/examples/*.json` (e.g. the hero
 *           card's `electron.json`). The expected tag list is derived
 *           by mapping each PUA codepoint in the fixture's
 *           `expected_sign_entry.hamnosys` to its symbol short_name.
 *
 *   2. The SiGML payload embedded in the card (the
 *      `<script type="application/xml" data-demo-payload="<slot>">`
 *      element that contribute-walkthrough.js hands to CWASA) carries
 *      exactly the same `<ham*-/>` tag list — in the same order — as
 *      that corpus entry.
 *
 * Pure file-system test; no browser, no deployed site, no LLM key.
 *
 * Run:  npx playwright test tests/contrib_demo_signs.spec.ts --reporter=list
 */
import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..');
const PUBLIC_DIR = path.join(REPO_ROOT, 'public');
const DATA_DIR = path.join(REPO_ROOT, 'data');

// Subset of backend/chat2hamnosys/hamnosys/symbols.py — only the
// codepoints used by the fixtures the demo cards point at. If a new
// fixture introduces a codepoint not in this map, the test throws
// with the unmapped codepoint so the maintainer adds it here.
//
// Source of truth is the Python module; this map lives in the test
// because the test deliberately has no Python dependency. Keep it in
// sync — drift would mean the test silently miscompares.
const HAMNOSYS_SHORT_NAMES: Record<number, string> = {
  0xe013: 'hamdoublebent',
  0xe020: 'hamextfingeru',
  0xe03a: 'hampalmr',
  0xe051: 'hamshoulders',
  0xe079: 'hamfingerbase',
  0xe093: 'hamcirclei',
  0xe0c5: 'hamdecreasing',
};

interface DemoCard {
  slot: string;
  gloss: string;
  corpus: string;
  payloadTags: string[];
}

function extractHamTags(sigml: string): string[] {
  const tags: string[] = [];
  const re = /<(ham[a-z0-9]+)\s*\/>/gi;
  let m: RegExpExecArray | null;
  while ((m = re.exec(sigml)) !== null) tags.push(m[1].toLowerCase());
  return tags;
}

function findCardsInHtml(html: string): DemoCard[] {
  const cards: DemoCard[] = [];
  // Each card declares slot + gloss + corpus on the same element. The
  // attribute order in HTML is fixed by the source file, so a single
  // regex covers both cards. If a future card lands with a different
  // attribute order this test will fail loudly — that's the desired
  // signal.
  const cardRe =
    /data-demo-card="([^"]+)"[^>]*data-gloss="([^"]+)"[^>]*data-corpus="([^"]+)"/g;
  let m: RegExpExecArray | null;
  while ((m = cardRe.exec(html)) !== null) {
    const slot = m[1];
    const gloss = m[2];
    const corpus = m[3];
    const payloadRe = new RegExp(
      `<script[^>]*data-demo-payload="${slot}"[^>]*>([\\s\\S]*?)<\\/script>`,
      'i',
    );
    const pm = payloadRe.exec(html);
    if (!pm) {
      throw new Error(`demo card "${slot}" has no data-demo-payload="${slot}" script`);
    }
    cards.push({
      slot,
      gloss,
      corpus,
      payloadTags: extractHamTags(pm[1]),
    });
  }
  return cards;
}

function loadSigmlCorpusTags(corpus: string, gloss: string): string[] {
  const file = path.join(DATA_DIR, corpus);
  if (!fs.existsSync(file)) {
    throw new Error(`corpus file not found: ${file}`);
  }
  const text = fs.readFileSync(file, 'utf8');
  // Escape regex specials in the gloss — corpus glosses include `(`,
  // `)`, `#`, `^`, etc.
  const escGloss = gloss.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const re = new RegExp(
    `<hns_sign\\s+gloss="${escGloss}"\\s*>([\\s\\S]*?)<\\/hns_sign>`,
    'i',
  );
  const m = re.exec(text);
  if (!m) {
    throw new Error(`gloss "${gloss}" not found in data/${corpus}`);
  }
  return extractHamTags(m[1]);
}

function loadFixtureExpectedTags(corpus: string, gloss: string): string[] {
  const file = path.join(REPO_ROOT, corpus);
  if (!fs.existsSync(file)) {
    throw new Error(`fixture not found: ${file}`);
  }
  const fixture = JSON.parse(fs.readFileSync(file, 'utf8'));
  const expected = fixture.expected_sign_entry;
  if (!expected || typeof expected.hamnosys !== 'string') {
    throw new Error(`fixture missing expected_sign_entry.hamnosys: ${file}`);
  }
  if ((expected.gloss || '').toLowerCase() !== gloss.toLowerCase()) {
    throw new Error(
      `fixture gloss mismatch: card declares "${gloss}", fixture has "${expected.gloss}"`,
    );
  }
  const tags: string[] = [];
  for (const ch of expected.hamnosys as string) {
    const cp = ch.codePointAt(0);
    if (cp == null) continue;
    const tag = HAMNOSYS_SHORT_NAMES[cp];
    if (!tag) {
      throw new Error(
        `fixture ${path.basename(file)} uses HamNoSys codepoint ` +
          `U+${cp.toString(16).padStart(4, '0').toUpperCase()} which is not in ` +
          `HAMNOSYS_SHORT_NAMES; add it (mirror backend/chat2hamnosys/hamnosys/symbols.py)`,
      );
    }
    tags.push(tag);
  }
  return tags;
}

function loadExpectedTags(corpus: string, gloss: string): string[] {
  return corpus.endsWith('.json')
    ? loadFixtureExpectedTags(corpus, gloss)
    : loadSigmlCorpusTags(corpus, gloss);
}

const HTML = fs.readFileSync(path.join(PUBLIC_DIR, 'contribute.html'), 'utf8');
const CARDS = findCardsInHtml(HTML);

test.describe('contribute.html demo cards', () => {
  test('hero and walkthrough cards both present', () => {
    const slots = CARDS.map((c) => c.slot).sort();
    expect(slots).toContain('hero');
    expect(slots).toContain('walk');
  });

  test('no fake silhouette markup remains', () => {
    // The legacy SVG silhouettes were the "CSS-only avatar mock
    // illustrations" the audit targeted. Their CSS classes were the
    // load-bearing identifiers; if any sneak back in, fail loudly.
    expect(HTML).not.toContain('c2-cwasa-poster');
    expect(HTML).not.toContain('c2-hero__cwasa-poster');
    expect(HTML).not.toContain('c2-viz-4__avatar-poster');
  });

  test('no unsubstantiated reviewer-count claim on the hero card', () => {
    // The hero card's previous "Reviewed by 2 Deaf signers" caption
    // was unsubstantiated — the ELECTRON fixture has 0 qualifying
    // approvals. Prompt 04 requires the badge be removed entirely
    // when no gloss qualifies. Catch any regression that re-asserts
    // a Deaf-reviewer count without backing data.
    expect(HTML).not.toMatch(/Reviewed by \d+ Deaf signers?/i);
  });

  for (const card of CARDS) {
    test.describe(`card "${card.slot}" (${card.gloss})`, () => {
      test('payload tag list matches the corpus entry', () => {
        const corpusTags = loadExpectedTags(card.corpus, card.gloss);
        expect(corpusTags.length).toBeGreaterThan(0);
        expect(card.payloadTags).toEqual(corpusTags);
      });

      test('payload contains the canonical gloss', () => {
        // The script body was already located by findCardsInHtml; here
        // we re-read it to assert the gloss attribute on <hns_sign> is
        // present inside the payload — protects against a paste-error
        // where the data-gloss attribute and the inline SiGML drift.
        const re = new RegExp(
          `<script[^>]*data-demo-payload="${card.slot}"[^>]*>([\\s\\S]*?)<\\/script>`,
          'i',
        );
        const body = re.exec(HTML)?.[1] ?? '';
        expect(body).toContain(`gloss="${card.gloss}"`);
      });
    });
  }
});
