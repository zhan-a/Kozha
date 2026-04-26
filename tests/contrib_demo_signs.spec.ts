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
 *      `data-corpus` file under `data/`.
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

function loadCorpusEntryTags(corpus: string, gloss: string): string[] {
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

  for (const card of CARDS) {
    test.describe(`card "${card.slot}" (${card.gloss})`, () => {
      test('payload tag list matches the corpus entry', () => {
        const corpusTags = loadCorpusEntryTags(card.corpus, card.gloss);
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
