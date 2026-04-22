# What's new — April 2026 polish release

Plain summary of everything that changed since the pre-polish state.

## Fixed

- **Translator bug where certain word → language combinations produced
  no output.** The original report was English "fruit" → LSF, which
  emitted a SiGML string containing a literal `[object Object]` and
  failed to render. Root cause was fixed in the translator's HamNoSys
  composer, and similar database entries using the unsupported
  `<hampalmud/>` tag were either repaired or quarantined so they stop
  silently breaking the avatar.
- **Contribute page pipeline could get stuck on common English
  terms.** Describing a sign like *"dominant hand up and waving"*
  used to bounce the "waving" token into an LLM fallback that
  sometimes returned a BadRequestError. The vocabulary now maps
  `waving` / `wave` / `waves` to the HamNoSys pendulum-swing
  codepoint, so the deterministic composer resolves them without
  calling out.

## New

- **Progress dashboard at `/progress`.** Public, live view of
  language coverage, reviewer status, contribution activity, and
  cross-language gloss gaps. The "Help wanted" section shows real
  gaps from the coverage matrix; clicking a gap pre-fills the
  contribute form for that gloss.
- **Expanded credits page at `/credits`.** Full per-corpus
  attribution: institution, license (where declared), entry count,
  reviewer provenance, and direct links to the upstream
  repository. Every corpus without a declared upstream license is
  marked plainly as "License unclear" and listed as
  clarification-pending.
- **"Reviewed by Deaf native signer" badges on every translated
  word.** The metadata layer records whether each sign in the
  library was reviewed by a native signer of the target sign
  language. The badge appears inline in translator output and on
  the contribute-page preview.
- **Health, metrics, and progress snapshot endpoints.** `/health`
  reports liveness; `/metrics` exposes counters; the snapshot
  pipeline regenerates `/progress` data on a schedule.
- **Governance page at `/governance`** — standing reviewer list,
  evaluation criteria, and refusal mechanism.

## Improved

- **Contribute page design now matches the main site.** The
  brutalist contribute shell was reconciled to the warm main-page
  design system. Every JS module ID is preserved; only the visual
  frame changed. The clarification chat, preview pane, and notation
  editor all keep their previous behaviour.
- **Translator smoothness, accessibility, and mobile experience.**
  Loading bar appears only past a 200ms threshold (so fast
  translations don't flash), reduced-motion preference is honoured,
  and the visible focus ring / character counter / keyboard hints
  are all enforced. Every public route now passes axe-core with zero
  critical or serious violations and scores 100 on Lighthouse
  accessibility.
- **Unified navigation, footer, sitemap, and 404 across every
  public page.** Previously each page had its own header; now one
  header ships across `/`, `/app`, `/contribute`, `/progress`,
  `/credits`, `/governance`, and `/404`.

## Honest limitations we ship with

- **Deaf native reviewer pool is still zero in the live snapshot.**
  The dashboard reflects the truth: we have the review infrastructure
  but not yet a reviewer. The "Reviewed by a Deaf native signer"
  badge appears only on signs whose upstream corpus already included
  native-signer verification (e.g. DictaSign BSL). Community
  contributions are marked "Not yet reviewed" until that changes.
- **Landing and translator pages score below Lighthouse's desktop
  performance target (77–78 vs 85).** The bottleneck is the CWASA
  avatar bundle, a licensed third-party dependency we cannot
  modify. Lazy-loading CWASA on the landing page is a documented
  follow-up.
- **The HamNoSys notation font (`bgHamNoSysUnicode.ttf`) is
  referenced by the CSS but the binary is not yet committed.**
  When the file is missing the browser falls through to the system
  monospace font; the page never errors. Getting the binary
  shipped through IDGS's distribution terms is on the post-launch
  punch list.

## References

- Full per-prompt rollback plan: `docs/polish/14-rollback.md`
- Final lighthouse scores: `docs/polish/14-lighthouse-final.md`
- End-to-end translator smoke: `docs/polish/14-e2e-smoke.md`
- Launch decision: `docs/polish/14-launch-decision.md`
