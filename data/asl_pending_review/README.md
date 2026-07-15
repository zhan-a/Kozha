# ASL signs held pending verification (NOT served, NOT counted)

The live translator serves only `data/American_SL_ASL.sigml` — **14 signs** that
were hand-authored, cross-checked against native-signer video, and confirmed to
render in the avatar. Everything below is parked here until each sign is
individually verified the same way.

- **`American_SL_ASL_unverified_curated.sigml`** — 21 hand-authored candidates
  that were **not** individually verified. Some fail on inspection: e.g. DRINK
  hung the CWASA renderer in the Jun render-verify pass, and BIG / COME were
  never render- or video-checked and use simplified movement/orientation.
- **`American_SL_ASL_unverified.sigml`** — 1,656 signs machine-converted from
  the [ASL-LEX 2.0](https://asl-lex.org) phonology. Handshape/location are
  faithful to ASL-LEX, but palm/finger orientation and movement direction are
  rule-derived heuristics ASL-LEX does not measure — unverified. Regenerate any
  time with `scripts/asl_lex_to_sigml.py` (a regeneration re-emits `thank_you`,
  which was removed here as redundant with the served, verified THANK-YOU).

To promote a sign to the served set: verify it against native-signer video,
render-check it (`scripts/render_verify_asl.mjs`), then move it into
`data/American_SL_ASL.sigml`.
