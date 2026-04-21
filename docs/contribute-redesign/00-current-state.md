# Current state — contribution surface

Snapshot: 2026-04-20, branch `main` @ `95c9700`. Read-only audit. No design
proposals here — just what exists.

---

## 1. Entry points visible to a prospective contributor

### `public/contribute.html` (691 lines)

The sole public landing page for contribution. Reachable from:

- `public/index.html:750` — nav link "Contribute"
- `public/index.html:753` — button "Volunteer"
- `public/index.html:965` — "Get involved" CTA inside the "Contribute to
  the project." banner
- `public/index.html:975` — footer nav link
- `public/app.html:604` — translator top-nav link
- `public/contribute.html:310` — self-link in its own nav

Structure, top-down:

1. `<nav>` (lines 304-313) — Bridgn logo, home / features / how-it-works
   links, active "Contribute" tab, "Open Translator →" button.
2. `<section class="page-hero">` (lines 316-334) — eyebrow "Open-source
   · Community-built", h1 "Help grow the sign dictionary.", paragraph
   explaining the Deaf-reviewed flow, two CTAs (`#contribute` and
   GitHub).
3. `<section class="why">` (lines 337-367) — three cards: "Signs reach
   real users", "Deaf-led review", "Under five minutes".
4. `<section>` (lines 370-400) — "Four steps to a published sign":
   Register, Describe, Preview & correct, Submit for review.
5. `<section id="contribute" class="signup">` (lines 403-469) — the
   form card.
6. `<section>` (lines 472-497) — 4-item FAQ.
7. `<footer>` (lines 500-511) — logo, link list, copyright.
8. `<script>` IIFE (lines 513-688) — captcha load, form submit, BYO
   key handling, post-success redirect.

#### Form behaviour (`contribute.html:413-466`)

- Fields: `name`, `contact` (email or phone — free text), hidden
  honeypot `website`, `captchaAnswer`, optional `openaiKey`.
- `GET /api/chat2hamnosys/contribute/captcha` on boot and on
  refresh click (lines 584-615). If response is `{disabled: true}`,
  hides the captcha row entirely.
- `POST /api/chat2hamnosys/contribute/register` on submit (lines
  644-686). Payload:
  `{name, contact, captcha_challenge, captcha_answer, website}`.
  **Note**: the BE model `RegisterIn` in
  `backend/chat2hamnosys/api/contributors.py:327-343` does not
  currently accept `contact_channel`, `accept_coc`, or
  `sign_languages` — the contributor does not declare which
  sign language they are contributing for at registration time.
- On success, stores in `localStorage`:
  - `bridgn.contributor_token`
  - `bridgn.contributor_expires_at`
  - `bridgn.contributor_profile` (`{id, name}`)
  - `bridgn.openai_api_key` (if the user entered one)

  Then redirects to `/chat2hamnosys/` after 700ms.
- On page load, if a non-expired token already exists, auto-redirects
  to `/chat2hamnosys/` (lines 566-573).

#### BYO OpenAI key panel (`contribute.html:446-457`)

Always visible, regardless of whether the server has
`OPENAI_API_KEY` configured. Comment at lines 559-563 notes this is
intentional pre-launch — contributors can swap to a personal key at
any time. The same `localStorage` value is read by the authoring UI
(`/chat2hamnosys/app.js` sends it as `X-OpenAI-Api-Key`).

---

## 2. Authoring UI — `public/chat2hamnosys/index.html` (292 lines)

Three-panel layout, loaded post-registration.

- **Chat panel** (left): gloss input (required, used as entry key),
  freeform description textarea, Send / Accept / Reject buttons,
  session-state subtitle.
- **Avatar preview** (right top): SiGML playback via `<video>` with
  CWASA fallback, clickable SVG region overlay, keyboard region picker,
  timeline slider, "Flag this moment" button, region popover for
  submitting targeted corrections.
- **Parameter inspector** (right bottom): list of slots with
  confirmed / inferred / gap badges.
- Mobile tabs (`chat` / `preview` / `inspect`) to swap panels on
  narrow viewports.
- Loads `/chat2hamnosys/avatar/region_map.js` and
  `/chat2hamnosys/app.js`; imports CWASA via `/cwa/cwasa.css` and
  `/cwa/allcsa.js`.

No sign-language picker in the authoring UI — the create-session
call defaults `sign_language` to `"bsl"` unless the client sends one
of `bsl|asl|dgs`.

---

## 3. Reviewer console — `public/chat2hamnosys/review/index.html` (246 lines)

Not surfaced from the main site. Reachable only by direct URL.

- **Sign-in gate**: bearer token pasted into a password field and
  stored in `localStorage`. Pre-seeded reviewer tokens only — there
  is no contributor-visible path to become a reviewer.
- **Queue**: filters by sign language (bsl/asl/dgs only, hard-coded
  in the `<select>` at lines 73-75), region text input,
  include-quarantined checkbox.
- **Detail panel**: gloss, status pill, language / region / domain /
  approval count, description, HamNoSys `<pre>`, reviewer history
  list, action row.
- **Actions**: Approve, Reject, Request revision, Flag (quarantine),
  Clear quarantine (board-only), Export to library (board-only).
- **Action dialog** (`<dialog id="actionDialog">`): comment
  textarea, reject category radios, clear-target radios,
  non-native justification textarea + checkbox.
- Fatigue meter shows `N / 25 reviewed this session`.

---

## 4. Backend — `backend/chat2hamnosys/`

### Sub-app mount — `server/server.py:642-647`

```python
try:
    from api import create_app as _create_chat2hamnosys_app
    _chat2hamnosys_sub_app = _create_chat2hamnosys_app()
    app.mount("/api/chat2hamnosys", _chat2hamnosys_sub_app)
except Exception as _c2h_mount_err:
    logger.warning("chat2hamnosys API not mounted: %s", _c2h_mount_err)
```

The sub-app is wrapped in a `try` — if import fails the translator
site still runs, but the contribute form will get 404s. The
mount is present in the current main branch; the chat2hamnosys
surface is reachable from live.

### `api/app.py` — `create_app()`

FastAPI sub-app assembled from four routers:

- `session_router` (from `api/router.py`) — no prefix, so endpoints
  land at `/api/chat2hamnosys/sessions*`.
- `contributors_router` (from `api/contributors.py`) — prefix
  `/contribute`.
- `admin_router` (from `api/admin.py`) — no prefix for `/metrics`
  and `/health*`; `/admin` prefix is in the paths.
- `review_router` (from `review/router.py`) — prefix `/review`.

Includes CORS, SlowAPI rate limiting, and
`_ByoOpenAIKeyMiddleware` which reads the `X-OpenAI-Api-Key`
header and threads it into a contextvar for the generator layer.

### Sign-language restriction — a single literal

`backend/chat2hamnosys/api/models.py:37`
`backend/chat2hamnosys/review/models.py:32`
`backend/chat2hamnosys/eval/models.py:68`

All three declare `Literal["bsl", "asl", "dgs"]`. Any other value
422s at Pydantic validation, which blocks the five additional
languages the FAQ advertises (LSF/LSE/PJM/NGT/GSL).

### Session state machine — `session/state.py`

`SessionState` enum: `AWAITING_DESCRIPTION → PARSING → CLARIFYING →
GENERATING → RENDERED → AWAITING_CORRECTION → APPLYING_CORRECTION →
FINALIZED | ABANDONED`.
Terminals: `FINALIZED`, `ABANDONED`.
Timeouts: `INACTIVITY_TIMEOUT = 24h`, `RETENTION_WINDOW = 30d`.

Orchestrator (`session/orchestrator.py`) exposes pure transitions
invoked by the router — `start_session`, `on_description`,
`on_clarification_answer`, `run_generation`, `on_correction`,
`apply_correction`, `on_accept`, `on_reject`, `check_timeout`.

### Review policy — `review/policy.py`

Defaults:

- `min_approvals = 2`
- `require_native_deaf = True` (at least one of the 2 approvals
  must be from a Deaf native signer, unless the non-native path is
  taken with a justification)
- `allow_single_approval = False`
- `fatigue_threshold = 25` (reviewer sign-offs per 24h before the
  UI warns)
- `require_region_match = True` (reviewer's regional background
  must match the sign's `regional_variant`)

### Storage — `storage.py`

SQLite stores for sessions, sign drafts / finalised entries,
contributor tokens, reviewer roster, and the export audit log.
`SQLiteSignStore.export_to_kozha_library()` writes
`data/hamnosys_<lang>_authored.sigml`. Two export gates:
`status == "validated"` AND
`qualifying_approval_count >= policy.effective_min_approvals()`.

Store locations are overridable via env vars:
`CHAT2HAMNOSYS_SESSION_DB`, `_SIGN_DB`, `_TOKEN_DB`, `_DATA_DIR`.
Default root: `data/chat2hamnosys/`.

### Contributor persistence

`data/chat2hamnosys/contributors.sqlite3` (default path). The
`contributors` table holds `id, name, contact, created_at`.
Tokens are hashed on disk and returned in plaintext once on
register.

### Pre-launch env flags

- `CHAT2HAMNOSYS_REQUIRE_CONTRIBUTOR` — default `"0"`. When `"1"`,
  session endpoints demand a valid contributor bearer. Gate is OFF
  today.
- `CHAT2HAMNOSYS_CAPTCHA_DISABLED` — when `"1"`, the captcha
  endpoint returns `{disabled: true}` and register accepts any
  answer (honeypot still enforced).
- `OPENAI_API_KEY` — optional. Recent commits thread
  `X-OpenAI-Api-Key` through to the generator so a contributor's
  personal key takes precedence when both are set.

---

## 5. Data layer — `data/`

See [00-language-coverage.md](00-language-coverage.md) for full
counts. Summary:

- 12 corpora SiGML files (11 with `<hns_sign>` format; Filipino
  uses an older `<sign_manual>` format).
- 6 alphabet SiGML files (A-Z fingerspelling each).
- 6 HamNoSys CSVs.
- **No authored-sign outputs exist yet**:
  `data/hamnosys_*_authored.sigml`, `data/authored_signs/`, and
  `data/chat2hamnosys/` are all absent at audit time.

---

## 6. Browser extension — `extension/`

Files: `background.js`, `content-shared.js`, `content-universal.js`,
`content-youtube.js`, `icons/`, `manifest.json`, `panel.css`,
`panel.html`, `popup.css`, `popup.html`, `popup.js`.

Grep for `contribut|authoring|register|chat2hamnosys` returns zero
matches in `extension/`. **The extension has no contribution
affordance.** Users signing on video pages via the extension have no
in-context way to learn the dictionary can be extended.

---

## 7. Flagged issues

1. **Language-coverage mismatch.** `contribute.html:489` advertises
   eight sign languages (BSL, DGS, ASL, LSF, LSE, PJM, NGT, GSL).
   The API accepts only three (`bsl|asl|dgs`, enforced in
   `api/models.py:37`, `review/models.py:32`, `eval/models.py:68`).
   A contributor acting on the FAQ promise will find that four of
   the listed languages cannot be chosen in the authoring UI and
   the fifth (LSE) has no corpus in `data/`.

2. **No language selector in the authoring UI.** The authoring
   page does not let the contributor pick a sign language — the
   create-session call falls back to `"bsl"`. Contributors
   intending to author for ASL or DGS have no visible way to do
   so from the current frontend.

3. **No contributor path to reviewer.** The reviewer console
   requires a bearer token that can only be minted via
   `POST /review/reviewers` by an existing board reviewer.
   `contribute.html` describes a Deaf-reviewed pipeline but has
   no "become a reviewer" flow.

4. **BYO-key panel always visible.** The panel ships visible even
   after `OPENAI_API_KEY` is provisioned on the server. Pre-launch
   scaffolding is intentional, but post-launch this is a visible
   "your key goes here" surface that most contributors will be
   confused by.

5. **Extension has no contribution link.** The browser extension
   is the main way users encounter Bridgn. It does not mention
   contribution, does not link to `/contribute`, and does not
   link to `/chat2hamnosys/`.

6. **Filename-convention divergence.** The Prompt 1 brief asks
   about `hamnosys_<lang>_*.sigml`, but:
   - Upstream corpora use `<Language>_SL[_code].sigml`
   - Alphabets use `<code>_alphabet_sigml.sigml`
   - Only the legacy `hamnosys_bsl_version1.sigml` matches the
     `hamnosys_<lang>_*` pattern today
   - The authoring export target `hamnosys_<lang>_authored.sigml`
     is the *write* side, and no such file has been produced yet
   The pattern is a write convention that hasn't produced output;
   the existing read-side files follow a different convention.

7. **Contribute form never asks for sign language.** The
   registration payload carries only name, contact, captcha, and
   honeypot. There is no structured "I sign BSL, I sign ASL"
   question. Reviewer filter by sign language therefore can't use
   contributor-declared competence at all.
