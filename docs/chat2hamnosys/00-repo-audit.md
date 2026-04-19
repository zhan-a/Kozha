# chat2hamnosys — Repo Audit & Integration Plan (Prompt 1)

Read-only audit of the Kozha repo (commit `1a8d3cc`, branch `main`). No code was
written during this pass. Acknowledgement is required before Prompt 2 begins.

---

## 0. TL;DR — premises that need correction

The prompt's framing has four load-bearing assumptions that do not match the
repo as it stands today:

| # | Prompt assumes | Reality | Impact |
|---|---|---|---|
| A1 | "Whisper ASR → spaCy NLP → SiGML lookup → CWASA avatar" pipeline | Whisper is **not** in this repo. Speech capture (if any) happens in-browser via Web Speech / the extension; the server only does NLP → glosses. | Low — doesn't affect chat2hamnosys. Noted for accuracy. |
| A2 | "FastAPI backend with … ~1,000 BSL signs" (implying the backend owns the sign table) | The FastAPI server (`server/server.py`) never reads or writes SiGML. **Sign lookup is 100% client-side** (`public/app.html`, `extension/panel.html`). The server returns glosses; the browser fetches the per-language SiGML file, parses `<hns_sign>` nodes, and hands XML to CWASA. | **High.** See §9. A backend authoring service can still *produce* signs, but it does not fit into a serving pipeline — it either (i) writes to the shared `data/*.sigml` files that the frontend fetches, or (ii) serves a second SiGML URL that the frontend already supports loading as an array. |
| A3 | "the 1,000-sign library" (singular) | There are **15 per-language SiGML databases** under `data/`, plus six per-language `hamnosys_<lang>.csv` concept→gloss→HamNoSys sheets. BSL alone has `data/hamnosys_bsl_version1.sigml` (21,205 lines) + `data/hamnosys_bsl.csv` (1,046 rows). | Medium — chat2hamnosys must pick a target language *and* a target file per output sign, not append to "the" library. |
| A4 | `pydantic`, `httpx`, `openai`, `tiktoken` may already be installed | Only `pydantic` is transitively pulled in via `fastapi`. `openai`, `httpx`, `tiktoken` are **not** declared in `server/requirements.txt`. | Low — we'll add them. |

Secondary observations worth surfacing:

- **No tests exist.** No `tests/`, no `conftest.py`, no `pytest.ini`, no `jest.config.*`, no `package.json` at the repo root. This is a greenfield test situation; we'll need to set up pytest from scratch.
- **The `prompts/`, `logs/`, and `run_chat2hamnosys.sh` artifacts are gitignored** — they are the local driver for this 20-prompt batch. Current run: `logs/chat2hamnosys_run_20260419_044451.log`. This is prompt 1 of 20.
- **`backend/` does not exist** — the requested path `backend/chat2hamnosys/` is free. But note: the existing Python server lives under `server/`, not `backend/`. Placing new Python code under `backend/` creates a split. See §9 for the trade-off.
- **There is a brand mismatch**: `public/contribute.html` says "Bridgn" while everything else says "Kozha". Not our problem, but flag for the team.

---

## 1. Directory tree (depth 3, excluding `.git`, `.DS_Store`, `__pycache__`)

```
Kozha/
├── .gitattributes
├── .gitignore                                     # staged M — see §7
├── .github/
│   └── workflows/
│       └── deploy.yml                             # SSH deploy to EC2 on push to main
├── README.md
├── run_chat2hamnosys.sh                           # local-only, gitignored — drives prompts 1-20
├── data/                                          # all sign-language assets (static, served at /data)
│   ├── hamnosys_bsl.csv                           # 1,046 BSL concept→gloss→HamNoSys rows
│   ├── hamnosys_dgs.csv                           # 1,123 DGS rows
│   ├── hamnosys_gsl.csv                           # 860 GSL rows
│   ├── hamnosys_lsf.csv                           # 380 LSF rows
│   ├── hamnosys_ngt.csv                           # 55 NGT rows
│   ├── hamnosys_pjm.csv                           # 1,914 PJM rows
│   ├── hamnosys_bsl_version1.sigml                # 21,205 lines — BSL signs (SiGML XML)
│   ├── Algerian_SL.sigml                          # 10,535 lines
│   ├── asl_alphabet_sigml.sigml                   # 261 lines — BSL alphabet shapes
│   ├── Bangla_SL.sigml                            # 2,379 lines
│   ├── bsl_alphabet_sigml.sigml                   # 892 lines — BSL A–Z for fingerspelling
│   ├── dgs_alphabet_sigml.sigml                   # 263 lines
│   ├── Dutch_SL_NGT.sigml                         # 1,284 lines
│   ├── Filipino_SL.sigml                          # 32 lines
│   ├── French_SL_LSF.sigml                        # 9,073 lines
│   ├── German_SL_DGS.sigml                        # 34,315 lines
│   ├── Greek_SL_GSL.sigml                         # 27,058 lines
│   ├── Indian_SL.sigml                            # 16,861 lines
│   ├── Kurdish_SL.sigml                           # 9,308 lines
│   ├── lsf_alphabet_sigml.sigml                   # 264 lines
│   ├── ngt_alphabet_sigml.sigml                   # 264 lines
│   ├── pjm_alphabet_sigml.sigml                   # 258 lines
│   ├── Polish_SL_PJM.sigml                        # 36,690 lines
│   └── Vietnamese_SL.sigml                        # 113,396 lines
├── extension/                                     # Chrome extension (MV3)
│   ├── manifest.json                              # 46 lines
│   ├── background.js                              # 129 lines — service worker
│   ├── content-shared.js                          # 386 lines — shared DOM helpers
│   ├── content-universal.js                       # 289 lines — "select and sign" mode
│   ├── content-youtube.js                         # 430 lines — YouTube caption hook
│   ├── popup.html / popup.js / popup.css          # 69 / 147 / 125 lines
│   ├── panel.html / panel.css                     # 365 / 292 lines — injected sign panel
│   └── icons/ (16 / 48 / 128 px)
├── public/                                        # static, served from FastAPI root "/"
│   ├── index.html                                 # 1,563 lines — marketing landing ("Bridgn"/Kozha)
│   ├── app.html                                   # 1,726 lines — full translator UI
│   ├── contribute.html                            # 166 lines — "Contribute" page (GitHub link only)
│   ├── LICENSE
│   └── cwa/                                       # CWASA avatar bundle (third-party, CC BY-ND)
│       ├── allcsa.js
│       ├── cwacfg.json
│       ├── cwasa.css
│       └── avatars/
├── server/
│   ├── server.py                                  # 641 lines — FastAPI app (see §2)
│   ├── requirements.txt                           # 12 lines
│   └── abbreviations.json                         # 211 lines — phrase→acronym table (e.g. "artificial intelligence" → "AI")
├── prompts/                                       # gitignored; local driver (prompt-1.md … prompt-20.md)
└── logs/                                          # gitignored; driver output
```

### Top-level modules (size + one-line purpose)

| Path | LOC | Purpose |
|---|---:|---|
| `server/server.py` | 641 | FastAPI app: spaCy NLP → glosses, plus Argos translate pass-through. No SiGML awareness. |
| `server/requirements.txt` | 12 | `fastapi`, `uvicorn`, `spacy>=3.8`, 7 spaCy models, `gunicorn`, `argostranslate`. |
| `server/abbreviations.json` | 211 | Phrase → letter-sequence map for fingerspelling expansion (e.g. "united states of america" → "usa"). |
| `public/app.html` | 1,726 | Full translator UI. Calls `/api/translate`, fetches `/data/*.sigml`, drives CWASA. Contains the gloss→SiGML lookup (§3). |
| `public/index.html` | 1,563 | Marketing landing page. |
| `public/contribute.html` | 166 | Static "Contribute on GitHub" page. **No** server-side submission endpoint. |
| `public/cwa/allcsa.js` | — | CWASA avatar engine (third-party, vendored). Entry for `CWASA.playSiGMLText(xml, 0)`. |
| `extension/*` | ~2,300 | Chrome MV3 extension with popup, context-menu, and YouTube modes. Mirrors the app's gloss→SiGML loader. |
| `data/*.sigml` | 284k total | Per-language `<hns_sign gloss="…">…</hns_sign>` XML files. Statically served at `/data/`. |
| `data/hamnosys_*.csv` | 5,384 total | Parallel per-language concept→gloss→HamNoSys sheets. Used to translate free-text concepts to the canonical gloss key. |
| `.github/workflows/deploy.yml` | 40 | `git reset --hard origin/main` + `pip install -r requirements.txt` + `systemctl restart kozha.service` on EC2. |
| `run_chat2hamnosys.sh` | 73 | Local driver: iterates `prompts/prompt-{1..20}.md` through `claude -p … --dangerously-skip-permissions`. Gitignored. |

---

## 2. Where the requested subsystems live today

| Subsystem | File | Lines | Notes |
|---|---|---|---|
| FastAPI entry point | `server/server.py` | 54 (`app = FastAPI(...)`), routes at 599–635 | Routes: `GET /api/health`, `POST /api/translate`, `POST /api/translate-text`, `POST /api/translate/batch`, `POST /api/plan`. Plus `StaticFiles` mounts for `/data/` (634–635) and `/` (641). |
| "1,000-sign library" | `data/hamnosys_bsl_version1.sigml` + `data/hamnosys_bsl.csv` | 21,205 / 1,046 | See A3 above. **Per-language**, not a singleton. |
| OOV fingerspelling fallback | `public/app.html` | 1438 (`fingerspellWord`), 1445 (call site), 1202 (`FALLBACK_ALPHABET`) + `extension/panel.html:184` | Happens in the **browser**, not the server. Uses the per-language `*_alphabet_sigml.sigml` loaded into `letterToSign`. |
| SiGML rendering path | `public/app.html` (1451 `playSigml` → `CWASA.playSiGMLText`), `public/cwa/allcsa.js` | — | CWASA avatar engine. Consumes a full `<sigml>…</sigml>` document. |
| Frontend "chat" UI | **Does not exist.** | — | `public/app.html` is a single-input translator, not a chat. The extension also isn't a chat. If chat2hamnosys needs a chat UI, we must build one. |
| Existing LLM / OpenAI calls | **None.** | — | The only keyword hit is `.anthropic/` in `.gitignore:17`. No `openai`, `tiktoken`, `anthropic`, or LLM HTTP call anywhere in the code. |

---

## 3. Every place that reads or writes SiGML / HamNoSys

All reads; the repo has **no writer**. (The `.sigml` files are checked in as static assets.)

| File | Lines | What it does |
|---|---|---|
| `public/app.html` | 22 | `<link rel="preload" href="/data/hamnosys_bsl_version1.sigml" as="fetch">` — BSL preload hint. |
| `public/app.html` | 772 | Manual-URL input placeholder (users can point at any SiGML URL). |
| `public/app.html` | 1081–1098 (`loadSigmlUrl`) | `fetch` → `DOMParser` → `querySelectorAll('hns_sign')` → populate `glossToSign: Map<string, outerHTML>`. |
| `public/app.html` | 1106–1120 (`loadAlphabetSigmlUrl`) | Same pattern, but filters to single A–Z glosses → `letterToSign`. |
| `public/app.html` | 1174–1190 (`SIGN_LANG_DB`) | Per-language config: `{ sigml: [urls], csv: url, alphabet: url }`. Note `sigml` is **already an array**, so we can add a second URL without code changes (see §9). |
| `public/app.html` | 1348–1364 (`mapTokensToGlosses`) | Maps NLP tokens to loaded glosses; OOV → fingerspell or `missing`. |
| `public/app.html` | 1438, 1445, 1448 (`fingerspellWord`, `buildSigml`) | Assembles outgoing SiGML by joining loaded `<hns_sign>` HTML fragments and wrapping in `<?xml…?><sigml>…</sigml>`. |
| `public/app.html` | 1451 (`playSigml`) | `CWASA.playSiGMLText(sigml, 0)` — the final handoff. |
| `public/app.html` | 1529, 1536, 1549 | Play-pipeline entry in the translate handler. |
| `extension/panel.html` | 54–66 | Extension's `SIGN_LANG_DB` — **mirrored** from `public/app.html`. |
| `extension/panel.html` | 82, 95, 146, 160 | `querySelectorAll('hns_sign')` loader (same shape as the app's). |
| `extension/panel.html` | 184, 201 (`fingerspellWord`) | Extension's fingerspelling fallback. |
| `extension/panel.html` | 206, 247, 251 | SiGML document assembly (same `<?xml…?><sigml>…</sigml>` pattern). |
| `extension/panel.html` | 226, 239, 278, 328–331 | `CWASA.playSiGMLText(...)` handoffs and `play_sigml` postMessage receiver. |
| `public/cwa/allcsa.js` | — (vendored) | CWASA engine — consumes SiGML. Do not modify. |
| `public/cwa/cwasa.css` / `cwacfg.json` | — | CWASA config/style. |
| `README.md` | §"How It Works" | Documents the pipeline. |

**Write path (today): none in code.** New `.sigml` files are added by hand and committed.

---

## 4. Dependencies (from `server/requirements.txt`)

```
fastapi
uvicorn[standard]
spacy>=3.8.0
en-core-web-sm @ <url>
de-core-news-sm @ <url>
fr-core-news-sm @ <url>
es-core-news-sm @ <url>
pl-core-news-sm @ <url>
nl-core-news-sm @ <url>
el-core-news-sm @ <url>
gunicorn
argostranslate
```

| Dep | Declared? | Status |
|---|---|---|
| `openai` | ❌ | Must add. |
| `pydantic` | ❌ declared; ✅ transitive via `fastapi` | Already available at import time. Safe to `from pydantic import BaseModel`. `server.py:5` already does this. |
| `httpx` | ❌ | Not required by `openai>=1.x` directly (it ships its own async client), but useful for retries/streaming tooling. Add only if we hit a need. |
| `tiktoken` | ❌ | Only needed if we want local token-count estimation. Defer unless a budget-gate task demands it. |

No `pyproject.toml`. No `package.json`. No frontend build step — everything in `public/` is hand-written HTML/JS/CSS.

---

## 5. Tests

**None.** No `tests/`, no `conftest.py`, no `pytest.ini`, no `pyproject.toml`,
no `*.test.*`, no fixture directories. The deploy workflow at
`.github/workflows/deploy.yml` does not run tests either — it just does
`git reset --hard` + `pip install` + `systemctl restart`.

Chat2hamnosys is the first code to get a test suite. Proposal: **pytest**, with
fixtures under `backend/chat2hamnosys/tests/fixtures/` (or wherever §9 lands).
Mock the OpenAI client at the SDK boundary, not via `httpx` interception.

---

## 6. Data format — sign entry, verbatim

### 6a. `data/hamnosys_bsl.csv` (concept sheet)

Header:

```
concept,language,gloss,hamnosys,video_url,page_url
```

One real row (line 3), shown as Python `repr` to reveal the HamNoSys byte
sequence — these are Unicode HamNoSys glyphs in the PUA range
(U+E000…U+EFFF) that render as boxes in this terminal but display correctly
in the HamNoSys font used by the frontend:

```text
accept,BSL,accept(v)#2,î\x83©î\x80\x85î\x80\x8eî\x80¨î\x83¦î\x80©î\x80¸î\x83¢î\x82\x8bî\x82ªî\x80\x80î\x80¸î\x82\x84î\x83£î\x81\x94î\x81\x99î\x83\x91,,https://www.sign-lang.uni-hamburg.de/dicta-sign/portal/concepts/cs/cs_3.html
```

- `concept` — English lowercase lemma (the lookup key).
- `language` — `BSL`, `DGS`, etc.
- `gloss` — canonical gloss in `<word>(<pos>)#<variant>` form, e.g. `accept(v)#2`.
  This gloss is what ties a CSV row to its SiGML entry.
- `hamnosys` — raw HamNoSys characters (PUA-encoded).
- `video_url` — optional reference video (empty here).
- `page_url` — DictaSign portal citation.

### 6b. `data/hamnosys_bsl_version1.sigml` — the actual signable entry

This is the format chat2hamnosys **must produce**. Taken verbatim from lines 2–19:

```xml
<sigml>
	<hns_sign gloss="abroad(a)#1">
		<hamnosys_nonmanual />
		<hamnosys_manual>
			<hamflathand />
			<hamthumboutmod />
			<hamfingerbendmod />
			<hambetween />
			<hamflathand />
			<hamthumboutmod />
			<hamextfingeru />
			<hampalmd />
			<hamhead />
			<hamlrat />
			<hammoveo />
			<hamsmallmod />
			<hamrepeatfromstart />
		</hamnosys_manual>
	</hns_sign>
	<!-- … more <hns_sign>… -->
</sigml>
```

Shape contract:

- Root `<sigml>` wraps an unbounded sequence of `<hns_sign>` children.
- `<hns_sign gloss="…">` — gloss attribute is the key the frontend looks up
  (`trim().toLowerCase()`; see `app.html:1092`).
- `<hamnosys_nonmanual>` — non-manual markers (optional `<hnm_mouthpicture picture="…"/>` children for lip shapes; see `bsl_alphabet_sigml.sigml:5–7`).
- `<hamnosys_manual>` — ordered HamNoSys tags. The vocabulary is the CWASA
  SiGML/HamNoSys tag set (`<hamflathand/>`, `<hampalmd/>`, `<hamparbegin/>`,
  `<hamparend/>`, `<hamplus/>`, `<hambetween/>`, `<hamreplace/>`,
  `<hamtouch/>`, `<hamfinger2/>` … `<hamfinger2345/>`, location tags like
  `<hamchest/>`, `<hamhead/>`, `<hamshoulders/>`, movement tags like
  `<hammoveo/>`, `<hammoved/>`, `<hammovel/>`, `<hamsmallmod/>`, repeat tags
  like `<hamrepeatfromstart/>`, symmetry like `<hamsymmlr/>`, etc.). Tags are
  self-closing. No attributes on the `ham*` tags themselves (except the
  `hnm_mouthpicture` exception).

One more example with the full structure (from `bsl_alphabet_sigml.sigml:4–42`)
showing `hamparbegin`/`hamparend` grouping and a non-manual mouth picture:

```xml
<hns_sign gloss="A">
    <hamnosys_nonmanual>
        <hnm_mouthpicture picture="V"/>
    </hamnosys_nonmanual>
    <hamnosys_manual>
        <hamparbegin/>
        <hamfinger2/>
        <hamthumbacrossmod/>
        <hamplus/>
        <hamfinger2345/>
        <hamthumboutmod/>
        <hambetween/>
        <hamflathand/>
        <hamthumboutmod/>
        <hamparend/>
        <hamparbegin/>
        <hamextfingerul/>
        <hampalmdl/>
        <hamplus/>
        <hamextfingeror/>
        <hampalmd/>
        <hambetween/>
        <hampalmdr/>
        <hamparend/>
        <hamparbegin/>
        <hamthumb/>
        <hamindexfinger/>
        <hamfingertip/>
        <hamplus/>
        <hamindexfinger/>
        <hamfingertip/>
        <hamparend/>
        <hamtouch/>
        <hamlrat/>
        <hamchest/>
        <hambetween/>
        <hamchest/>
        <hamclose/>
    </hamnosys_manual>
</hns_sign>
```

Key semantics the authoring layer must respect:

1. **The `gloss` attribute is the lookup key.** Two entries with the same `gloss` collide — `Map.set` in `app.html:1092` keeps the **last** loaded one. If we want authored signs to win, load their SiGML **after** the upstream file.
2. **The frontend lowercases on lookup** (`app.html:1092`), so `gloss="HELLO"` and `gloss="hello"` are equivalent. Follow the upstream convention: lowercase lemma + `(pos)#variant`, e.g. `hello(v)#1`.
3. **Alphabet letters are single uppercase chars** (`A`–`Z`) and are routed to `letterToSign`, not `glossToSign` (`app.html:1117`).
4. **Tag ordering matters.** HamNoSys is a sequence, not a tree. Authoring code must emit tags in the order prescribed by the HamNoSys 2018 spec (`README` links it).

---

## 7. Current repo state (for transparency)

- `git status`: `.gitignore` is staged-modified. The current `.gitignore` adds `prompts/`, `logs/`, `run_*.sh`, `.claude/`, `.anthropic/`, `.env*`, `*.key`, `*.pem`, `.DS_Store`, editor noise — see `.gitignore:1–31`.
- HEAD: `1a8d3cc — credit all sign language database sources with repo links`.
- Last five commits are all documentation/attribution — no code churn recently.
- The `run_chat2hamnosys.sh` script loops `prompts/prompt-{1..20}.md` through `claude -p --model opus --effort max --dangerously-skip-permissions` and tees each to `logs/p-<i>_<ts>.log`. It also records commits-per-prompt via `git rev-parse`. So **each prompt is expected to land a commit**.

---

## 8. Prompt-1 scope check

The prompt explicitly forbids code changes in this pass and asks the plan to be
acknowledged before proceeding. I'm honoring both: only `docs/chat2hamnosys/00-repo-audit.md`
is new on disk after this pass. No other files touched. No commit yet —
awaiting acknowledgement.

---

## 9. Integration plan for `chat2hamnosys`

### 9a. Goal (as I read it)

Add a backend subsystem that authors new SiGML sign entries via an LLM and
persists them into the existing per-language SiGML store, **without breaking
the live translation pipeline**. The primary audience is the repo operator
(and possibly a future "suggest a sign" UI); the output is consumed by the
same frontend loaders that already exist.

### 9b. Placement — where to put the code

The prompt says `backend/chat2hamnosys/`. There's a split to flag:

- Existing Python lives under `server/`. The deploy script pins `cd server`
  and `pip install -r requirements.txt` (`deploy.yml:33–35`). If we put new
  code under `backend/`, deploy won't pick it up unless we also update the
  workflow and create `backend/requirements.txt`.
- Easiest with least disruption: **`server/chat2hamnosys/`**, share the
  existing `requirements.txt`, and mount the router from `server/server.py`
  with `app.include_router(chat2hamnosys.router)`.

**Proposal:** use `server/chat2hamnosys/` (not `backend/chat2hamnosys/`).
If you prefer `backend/`, I'll also update `deploy.yml` in prompt 2 so
deployment still works — but it costs one extra file and a CI path change.
**Please pick one before prompt 2.**

### 9c. Module layout (working sketch)

```
server/chat2hamnosys/
├── __init__.py
├── router.py             # FastAPI APIRouter mounted at /api/chat2hamnosys
├── schemas.py            # pydantic request/response models
├── openai_client.py      # lazy singleton; reads OPENAI_API_KEY; model pinned
├── prompts.py            # system prompt(s) + few-shot examples extracted from hamnosys_bsl_version1.sigml
├── tools.py              # tool-calling schemas (HamNoSys tag set, validators)
├── sigml_writer.py       # merges new <hns_sign> into data/hamnosys_<lang>_authored.sigml (see §9e)
├── validators.py         # SiGML well-formedness, tag-vocab allow-list, gloss uniqueness check
└── tests/
    ├── conftest.py
    ├── fixtures/
    │   ├── sample_response.json
    │   └── golden_bsl_hello.sigml
    ├── test_validators.py
    ├── test_sigml_writer.py
    └── test_router.py    # uses FastAPI TestClient + a mocked OpenAI client
```

### 9d. HTTP contract

`POST /api/chat2hamnosys/author`

```jsonc
// request
{
  "concept": "videoconference",          // English lemma, required
  "sign_language": "bsl",                // required; one of the SIGN_LANG_DB keys in app.html:1174
  "pos": "n",                            // noun/verb/adj/... optional
  "variant": 1,                          // optional; defaults to next free #N
  "hints": "right hand mimics screen, left hand gestures at face",  // optional free text
  "persist": false                       // if true, append to the authored SiGML file
}

// response (200)
{
  "gloss": "videoconference(n)#1",
  "sigml": "<hns_sign gloss=\"videoconference(n)#1\">…</hns_sign>",
  "hamnosys_tags": ["hamflathand","hamextfingeru","…"],
  "validated": true,
  "persisted_to": "data/hamnosys_bsl_authored.sigml"   // only when persist=true
}
```

Additional endpoints to consider (low-cost additions):

- `GET /api/chat2hamnosys/allowed-tags` — returns the HamNoSys tag allow-list that the LLM/validator uses. Useful for debugging.
- `POST /api/chat2hamnosys/validate` — accepts a SiGML fragment, returns pass/fail + reasons. No LLM call. Useful for CI.
- `GET /api/chat2hamnosys/health` — basic liveness.

### 9e. Persistence strategy — **the critical design choice**

The frontend's `SIGN_LANG_DB` already takes an **array** of SiGML URLs
(`public/app.html:1174–1190`) and the loader at line 1215 iterates. So we do
**not** have to mutate the upstream vendor SiGML files (e.g.
`hamnosys_bsl_version1.sigml`, which has license constraints from the
DictaSign corpus).

Recommended approach:

1. Write authored entries to a **new file per language**:
   `data/hamnosys_<lang>_authored.sigml` (e.g. `data/hamnosys_bsl_authored.sigml`).
2. In `public/app.html:1174` and `extension/panel.html:54`, **append** that URL
   to the `sigml: []` array so the frontend loads both:
   ```js
   bsl: { sigml: ['/data/hamnosys_bsl_version1.sigml',
                   '/data/hamnosys_bsl_authored.sigml'], … }
   ```
   Because the authored URL is listed **second** and the loader does `Map.set`
   per entry, authored signs **override** upstream signs on gloss collision
   (`app.html:1092`). If we want the opposite, flip the order.
3. Never edit `data/hamnosys_bsl_version1.sigml` (DictaSign CC BY-NC-SA 3.0
   attribution concerns — see README credits).

This is a two-line change to the frontend config and zero change to the
loader logic. The existing pipeline is untouched.

**Alternative considered and rejected:** patching the upstream SiGML file
in-place. Rejected because (a) license attribution pressure, (b) merge-pain
against upstream refreshes, (c) harder to roll back an LLM mistake.

### 9f. LLM approach (chat completions + tool-calling)

- SDK: `openai>=1.50` (new unified client). Add to `server/requirements.txt`.
- Model: pin to a specific snapshot (e.g. `gpt-4o-2024-08-06` or newer equivalent) via an env var with a default. **Do not default to a moving "latest" alias.**
- Mechanism: tool-calling with a single tool `emit_sign`, whose JSON schema mirrors the HamNoSys tag grammar (ordered list of tag names, each one an `enum` drawn from the validated tag allow-list).
- Why tool-calling instead of free text: forces the model to emit a structured list of tags, so the validator only has to check tag membership + ordering rules, not arbitrary XML.
- Few-shot: pack 3–5 real `<hns_sign>` entries from `hamnosys_bsl_version1.sigml` into the system prompt. Rotate them per request to avoid overfitting.
- Validation before return (always on):
  1. Every emitted tag ∈ allow-list built from a scan of the existing SiGML corpora.
  2. `hamparbegin` / `hamparend` balanced.
  3. Resulting `<hns_sign>` parses as XML.
  4. Gloss doesn't already exist in the *authored* file (upstream collision is allowed — it becomes an override).
- Secrets: `OPENAI_API_KEY` read from env; never logged; never returned in responses.

### 9g. Testing plan (pytest, because there is nothing today)

- `test_validators.py` — tag allow-list, paren balance, gloss format regex.
- `test_sigml_writer.py` — append, idempotent re-append (dedupe by gloss), atomic write via tmpfile + `os.replace`, no corruption on partial write.
- `test_router.py` — FastAPI `TestClient`; `openai.OpenAI` patched with a fake that returns a canned tool-call response; asserts 200 + schema. A second test asserts 422 on invalid input.
- Golden files under `tests/fixtures/`. Keep them under 50 lines each.

### 9h. Guardrails / "don't break the pipeline"

- **No imports from `chat2hamnosys` into `server.py`'s existing code paths.** The router is mounted; existing code keeps working verbatim.
- **Failure isolation:** if `OPENAI_API_KEY` is missing, the router returns 503 on its own endpoints but the rest of the app is unaffected. The import-level code must not crash on missing env.
- **Frontend config change is one line per language in two files.** We can gate it behind a `?chat2hamnosys=1` query param if the team wants to A/B it.
- **No changes to `deploy.yml`** unless we move to `backend/`.

### 9i. Open questions (please answer before prompt 2)

1. **Path:** `server/chat2hamnosys/` (no deploy change) or `backend/chat2hamnosys/` (deploy change + split Python layout)?
2. **Target languages:** BSL only for v1, or all 15 from the start? (Recommend BSL only; broaden once the loop works.)
3. **Auth on `/api/chat2hamnosys/author`:** open, header token, or Basic? Today the rest of `/api/*` is unauth'd and CORS `*`. Authoring is more sensitive (it spends money and mutates `data/`).
4. **Model:** is there a preferred OpenAI model and a budget cap we should enforce per request (max_tokens, monthly $ limit)?
5. **Persistence file naming:** `data/hamnosys_bsl_authored.sigml` okay, or do you want `data/chat2hamnosys_bsl.sigml`?
6. **Frontend wiring:** add the authored URL to `SIGN_LANG_DB` now (in prompt 2) or keep the backend self-contained for several prompts first?

---

## 10. What prompt 2 would do (sketch, pending acknowledgement)

If the above plan is accepted:

1. Add `openai` to `server/requirements.txt`.
2. Create `server/chat2hamnosys/` with `router.py`, `schemas.py`, stub `openai_client.py`.
3. Mount the router in `server/server.py` with two new lines (`from server.chat2hamnosys.router import router as chat2hamnosys_router` + `app.include_router(...)`).
4. Add `GET /api/chat2hamnosys/health` and `GET /api/chat2hamnosys/allowed-tags` only — no OpenAI call yet.
5. Land the pytest scaffolding (`pytest.ini`, `conftest.py`, one smoke test for `/health`).
6. Commit with a title like `chat2hamnosys: scaffold router and health check (no-op for existing pipeline)`.

That scopes prompt 2 to pure additions with no behavior change, and gives
later prompts a clean surface to add validators, the OpenAI call, and finally
the persistence + frontend wiring.

---

**Awaiting acknowledgement** on §9i before proceeding.
