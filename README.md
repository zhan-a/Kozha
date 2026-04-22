# Kozha — Speech-to-Sign Language Translation

**Kozha** is an open-source pipeline that translates spoken language into 3D sign language animations. It bridges the communication gap between hearing and deaf communities by converting speech input into animated sign language output via a linguistically grounded intermediate representation.

🌐 **Live Demo:** [kozha-translate.com](https://kozha-translate.com)

---

## Problem

Over 70 million people worldwide use sign language as their primary language, yet accessible real-time translation tools remain scarce. Most existing solutions rely on pre-recorded video dictionaries with limited vocabulary and no grammatical flexibility. Қожа takes a different approach: it generates sign language dynamically from text using a formal notation system, enabling grammatically correct and extensible output.

## How It Works

Kozha uses a multi-stage pipeline:

```
Speech → Text → NLP Processing → HamNoSys → SiGML → 3D Animation
```

1. **Speech-to-Text** — Audio input is transcribed into text.
2. **NLP Processing (spaCy)** — The text is parsed, tokenized, and mapped to sign-language-compatible grammatical structures.
3. **HamNoSys Encoding** — Parsed tokens are converted into [Hamburg Notation System (HamNoSys)](https://www.sign-lang.uni-hamburg.de/dgs-korpus/files/inhalt_pdf/HamNoSys_2018.pdf), a phonetic transcription system for sign languages.
4. **SiGML Generation** — HamNoSys representations are serialized into [Signing Gesture Markup Language (SiGML)](http://vh.cmp.uea.ac.uk/index.php/SiGML), an XML-based format.
5. **3D Avatar Rendering (CWASA)** — SiGML is rendered as 3D sign language animation using the [CWASA](http://vh.cmp.uea.ac.uk/index.php/CWA) avatar system.

## Multilingual NLP

The backend supports native NLP processing in 7 languages via dedicated spaCy models:

| Language | spaCy Model |
|---|---|
| English | `en_core_web_sm` |
| German | `de_core_news_sm` |
| French | `fr_core_news_sm` |
| Spanish | `es_core_news_sm` |
| Polish | `pl_core_news_sm` |
| Dutch | `nl_core_news_sm` |
| Greek | `el_core_news_sm` |

Languages without a dedicated spaCy model are handled via server-side translation (Argos Translate) to the sign language's base language before NLP processing. Models are loaded on demand with an LRU cache (max 4 concurrent).

## Tech Stack

| Layer | Technology |
|---|---|
| NLP | spaCy (7 language models) |
| Translation | Argos Translate (server-side) |
| Sign notation | HamNoSys |
| Markup | SiGML (XML) |
| 3D rendering | CWASA avatar |

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/kozha.git
cd kozha

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (includes all spaCy language models)
pip install -r server/requirements.txt
```

### Running Locally

```bash
# Start the application
uvicorn server.server:app --reload

# The app will be available at http://localhost:8000
```

## Chrome Extension

The `extension/` directory contains a Chrome extension that translates text into sign language on any webpage.

**Three modes:**

1. **Popup** — Click the extension icon (or `Ctrl+Shift+K` / `Cmd+Shift+K`) to open a standalone translator. Type text, pick input and sign languages, and click Sign.
2. **Context menu** — Select text on any webpage, right-click, and choose "Sign this text". A floating panel appears with the 3D avatar signing the selection.
3. **YouTube** — Automatically injects on YouTube watch pages. Extracts captions, translates them, and syncs sign playback with the video timeline. Supports windowed translation for long videos (>200 segments).

**Install:** Open `chrome://extensions`, enable Developer Mode, click "Load unpacked", and select the `extension/` folder.

**Keyboard shortcut:** `Ctrl+Shift+K` (Mac: `Cmd+Shift+K`) opens the popup.

**Limitations:**
- BSL has the most complete sign database; other sign languages have varying coverage
- Requires the kozha-translate.com backend to be running
- YouTube mode requires captions (auto-generated or manual)
- CWASA avatar requires WebGL (falls back to text-only gloss display)

## Project Structure

```
kozha/
├── .github/workflows/deploy.yml
├── extension/
│   ├── manifest.json
│   ├── background.js
│   ├── popup.html / popup.js / popup.css
│   ├── panel.html / panel.css
│   ├── content-shared.js
│   ├── content-youtube.js
│   ├── content-universal.js
│   └── icons/
├── server/
│   ├── server.py
│   ├── requirements.txt
│   └── abbreviations.json
├── public/
│   ├── index.html
│   ├── app.html
│   ├── contribute.html
│   └── LICENSE
├── data/
│   ├── hamnosys_bsl_version1.sigml
│   ├── hamnosys_bsl.csv
│   ├── bsl_alphabet_sigml.sigml
│   ├── asl_alphabet_sigml.sigml
│   ├── dgs_alphabet_sigml.sigml
│   ├── lsf_alphabet_sigml.sigml
│   ├── pjm_alphabet_sigml.sigml
│   ├── ngt_alphabet_sigml.sigml
│   └── ... (15 sign language databases)
└── README.md
```

**Known limitations:**
- Vocabulary coverage is still expanding.
- BSL has the most complete sign database; other sign languages have varying coverage.
- Languages without a dedicated spaCy model rely on client-side translation before NLP processing.

## Pre-launch setup (chat2hamnosys)

Until the deployed Fly host has the OpenAI and captcha secrets
provisioned, two temporary measures keep the live
[kozha-translate.com/contribute.html](https://kozha-translate.com/contribute.html)
flow usable.

### 1. Provide your own OpenAI key from the browser

The contribute page now has an optional **"Your OpenAI API key"** field.
Paste an `sk-…` key there; the browser stores it in `localStorage`
(`bridgn.openai_api_key`) and sends it as the `X-OpenAI-Api-Key`
header on every authoring call. The backend's `LLMClient` consults
this header before falling back to the `OPENAI_API_KEY` env var, so
contributions work before the project key is set.

Precedence, highest to lowest:

1. Explicit `api_key=` argument (used by tests and fixture recorders).
2. Per-request header from the browser (this pre-launch path).
3. `OPENAI_API_KEY` environment variable / Fly secret (the long-term
   path — once set, contributors can leave the field blank and the
   project picks up the bill).

To **stop** using a personal key, clear the field on contribute.html
(or `localStorage.removeItem('bridgn.openai_api_key')` from the
browser console). Keys are never logged server-side and are scoped
to a single request.

Once the project secret is provisioned and you want the field gone
entirely, delete the `<!-- BYO OpenAI key … -->` block in
`public/contribute.html` and the corresponding `readBYOOpenAIKey()`
call in `public/chat2hamnosys/app.js`.

### 2. Captcha is temporarily off

`fly.toml` sets `CHAT2HAMNOSYS_CAPTCHA_DISABLED = "1"` because
`CHAT2HAMNOSYS_CAPTCHA_SECRET` has not been provisioned yet. In this
state the registration form hides the math challenge and the backend
accepts any `captcha_challenge`/`captcha_answer` pair. The honeypot
field stays active — it is the only spam defence while the captcha
is off, so watch new registrations until the captcha is back on.

**To re-enable the captcha:**

```bash
fly secrets set CHAT2HAMNOSYS_CAPTCHA_SECRET="$(openssl rand -hex 32)"
# Then in fly.toml remove (or flip to "0") the
# CHAT2HAMNOSYS_CAPTCHA_DISABLED line and redeploy:
fly deploy
```

The frontend auto-reveals the challenge row on the next page load —
no further code changes needed.

---

## Contributing

Contributions are welcome. To get started:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Commit your changes with clear messages.
4. Open a pull request describing what you changed and why.

If you're unsure where to start, check the open issues or reach out to the maintainers.

## Credits

Full attribution, license detail, per-corpus entry counts, and reviewer-provenance statements live on the public credits page: **[kozha-translate.com/credits](https://kozha-translate.com/credits)** (source: [`public/credits.html`](public/credits.html)). The section below is a pointer summary — if the two disagree, the `/credits` page is the authoritative version.

### Avatar and rendering
- **[CWASA](https://vhg.cmp.uea.ac.uk/tech/jas/vhg2024/cwa/)** — 3D signing avatar engine by the [Virtual Humans Group](https://vhg.cmp.uea.ac.uk/), University of East Anglia. License: Creative Commons Attribution-NoDerivatives (CC BY-ND), per the upstream [CWASA Conditions of Use](https://vh.cmp.uea.ac.uk/index.php/CWASA_Conditions_of_Use).

### Notation system
- **[HamNoSys](https://www.sign-lang.uni-hamburg.de/dgs-korpus/files/inhalt_pdf/HamNoSys_2018.pdf)** — Hamburg Notation System for sign language transcription by the [Institute of German Sign Language (IDGS)](https://www.sign-lang.uni-hamburg.de/), Universität Hamburg. License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — share and adapt with attribution, including commercial use.
- **bgHamNoSysUnicode font** — HamNoSys 4.0 Unicode glyph font distributed by IDGS, Universität Hamburg. Used on the contribute page to render HamNoSys glyphs at their intended visual fidelity; loaded locally from `public/fonts/bgHamNoSysUnicode.ttf` (see [`public/fonts/bgHamNoSysUnicode.LICENSE.txt`](public/fonts/bgHamNoSysUnicode.LICENSE.txt) for provenance and distribution terms).

### Sign-language databases

Corpora with declared licenses:

- **BSL** — [DictaSign Corpus](https://www.sign-lang.uni-hamburg.de/dicta-sign/portal/), IDGS, Universität Hamburg. License: [CC BY-NC-SA 3.0 Unported](https://creativecommons.org/licenses/by-nc-sa/3.0/). 881 signs loaded; aliased to ASL (lexical fallback + ASL fingerspelling).
- **LSF, GSL** — DictaSign LSF and GSL portions, redistributed via [SignAvatars](https://github.com/ZhengdiYu/SignAvatars). License: CC BY-NC-SA 3.0 Unported (inherited from DictaSign). 381 + 889 signs.
- **DGS** — DGS Lexicon (IDGS / DGS-Korpus, Universität Hamburg), redistributed via SignAvatars. License: per the upstream DGS Lexicon terms. 1,914 signs.
- **PJM** — [PJM Dictionary (słownik PJM)](https://slownikpjm.uw.edu.pl/), Section for Sign Linguistics, University of Warsaw, redistributed via SignAvatars. License: per the upstream Warsaw PJM Dictionary terms. 1,932 signs.

Community repositories — license status unclear (being clarified with authors; if clarification is not obtained the source will be removed from the translator rather than used silently):

- **NGT** (Dutch) — [SignLanguageSynthesis](https://github.com/LykeEsselink/SignLanguageSynthesis) by Lyke Esselink. No license declared upstream. 39 signs.
- **Algerian SL** — [algerianSignLanguage-avatar](https://github.com/linuxscout/algerianSignLanguage-avatar) by Taha Zerrouki. No license declared upstream. 1 sign (seed).
- **Bangla SL** — [bdsl-3d-animation](https://gitlab.com/devarifkhan/bdsl-3d-animation) by Devr Arif Khan. No license declared upstream. 81 signs.
- **Indian SL** — [Text-to-Sign-Language](https://github.com/human-divanshu/Text-to-Sign-Language) by Divanshu and [text_to_isl](https://github.com/shoebham/text_to_isl) by Shoebham. No license declared upstream. 763 signs combined.
- **Kurdish SL** — [KurdishSignLanguage](https://github.com/KurdishBLARK/KurdishSignLanguage) by KurdishBLARK. No license declared upstream. 558 signs.
- **Vietnamese SL** — [VSL](https://github.com/raianrido/VSL) by Raian Rido. No license declared upstream. 3,564 signs (largest community contribution; license clarification is a priority).
- **Filipino SL** — [syntheticfsl](https://github.com/jennieablog/syntheticfsl) and [signtyper](https://github.com/jennieablog/signtyper) by Jennie Ablog. No license declared upstream. Filipino SL data ships in a SiGML variant the current loader does not parse (`<hamgestural_sign>`), so lexical signs do not load today; fingerspelling works.

### Translation layer

- **[argostranslate](https://github.com/argosopentech/argos-translate)** — offline text translation by Argos Open Technologies. License: MIT (library). Translation-model terms vary per pair.
- **[spaCy](https://spacy.io/)** — NLP pipeline (tokenisation, lemmatisation, POS, sentence segmentation) by Explosion AI. License: MIT (library). Model terms per the [spaCy model registry](https://github.com/explosion/spacy-models); the small-news and core-web models Bridgn uses are MIT-licensed as of spaCy 3.8.

### Team

- **Zhan** — developer
- **Bogdan** — developer
- **Advisor**: Askhat Zhumabekov

### Deaf advisory board

The Deaf advisory board is being seated (zero candidates confirmed). Until the board is seated, no signs from community contributions are exported to the public translator — see the [governance page](https://kozha-translate.com/governance.html) for the full review policy.

### Funding and contributor compensation

No external funding to date; infrastructure costs are covered by the core team. A contributor compensation policy for Deaf reviewers and advisory-board members is not yet in place — drafting it is a prerequisite for seating the board. Both status notes are maintained on the [credits page](https://kozha-translate.com/credits).
