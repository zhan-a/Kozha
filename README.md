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

## Contributing

Contributions are welcome. To get started:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Commit your changes with clear messages.
4. Open a pull request describing what you changed and why.

If you're unsure where to start, check the open issues or reach out to the maintainers.

## Credits

### Avatar & Rendering
- **[CWASA](https://vhg.cmp.uea.ac.uk/tech/jas/vhg2024/cwa/)** — 3D signing avatar engine by the [Virtual Humans Group](https://vhg.cmp.uea.ac.uk/), University of East Anglia ([CC BY-ND](https://vh.cmp.uea.ac.uk/index.php/CWASA_Conditions_of_Use))

### Notation System
- **[HamNoSys](https://www.sign-lang.uni-hamburg.de/dgs-korpus/files/inhalt_pdf/HamNoSys_2018.pdf)** — Hamburg Notation System for sign language transcription by the [Institute of German Sign Language](https://www.sign-lang.uni-hamburg.de/), Universitat Hamburg ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))

### Sign Language Databases
- **BSL, DGS, LSF, GSL** — [DictaSign Corpus](https://www.sign-lang.uni-hamburg.de/dicta-sign/portal/), Universitat Hamburg. DictaSign was an EU FP7-funded project (2009–2012) with partners including LIMSI-CNRS (France) and Aristotle University of Thessaloniki (Greece) ([CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/))
- **PJM** (Polish) — derived from DictaSign-adjacent Hamburg corpus work
- **NGT** (Dutch) — derived from DictaSign-adjacent Hamburg corpus work
- **Algerian, Bangla, Indian, Kurdish, Vietnamese, Filipino SL** — compiled from publicly available SiGML research datasets using the JASigning/CWASA pipeline format

## Team

- **Zhan** — developer
- **Bogdan** — developer
- **Advisor: Askhat Zhumabekov**
