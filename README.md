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

Languages without a dedicated spaCy model are handled via client-side translation (OPUS-MT) to the sign language's base language before NLP processing. Models are loaded on demand with an LRU cache (max 4 concurrent).

## Tech Stack

| Layer | Technology |
|---|---|
| NLP | spaCy (7 language models) |
| Translation | OPUS-MT (client-side, via Transformers.js) |
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

The `extension/` directory contains a Chrome extension that adds real-time sign language translation to YouTube videos.

**What it does:** Extracts captions from any YouTube video, translates them into sign language glosses via the Kozha backend, and renders a 3D signing avatar in an overlay panel.

**Install:** Open `chrome://extensions`, enable Developer Mode, click "Load unpacked", and select the `extension/` folder.

**How it works:** The content script extracts YouTube's caption track, sends segments to `kozha-translate.com/api/translate/batch` for NLP processing, then the panel iframe loads the CWASA avatar and plays SiGML animations mapped from the returned glosses. Playback syncs with the video timeline.

**Limitations:**
- BSL sign database only (other sign languages have alphabet-only coverage)
- Requires the kozha-translate.com backend to be running
- Video must have captions (auto-generated or manual)
- CWASA avatar requires WebGL support

## Project Structure

```
kozha/
├── .github/workflows/deploy.yml
├── extension/
│   ├── manifest.json
│   ├── content.js
│   ├── background.js
│   ├── panel.html
│   ├── panel.css
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

## Team

- **Zhan** — developer
- **Bogdan** — developer
- **Advisor: Askhat Zhumabekov**
