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
5. **3D Avatar Rendering (CAWSA)** — SiGML is rendered as 3D sign language animation using the [CAWSA](http://vh.cmp.uea.ac.uk/index.php/CWA) avatar system.

## Tech Stack

| Layer | Technology |
|---|---|
| NLP | spaCy |
| Sign notation | HamNoSys |
| Markup | SiGML (XML) |
| 3D rendering | CAWSA avatar |

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/kozha.git
cd kozha

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r server/requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

### Running Locally

```bash
# Start the application
uvicorn server.server:app --reload

# The app will be available at http://localhost:8000
```

## Project Structure

```
kozha/
├── .github/workflows/deploy.yml
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
- Currently optimized for BSL.

**Planned improvements:**
- Expanded sign vocabulary and grammar rules.
- Support for additional sign languages.

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
