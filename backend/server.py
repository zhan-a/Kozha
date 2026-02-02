from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from typing import Dict, Optional
import json
import re
import spacy
from spacy.matcher import PhraseMatcher

APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parent
FRONTEND_DIR = REPO_ROOT / "frontend"
DATA_DIR = FRONTEND_DIR / "data"

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

def load_abbreviations(filepath: Path) -> Dict[str, str]:
    if not filepath.exists():
        print(f"[WARN] Abbreviation file not found: {filepath}")
        return {}
    with filepath.open("r", encoding="utf-8") as f:
        raw = json.load(f)
        return {k.strip().lower(): v.strip().lower() for k, v in raw.items()}

ABBREVIATIONS: Dict[str, str] = load_abbreviations(APP_ROOT / "abbreviations.json")

CUSTOM_STOPWORDS = {
    "a","an","the","and","or","but","if","then","than",
    "of","to","in","on","at","for","from","with","as","by","is","are","am",
    "be","been","was","were","do","does","did","that","this","those","these",
    "it","my","your","our"
}
ABBREV_VALUES = {v for v in ABBREVIATIONS.values() if v not in CUSTOM_STOPWORDS}

TIME_WORDS = {
    "today","yesterday","tomorrow",
    "morning","afternoon","evening","night","tonight",
    "now","later","soon",
    "week","month","year",
    "monday","tuesday","wednesday","thursday","friday","saturday","sunday",
    "january","february","march","april","may","june","july",
    "august","september","october","november","december"
}

nlp = spacy.load("en_core_web_sm")

ABBREV_MATCHER = PhraseMatcher(nlp.vocab, attr="LOWER")
if ABBREVIATIONS:
    ABBREV_MATCHER.add("ABBREV_PHRASE", [nlp.make_doc(k) for k in ABBREVIATIONS.keys()])

def _norm_phrase(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def get_abbreviation_spans(doc) -> Dict[int, tuple[int, list[str]]]:
    if not ABBREVIATIONS:
        return {}

    matches = ABBREV_MATCHER(doc)
    spans = [doc[start:end] for _, start, end in matches]
    spans = spacy.util.filter_spans(spans)

    out: Dict[int, tuple[int, list[str]]] = {}
    for span in spans:
        phrase = _norm_phrase(span.text)
        abbr = ABBREVIATIONS.get(phrase)
        if not abbr:
            continue
        letters = [c.lower() for c in abbr if c.isalnum()]
        out[span.start] = (span.end, letters)
    return out

def process_sentence(doc_or_span, stopwords: set, abbr_spans: Optional[Dict[int, tuple[int, list[str]]]] = None) -> str:
    time_tokens = []
    main_tokens = []

    tokens = list(doc_or_span)
    j = 0
    while j < len(tokens):
        token = tokens[j]
        text = token.text.lower()

        if abbr_spans:
            start_i = token.i
            if start_i in abbr_spans:
                end_i, letters = abbr_spans[start_i]
                for ch in letters:
                    main_tokens.append((ch, False))
                j += 1
                while j < len(tokens) and tokens[j].i < end_i:
                    j += 1
                continue

        candidate = None
        do_dedupe = True

        if text in ABBREV_VALUES:
            letters = [c.lower() for c in text if c.isalnum()]
            for ch in letters:
                main_tokens.append((ch, False))
            j += 1
            continue

        if text in {"you","we","they","he","she","me","them","her","him","us"}:
            candidate = text
        elif text == "i":
            candidate = "me"
        elif token.pos_ == "VERB":
            candidate = token.lemma_.lower()
        elif token.pos_ in {"NOUN","PROPN","ADJ"} and text not in stopwords:
            candidate = text

        if not candidate or candidate in stopwords:
            j += 1
            continue

        if candidate in TIME_WORDS:
            time_tokens.append((candidate, do_dedupe))
        else:
            main_tokens.append((candidate, do_dedupe))

        j += 1

    combined = time_tokens + main_tokens
    seen = set()
    dedup = []
    for t, dedupe_flag in combined:
        if dedupe_flag and t in seen:
            continue
        if dedupe_flag:
            seen.add(t)
        dedup.append(t)

    return " ".join(dedup + ["."])

def process_text(text: str) -> str:
    doc = nlp(text)
    abbr_spans = get_abbreviation_spans(doc)

    lines = []
    for sent in doc.sents:
        line = process_sentence(sent, CUSTOM_STOPWORDS, abbr_spans)
        if line:
            lines.append(line)
    return "\n".join(lines)

def plan_from_text(text: str, language_hint: Optional[str] = None) -> Dict[str, object]:
    text = (text or "").strip()
    if not text:
        return {"error": "Empty text."}
    rewritten = process_text(text)
    return {"allowed": [], "raw": text, "final": rewritten}

@app.get("/api/health")
def health():
    return {"ok": True}

@app.post("/api/plan")
def api_plan(req: TextRequest):
    return plan_from_text(req.text)

if DATA_DIR.exists():
    app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

@app.get("/", include_in_schema=False)
def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")

app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

class TextRequest(BaseModel):
    text: str
