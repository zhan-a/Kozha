# server.py
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from typing import List, Dict, Optional
import json, re, spacy

# -------- your original code ------------
APP_ROOT = Path(__file__).resolve().parent
DB_DIR = APP_ROOT / "database"

def list_database_words() -> List[str]:
    if not DB_DIR.exists():
        return []
    return sorted({p.stem.lower() for p in DB_DIR.glob("*.mp4")})

def load_abbreviations(filepath: Path) -> dict[str, str]:
    if not filepath.exists():
        print(f"[ERROR] File not found: {filepath}")
        return {}
    with filepath.open("r", encoding="utf-8") as f:
        raw = json.load(f)
        return {k.strip().lower(): v.strip().lower() for k, v in raw.items()}

ABBREVIATIONS = load_abbreviations(APP_ROOT / "abbreviations.json")
CUSTOM_STOPWORDS = {
    "a","an","the","and","or","but","if","then","than",
    "of","to","in","on","at","for","from","with","as","by","is","are","am",
    "be","been","was","were","do","does","did","that","this","those","these","it","my","your","our"
}

def replace_abbreviations(text: str, abbreviations: dict[str, str]) -> str:
    if not abbreviations:
        return text
    keys_sorted = sorted(abbreviations.keys(), key=lambda k: -len(k))
    for phrase in keys_sorted:
        pattern = re.compile(re.escape(phrase), flags=re.IGNORECASE)
        replacement = abbreviations[phrase.lower()]
        text = pattern.sub(replacement, text)
    return text

nlp = spacy.load("en_core_web_sm")

def process_sentence(doc_or_span, stopwords):
    tokens = []
    for token in doc_or_span:
        text = token.text.lower()
        if text in {"i", "you", "we", "they", "he", "she"}:
            tokens.append(text)
        elif token.pos_ == "VERB":
            tokens.append(token.lemma_.lower())
        elif token.pos_ in {"NOUN", "PROPN"} and text not in stopwords:
            tokens.append(text)
    tokens = [t for t in tokens if t not in stopwords]
    return " ".join(dict.fromkeys(tokens)) + "."

def process_text(text: str, nlp, abbreviations: Dict[str, str], stopwords: set) -> str:
    replaced_text = replace_abbreviations(text, abbreviations)
    doc = nlp(replaced_text)
    output_lines = []
    for sent in doc.sents:
        line = process_sentence(sent, stopwords)
        output_lines.append(line)
    return "\n".join(output_lines)

def plan_from_text(text: str, language_hint: Optional[str] = None) -> Dict[str, object]:
    allowed = list_database_words()
    if not allowed:
        return {"error": "database/ is empty — add <token>.mp4 files first."}
    text = (text or "").strip()
    if not text:
        return {"error": "Empty text."}
    rewritten = process_text(text, nlp, ABBREVIATIONS, CUSTOM_STOPWORDS)
    return {"allowed": allowed, "raw": text, "final": rewritten}

# -------- FastAPI wrapper ------------
app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/plan")
def api_plan(req: TextRequest):
    return plan_from_text(req.text)
