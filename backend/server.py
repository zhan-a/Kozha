# server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from typing import Dict, Optional
import json
import re
import spacy
from spacy.matcher import PhraseMatcher

APP_ROOT = Path(__file__).resolve().parent

# ---------- abbreviations ----------

def load_abbreviations(filepath: Path) -> Dict[str, str]:
    """
    Load abbreviations from JSON of the form:
      {
        "carbon dioxide": "co2",
        "i love you": "ily",
        ...
      }
    All keys/values normalized to lowercase.
    """
    if not filepath.exists():
        print(f"[WARN] Abbreviation file not found: {filepath}")
        return {}
    with filepath.open("r", encoding="utf-8") as f:
        raw = json.load(f)
        return {
            k.strip().lower(): v.strip().lower()
            for k, v in raw.items()
        }

ABBREVIATIONS: Dict[str, str] = load_abbreviations(APP_ROOT / "abbreviations.json")

CUSTOM_STOPWORDS = {
    "a","an","the","and","or","but","if","then","than",
    "of","to","in","on","at","for","from","with","as","by","is","are","am",
    "be","been","was","were","do","does","did","that","this","those","these",
    "it","my","your","our"
}
ABBREV_VALUES = {v for v in ABBREVIATIONS.values() if v not in CUSTOM_STOPWORDS}

# Simple lexical set for time references we want to front
TIME_WORDS = {
    "today","yesterday","tomorrow",
    "morning","afternoon","evening","night","tonight",
    "now","later","soon",
    "week","month","year",
    "monday","tuesday","wednesday","thursday","friday","saturday","sunday",
    "january","february","march","april","may","june","july",
    "august","september","october","november","december"
}

# ---------- spaCy EDU-ish pipeline ----------

nlp = spacy.load("en_core_web_sm")

# Build a PhraseMatcher for abbreviation phrases (multi-word keys)
ABBREV_MATCHER = PhraseMatcher(nlp.vocab, attr="LOWER")
if ABBREVIATIONS:
    ABBREV_MATCHER.add("ABBREV_PHRASE", [nlp.make_doc(k) for k in ABBREVIATIONS.keys()])

def process_sentence(doc_or_span, stopwords: set, abbr_spans: Optional[Dict[int, tuple[int, list[str]]]] = None) -> str:
    """
    Keep:
      - abbreviation outputs (co2, dna, ily, ...)  -> spelled as separate letters
      - pronouns (i, you, we, they, he, she)
      - verb lemmas
      - nouns / proper nouns
      - adjectives

    Reorder:
      - time words (today, tomorrow, monday, ...) moved to FRONT

    Remove:
      - stopwords
      - duplicates (keep first occurrence) BUT:
        abbreviation letters bypass dedupe so "asap" stays "a s a p"
    """
    # store (token_text, do_dedupe)
    time_tokens = []
    main_tokens = []

    tokens = list(doc_or_span)
    j = 0
    while j < len(tokens):
        token = tokens[j]
        text = token.text.lower()

        # --- phrase-level abbreviation replacement (multi-word keys) ---
        if abbr_spans:
            start_i = token.i  # global doc token index
            if start_i in abbr_spans:
                end_i, letters = abbr_spans[start_i]
                # push letters as main tokens, bypass dedupe
                for ch in letters:
                    main_tokens.append((ch, False))

                # skip tokens until we reach end_i (global index)
                j += 1
                while j < len(tokens) and tokens[j].i < end_i:
                    j += 1
                continue

        candidate = None
        do_dedupe = True

        # --- single-token abbreviation values (user types "asap" directly, etc.) ---
        if text in ABBREV_VALUES:
            # spell as separate letters, bypass dedupe
            letters = [c.lower() for c in text if c.isalnum()]
            for ch in letters:
                main_tokens.append((ch, False))
            j += 1
            continue

        # Pronouns that are meaningful in glosses
        if text in {"you", "we", "they", "he", "she", "me", "them", "her", "him", "us"}:
            candidate = text


        # Treat spoken "I" as "me" so single-letter i is only used for fingerspelling
        elif text == "i":
            candidate = "me"

        # Verbs as lemmas
        elif token.pos_ == "VERB":
            candidate = token.lemma_.lower()

        # Nouns, proper nouns, adjectives
        elif token.pos_ in {"NOUN", "PROPN", "ADJ"} and text not in stopwords:
            candidate = text

        if not candidate:
            j += 1
            continue

        # Extra stopword guard
        if candidate in stopwords:
            j += 1
            continue

        # Decide which bucket this token goes into
        if candidate in TIME_WORDS:
            time_tokens.append((candidate, do_dedupe))
        else:
            main_tokens.append((candidate, do_dedupe))

        j += 1

    combined = time_tokens + main_tokens

    # Dedupe while preserving order, but DO NOT dedupe abbreviation letters
    seen = set()
    dedup = []
    for t, dedupe_flag in combined:
        if dedupe_flag:
            if t in seen:
                continue
            seen.add(t)
        dedup.append(t)
    return " ".join(dedup + ["."])


def _norm_phrase(s: str) -> str:
    # normalize whitespace + lowercase to match ABBREVIATIONS keys
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def get_abbreviation_spans(doc) -> Dict[int, tuple[int, list[str]]]:
    """
    Return mapping: start_token_index -> (end_token_index, letter_tokens)
    Example phrase "as soon as possible" -> "asap" -> ["a","s","a","p"]
    """
    if not ABBREVIATIONS:
        return {}

    matches = ABBREV_MATCHER(doc)
    spans = [doc[start:end] for _, start, end in matches]
    spans = spacy.util.filter_spans(spans)  # keep longest / remove overlaps

    out: Dict[int, tuple[int, list[str]]] = {}
    for span in spans:
        phrase = _norm_phrase(span.text)
        abbr = ABBREVIATIONS.get(phrase)
        if not abbr:
            continue

        # spell into separate tokens (keeps digits too: co2 -> c o 2)
        letters = [c.lower() for c in abbr if c.isalnum()]
        out[span.start] = (span.end, letters)

    return out

def process_text(text: str) -> str:
    """
    1) Run spaCy on the ORIGINAL text
    2) Precompute abbreviation phrase matches on the doc
    3) Summarise each sentence → one line of key tokens,
       with time references fronted
    """
    doc = nlp(text)

    # ✅ phrase abbreviations on original doc (before filtering/lemmatizing)
    abbr_spans = get_abbreviation_spans(doc)

    lines = []
    for sent in doc.sents:
        line = process_sentence(sent, CUSTOM_STOPWORDS, abbr_spans)
        if line:
            lines.append(line)

    rewritten = "\n".join(lines)

    # With phrase matcher, this is now mostly redundant.
    # Keep it only if you want a fallback for edge-cases.
    # final = apply_abbreviations_last(rewritten, ABBREVIATIONS)
    # return final

    return rewritten


# ---------- main planner ----------

def plan_from_text(text: str, language_hint: Optional[str] = None) -> Dict[str, object]:
    text = (text or "").strip()
    if not text:
        return {"error": "Empty text."}

    rewritten = process_text(text)

    # NOTE: frontend never uses backend 'allowed';
    # it builds allowed glosses from the browser's .sigml DB.
    # We keep the field for compatibility.
    return {
        "allowed": [],
        "raw": text,
        "final": rewritten,
    }

# ---------- FastAPI + CORS ----------

app = FastAPI()

@app.get("/")
def root():
    return {"ok": True}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://mironovb.github.io"],
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

class TextRequest(BaseModel):
    text: str

@app.post("/plan")
def api_plan(req: TextRequest):
    return plan_from_text(req.text)
