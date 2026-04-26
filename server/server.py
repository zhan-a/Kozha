import mimetypes
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

# Register web-font MIME types up-front so StaticFiles emits the right
# Content-Type for the HamNoSys binaries served from /public/fonts/.
# Python 3.9+ knows woff2 and ttf out of the box, but stripped-down
# container images (e.g. python-slim without the mailcap package) can
# ship a /etc/mime.types that lacks them — without these explicit
# entries the response goes out as application/octet-stream, which
# modern browsers refuse to parse as a font and the @font-face
# silently fails to apply.
mimetypes.add_type("font/woff2", ".woff2")
mimetypes.add_type("font/woff", ".woff")
mimetypes.add_type("font/ttf", ".ttf")
mimetypes.add_type("font/otf", ".otf")

# chat2hamnosys uses flat imports (``from session import ...``); make its
# package directory importable before anything inside it is referenced.
_CHAT2HAMNOSYS_ROOT = Path(__file__).resolve().parent.parent / "backend" / "chat2hamnosys"
if str(_CHAT2HAMNOSYS_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHAT2HAMNOSYS_ROOT))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from collections import OrderedDict
from threading import Lock
import json
import re
import logging
import spacy
from spacy.matcher import PhraseMatcher

import kozha_obs as _obs  # noqa: E402 — sibling module under ``server/``

_obs.configure_logging(level=os.environ.get("KOZHA_LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

_argos_ready = False
def _ensure_argos():
    global _argos_ready
    if _argos_ready:
        return
    try:
        from argostranslate import package
        package.update_package_index()
        available = package.get_available_packages()
        installed_codes = {(p.from_code, p.to_code) for p in package.get_installed_packages()}
        langs = ["en", "fr", "de", "es", "pl", "nl", "el", "ru", "ar"]
        for src in langs:
            for tgt in langs:
                if src == tgt:
                    continue
                if (src, tgt) in installed_codes:
                    continue
                pkg = next((p for p in available if p.from_code == src and p.to_code == tgt), None)
                if pkg:
                    logger.info("Installing argos translation package: %s → %s", src, tgt)
                    pkg.install()
        _argos_ready = True
    except Exception as e:
        logger.warning("Could not initialize argostranslate: %s", e)
        _argos_ready = True

class TextRequest(BaseModel):
    text: str
    language: str = "en"
    sign_language: str = "bsl"
    # Prompt 6: optional "strict" mode. When true, the backend annotates each
    # token with whether the target sign_language has a Deaf-native-reviewed
    # record for it; unreviewed tokens get force_fingerspell=true so the
    # frontend can fall through to the fingerspelling alphabet instead of
    # serving a questionable sign. Default False preserves legacy behavior.
    reviewed_only: bool = False

APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parent
PUBLIC_DIR = REPO_ROOT / "public"
DATA_DIR = REPO_ROOT / "data"

import review_metadata as _review_meta  # noqa: E402 — after DATA_DIR is set

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# Request-id middleware (polish-13 §2).
#
# Every request gets a hex uuid as ``request.state.request_id`` — the
# translation route handlers log it, the 5xx error handler puts it in
# the response body, and the ``X-Request-ID`` response header mirrors
# it so a user quoting the id from the browser's devtools can be
# correlated with log lines the operator sees.
#
# The middleware only touches two dict lookups and one uuid.hex call,
# so the per-request overhead is well under a millisecond — safely
# inside the 20 ms instrumentation budget.
@app.middleware("http")
async def _kozha_request_id_middleware(request: Request, call_next):
    request.state.request_id = _obs.new_request_id()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response

def load_abbreviations(filepath: Path) -> Dict[str, str]:
    if not filepath.exists():
        print(f"[WARN] Abbreviation file not found: {filepath}")
        return {}
    with filepath.open("r", encoding="utf-8") as f:
        raw = json.load(f)
        return {k.strip().lower(): v.strip().lower() for k, v in raw.items()}

ABBREVIATIONS: Dict[str, str] = load_abbreviations(APP_ROOT / "abbreviations.json")
ABBREV_VALUES = {v for v in ABBREVIATIONS.values()}

MODEL_MAP = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm",
    "es": "es_core_news_sm",
    "pl": "pl_core_news_sm",
    "nl": "nl_core_news_sm",
    "el": "el_core_news_sm",
}

LANG_MODELS: OrderedDict[str, spacy.Language] = OrderedDict()
_model_lock = Lock()
MAX_MODELS = 4

try:
    LANG_MODELS["en"] = spacy.load("en_core_web_sm", disable=["ner"])
except OSError:
    print("[ERROR] English spaCy model 'en_core_web_sm' not installed. Run: python -m spacy download en_core_web_sm")
    LANG_MODELS["en"] = spacy.blank("en")
    LANG_MODELS["en"].add_pipe("sentencizer")

def get_nlp(lang: str) -> spacy.Language:
    with _model_lock:
        if lang in LANG_MODELS:
            LANG_MODELS.move_to_end(lang)
            return LANG_MODELS[lang]

    model_name = MODEL_MAP.get(lang)
    if not model_name:
        with _model_lock:
            return LANG_MODELS["en"]

    try:
        nlp = spacy.load(model_name, disable=["ner"])
    except OSError:
        print(f"[WARN] spaCy model '{model_name}' not installed, falling back to English")
        with _model_lock:
            return LANG_MODELS["en"]

    with _model_lock:
        LANG_MODELS[lang] = nlp
        LANG_MODELS.move_to_end(lang)
        while len(LANG_MODELS) > MAX_MODELS:
            oldest = next(iter(LANG_MODELS))
            if oldest == "en":
                keys = list(LANG_MODELS.keys())
                if len(keys) > 1:
                    del LANG_MODELS[keys[1]]
                else:
                    break
            else:
                del LANG_MODELS[oldest]
        return nlp

STOPWORDS = {
    "en": {
        "a","an","the","and","or","but","if","then","than",
        "of","to","in","on","at","for","from","with","as","by","is","are","am",
        "be","been","was","were","do","does","did","it","that","this","those","these",
    },
    "de": {
        "der","die","das","ein","eine","einem","einen","einer",
        "von","zu","in","an","auf","für","aus","als","durch",
        "über","unter","nach","bei","zwischen",
        "ist","sind","bin","war","waren",
        "dass","ob","da",
        "dies","jene",
        "mein","dein","unser",
        "sehr","nur","so","wie",
    },
    "fr": {
        "le","la","les","un","une","des",
        "de","du","au","aux","à","en","dans","sur","pour","par","avec",
        "est","sont","suis","été","être","fait","avoir",
        "que","qui","dont","où",
        "ce","cette","ces",
        "ne","se","si",
        "l'","d'","n'","s'","c'","qu'",
    },
    "es": {
        "el","la","los","las","un","una","unos","unas","y","o","pero","si",
        "de","del","al","en","por","para","con","sin","sobre","entre",
        "es","son","soy","está","estar","ser","fue","ha","hay",
        "este","esta","estos","estas","ese","esa","esos","esas",
        "mi","tu","su","nuestro","muy","más","menos","no","se","lo","le"
    },
    "pl": {
        "i","a","ale","lub","czy","że",
        "to","ten","ta","te",
        "w","na","z","do","od","po","za","o","przy","nad","pod","przed",
        "jest","są","był","była","było","być",
        "się","go","mu",
        "tak",
        "bardzo","więcej","mniej","też",
    },
    "nl": {
        "de","het","een",
        "en","als",
        "van","in","op","aan","voor","door","uit","over","na",
        "bij","tot","om","naar","tegen",
        "is","zijn","was","waren","wordt","werd","heeft","had","ben","bent",
        "die","dat",
        "er","al","zo",
    },
    "el": {
        "ο","η","το","οι","τα","ένα","μια","και","ή","αλλά","αν",
        "από","σε","με","για","στο","στη","στον","στην","του","της","των",
        "είναι","ήταν","έχει","είχε","δεν","μην","θα","να","πολύ",
        "αυτός","αυτή","αυτό","εγώ","εσύ","εμείς","εσείς","αυτοί",
        "μου","σου","μας","σας","τους","πιο","πως","που","ότι","όταν"
    },
}
STOPWORDS["default"] = STOPWORDS["en"]

TIME_WORDS = {
    "en": {
        "today","yesterday","tomorrow",
        "morning","afternoon","evening","night","tonight",
        "now","later","soon",
        "week","month","year",
        "monday","tuesday","wednesday","thursday","friday","saturday","sunday",
        "january","february","march","april","may","june","july",
        "august","september","october","november","december",
        "last","next","ago","previous","following",
        "always","never","sometimes","often","usually",
        "rarely","frequently","occasionally",
        "recently","early","late",
        "past","future","present",
        "daily","weekly","monthly","yearly",
        "already","still","yet",
        "before","after",
        "dawn","dusk","noon","midnight",
        "weekend","midday",
    },
    "de": {
        "heute","gestern","morgen",
        "vormittag","nachmittag","abend","nacht",
        "jetzt","später","bald",
        "woche","monat","jahr",
        "montag","dienstag","mittwoch","donnerstag","freitag","samstag","sonntag",
        "januar","februar","märz","april","mai","juni","juli",
        "august","september","oktober","november","dezember",
        "früher","damals","vorhin",
        "immer","nie","niemals","manchmal","oft",
        "übermorgen","vorgestern",
        "mittag","mitternacht",
        "morgens","abends","nachts",
        "stunde","minute",
        "frühling","sommer","herbst","winter",
    },
    "fr": {
        "aujourd'hui","hier","demain",
        "matin","après-midi","soir","nuit",
        "maintenant","plus tard","bientôt",
        "semaine","mois","année","an",
        "lundi","mardi","mercredi","jeudi","vendredi","samedi","dimanche",
        "janvier","février","mars","avril","mai","juin","juillet",
        "août","septembre","octobre","novembre","décembre",
        "avant-hier","après-demain",
        "midi","minuit",
        "heure","minute","jour",
        "week-end",
        "toujours","souvent","parfois",
        "déjà","tard","tôt",
    },
    "es": {
        "hoy","ayer","mañana",
        "mañana","tarde","noche",
        "ahora","luego","pronto",
        "semana","mes","año",
        "lunes","martes","miércoles","jueves","viernes","sábado","domingo",
        "enero","febrero","marzo","abril","mayo","junio","julio",
        "agosto","septiembre","octubre","noviembre","diciembre"
    },
    "pl": {
        "dzisiaj","wczoraj","jutro",
        "rano","popołudnie","wieczór","noc",
        "teraz","później","wkrótce",
        "tydzień","miesiąc","rok",
        "poniedziałek","wtorek","środa","czwartek","piątek","sobota","niedziela",
        "styczeń","luty","marzec","kwiecień","maj","czerwiec","lipiec",
        "sierpień","wrzesień","październik","listopad","grudzień",
        "przedwczoraj","pojutrze",
        "południe","północ",
        "dawno","zawsze","nigdy",
        "często","codziennie",
        "godzina","minuta",
        "lato","zima","wiosna","jesień",
        "już",
    },
    "nl": {
        "vandaag","gisteren","morgen",
        "ochtend","middag","avond","nacht",
        "nu","later","binnenkort",
        "week","maand","jaar",
        "maandag","dinsdag","woensdag","donderdag","vrijdag","zaterdag","zondag",
        "januari","februari","maart","april","mei","juni","juli",
        "augustus","september","oktober","november","december",
        "vroeger","straks","toen","eerder","daarna",
        "vanavond","vanmorgen","vanochtend","vanmiddag","vannacht",
        "overmorgen","eergisteren",
        "altijd","nooit","vaak","soms",
        "vroeg","laat",
    },
    "el": {
        "σήμερα","χθες","αύριο",
        "πρωί","απόγευμα","βράδυ","νύχτα",
        "τώρα","αργότερα","σύντομα",
        "εβδομάδα","μήνας","χρόνος",
        "δευτέρα","τρίτη","τετάρτη","πέμπτη","παρασκευή","σάββατο","κυριακή",
        "ιανουάριος","φεβρουάριος","μάρτιος","απρίλιος","μάιος","ιούνιος","ιούλιος",
        "αύγουστος","σεπτέμβριος","οκτώβριος","νοέμβριος","δεκέμβριος"
    },
}
TIME_WORDS["default"] = TIME_WORDS["en"]

PRONOUNS = {
    "en": {
        "keep": {"you","we","they","them","us","my","your","our","their","not"},
        "normalize": {
            "i": "me", "me": "me",
            "he": "he", "him": "he", "she": "he", "it": "he",
            "her": "he", "his": "his", "its": "his",
            "n't": "not",
        },
    },
    "de": {
        "keep": {"du","wir","ihr","uns",
                 "nicht","nichts",
                 "und","oder","aber","wenn","weil"},
        "normalize": {
            "ich": "ich", "mich": "ich", "mir": "ich",
            "er": "er", "ihn": "er", "ihm": "er",
            "sie": "er", "es": "er",
            "dich": "du", "dir": "du",
            "euch": "ihr",
            "ihnen": "ihnen",
        },
    },
    "fr": {
        "keep": {"tu","nous","vous","moi","toi",
                 "mon","ma","mes","ton","ta","tes","son","sa","ses",
                 "notre","votre","leur","pas","très","plus","moins",
                 "et","ou","mais"},
        "normalize": {
            "je": "moi", "j'": "moi",
            "il": "il", "lui": "il",
            "elle": "il",
            "ils": "ils", "eux": "ils",
            "elles": "ils",
            "on": "nous",
        },
    },
    "es": {
        "keep": {"tú","nosotros","ellos","ellas","él","ella","usted","ustedes","mí","ti"},
        "normalize": {"yo": "yo"},
    },
    "pl": {
        "keep": {"ja","ty","my","wy","oni","one","on",
                 "mój","twój","nasz","wasz","jego","jej","ich",
                 "nie","nic","nikt"},
        "normalize": {
            "ja": "ja", "mnie": "ja", "mi": "ja", "mną": "ja",
            "ciebie": "ty", "ci": "ty", "tobą": "ty",
            "jemu": "on", "nim": "on",
            "ona": "on", "ono": "on",
            "nią": "on", "niej": "on",
            "nas": "my", "nami": "my", "nam": "my",
            "was": "wy", "wami": "wy", "wam": "wy",
            "im": "oni", "nich": "oni",
        },
    },
    "nl": {
        "keep": {"jij","wij","hij","ons","u",
                 "mijn","jouw","dit","deze",
                 "maar","of","ook","nog","wel","meer","minder","zeer"},
        "normalize": {
            "ik": "ik", "mij": "ik", "me": "ik",
            "je": "jij", "jou": "jij",
            "we": "wij",
            "zij": "hij", "ze": "hij",
            "hem": "hij", "haar": "hij",
            "hen": "zij", "hun": "zij",
            "niet": "niet",
        },
    },
    "el": {
        "keep": {"εσύ","εμείς","εσείς","αυτός","αυτή","αυτοί","αυτές","εμένα","εσένα"},
        "normalize": {"εγώ": "εγώ"},
    },
}
PRONOUNS["default"] = PRONOUNS["en"]

QUESTION_WORDS = {
    "en": {"what","where","when","who","why","how","which"},
    "de": {"was","wo","wann","wer","warum","wie","welche","welcher","welches"},
    "fr": {"quoi","où","quand","qui","pourquoi","comment","quel","quelle"},
    "pl": {"co","kto","gdzie","kiedy","dlaczego","jak","który","która","które"},
    "nl": {"wat","waar","wanneer","wie","waarom","hoe","welk","welke"},
    "el": {"τι","πού","πότε","ποιος","γιατί","πώς","ποιο"},
}
QUESTION_WORDS["default"] = QUESTION_WORDS["en"]

def get_question_words(lang: str) -> set:
    return QUESTION_WORDS.get(lang, QUESTION_WORDS["default"])

GRAMMAR_PROFILES = {
    "bsl":      {"verb_final": False, "time_first": True},
    "asl":      {"verb_final": False, "time_first": True},
    "dgs":      {"verb_final": True,  "time_first": True},
    "lsf":      {"verb_final": True,  "time_first": True},
    "lse":      {"verb_final": False, "time_first": True},
    "pjm":      {"verb_final": False, "time_first": True},
    "gsl":      {"verb_final": False, "time_first": True},
    "ngt":      {"verb_final": True,  "time_first": True},
    "rsl":      {"verb_final": True,  "time_first": True},
    "algerian": {"verb_final": False, "time_first": True},
    "bangla":   {"verb_final": False, "time_first": True},
    "fsl":      {"verb_final": False, "time_first": True},
    "isl":      {"verb_final": False, "time_first": True},
    "kurdish":  {"verb_final": False, "time_first": True},
    "vsl":      {"verb_final": False, "time_first": True},
    "ksl":      {"verb_final": True,  "time_first": True},
}
DEFAULT_GRAMMAR = {"verb_final": False, "time_first": True}

def get_stopwords(lang: str) -> set:
    return STOPWORDS.get(lang, STOPWORDS["default"])

def get_time_words(lang: str) -> set:
    return TIME_WORDS.get(lang, TIME_WORDS["default"])

def get_pronouns(lang: str) -> dict:
    return PRONOUNS.get(lang, PRONOUNS["default"])

def get_grammar(sign_language: str) -> dict:
    return GRAMMAR_PROFILES.get(sign_language, DEFAULT_GRAMMAR)

en_nlp = LANG_MODELS["en"]
ABBREV_MATCHER = PhraseMatcher(en_nlp.vocab, attr="LOWER")
if ABBREVIATIONS:
    ABBREV_MATCHER.add("ABBREV_PHRASE", [en_nlp.make_doc(k) for k in ABBREVIATIONS.keys()])

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

def reorder_tokens(tokens_with_pos, sign_language: str, time_words: set,
                    question_words: set = None) -> list:
    grammar = get_grammar(sign_language)
    time_tokens = []
    verb_tokens = []
    question_tokens = []
    other_tokens = []

    for text, pos in tokens_with_pos:
        if text in time_words:
            time_tokens.append(text)
        elif question_words and text in question_words:
            question_tokens.append(text)
        elif pos == "VERB" and grammar["verb_final"]:
            verb_tokens.append(text)
        else:
            other_tokens.append(text)

    result = []
    if grammar["time_first"]:
        result.extend(time_tokens)
    else:
        other_tokens = time_tokens + other_tokens

    result.extend(other_tokens)

    if grammar["verb_final"]:
        result.extend(verb_tokens)

    result.extend(question_tokens)

    return result

def process_sentence(doc_or_span, stopwords: set, time_words: set, pronouns: dict,
                     sign_language: str, abbr_spans: Optional[Dict[int, tuple[int, list[str]]]] = None,
                     question_words: set = None) -> str:
    tokens_with_pos = []
    pron_keep = pronouns.get("keep", set())
    pron_norm = pronouns.get("normalize", {})

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
                    tokens_with_pos.append((ch, "ABBR"))
                j += 1
                while j < len(tokens) and tokens[j].i < end_i:
                    j += 1
                continue

        if text in ABBREV_VALUES and text not in stopwords:
            letters = [c.lower() for c in text if c.isalnum()]
            for ch in letters:
                tokens_with_pos.append((ch, "ABBR"))
            j += 1
            continue

        candidate = None
        pos = token.pos_

        if text in pron_norm:
            candidate = pron_norm[text]
        elif text in pron_keep:
            candidate = text
        elif question_words and text in question_words:
            candidate = text
        elif pos == "VERB":
            candidate = token.lemma_.lower()
        elif pos in {"NOUN","PROPN","ADJ","INTJ","NUM","ADV","DET"} and text not in stopwords:
            candidate = text

        if not candidate or candidate in stopwords:
            j += 1
            continue

        tokens_with_pos.append((candidate, pos))
        j += 1

    reordered = reorder_tokens(tokens_with_pos, sign_language, time_words, question_words)

    seen = set()
    dedup = []
    for t in reordered:
        if t in seen:
            continue
        seen.add(t)
        dedup.append(t)

    return " ".join(dedup + ["."])

MAX_INPUT_LENGTH = 10000

def process_text(text: str, language: str = "en", sign_language: str = "bsl") -> str:
    text = (text or "").strip()
    if not text:
        return ""
    truncated = False
    if len(text) > MAX_INPUT_LENGTH:
        text = text[:MAX_INPUT_LENGTH]
        truncated = True
    nlp = get_nlp(language)
    stopwords = get_stopwords(language)
    time_words = get_time_words(language)
    pronouns = get_pronouns(language)
    question_words = get_question_words(language)
    doc = nlp(text)

    abbr_spans = {}
    if language == "en":
        abbr_spans = get_abbreviation_spans(doc)

    lines = []
    for sent in doc.sents:
        line = process_sentence(sent, stopwords, time_words, pronouns, sign_language, abbr_spans, question_words)
        if line and line != ".":
            lines.append(line)
    result = "\n".join(lines)
    if truncated:
        result = "[truncated] " + result
    return result

def plan_from_text(
    text: str,
    language: str = "en",
    sign_language: str = "bsl",
    reviewed_only: bool = False,
) -> Dict[str, object]:
    text = (text or "").strip()
    if not text:
        return {"error": "Empty text."}
    rewritten = process_text(text, language, sign_language)

    # Prompt 6: attach per-token review metadata so the translator chips can
    # render a Reviewed / Not yet reviewed badge and so reviewed_only mode
    # can route unreviewed glosses through the fingerspelling fallback. The
    # metadata is advisory — the client may ignore it without any breakage.
    tokens: list[str] = []
    for line in (rewritten or "").split("\n"):
        for tok in line.split():
            if tok == ".":
                continue
            tokens.append(tok)

    index = _review_meta.get_index()
    per_token: list[dict] = []
    for tok in tokens:
        rec = index.get(sign_language, tok)
        info = rec.to_dict()
        info["gloss"] = tok
        info["force_fingerspell"] = bool(reviewed_only and not rec.deaf_native_reviewed)
        per_token.append(info)

    return {
        "allowed": [],
        "raw": text,
        "final": rewritten,
        "language": language,
        "sign_language": sign_language,
        "reviewed_only": reviewed_only,
        "per_token_review": per_token,
    }

class TranslateTextRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str
    # Prompt 4: the client now sends the target *sign* language too so the
    # server can log/validate it independently of the argos base-lang route.
    target_sign_lang: Optional[str] = None
    # Alias accepted for compatibility with prompt 4's explicit payload
    # shape ({source_lang, target_sign_lang, source_text}). Treated as a
    # fallback for `text`.
    source_text: Optional[str] = None

class TranslateRequest(BaseModel):
    text: str
    source_lang: str = "en"

class TranslateSegment(BaseModel):
    text: str
    start: float
    duration: float

class BatchTranslateRequest(BaseModel):
    segments: List[TranslateSegment]
    source_lang: str = "en"

def text_to_glosses(text: str, source_lang: str = "en") -> list:
    result = process_text(text, language=source_lang)
    if not result:
        return []
    tokens = []
    for line in result.split("\n"):
        tokens.extend(line.split())
    return tokens

@contextmanager
def _instrument_translation(request: Request, source_lang: str, target_sign_lang: str, source_text: str):
    """Stopwatch + log + Prometheus update for translation routes.

    The ``outcome`` is mutable on the yielded context so the route
    handler can refine ``"success"`` to ``"validation_error"`` or
    ``"missing_gloss"`` after the fact. Any exception inside the
    block promotes the outcome to ``"server_error"`` and re-raises
    (the global exception handler renders the JSON body).
    """
    t0 = time.perf_counter()
    ctx: Dict[str, object] = {"outcome": "success"}
    try:
        yield ctx
    except Exception:
        ctx["outcome"] = "server_error"
        raise
    finally:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        outcome = str(ctx.get("outcome") or "server_error")
        _obs.record_translation(
            source_lang=source_lang,
            target_sign_lang=target_sign_lang,
            outcome=outcome,
            latency_ms=latency_ms,
        )
        client_ip = request.client.host if request.client else None
        # Honour proxy header when present (nginx terminates TLS and
        # forwards via loopback). ``X-Forwarded-For`` may be a list;
        # the first entry is the original client.
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            client_ip = fwd.split(",")[0].strip() or client_ip
        _obs.log_request(
            logger,
            request_id=getattr(request.state, "request_id", ""),
            method=request.method,
            path=request.url.path,
            source_lang=source_lang,
            target_sign_lang=target_sign_lang,
            source_text_length=len(source_text or ""),
            cache_hit=bool(ctx.get("cache_hit", False)),
            latency_ms=latency_ms,
            outcome=outcome,
            source_text=source_text,
            ip_hash=_obs.hash_ip(client_ip),
        )


@app.get("/api/health")
def health():
    return {"ok": True}


# Polish-13 §7: liveness probe. Returns 200 whenever the process is
# able to answer HTTP — no external checks, no data reads. Used by
# load balancers and uptime monitors that just want "is this pid
# alive and servicing requests".
@app.get("/health", include_in_schema=False)
def health_live():
    return {"status": "ok"}


# Polish-13 §7: readiness probe. Returns 200 only when the translator
# can actually serve traffic (data files readable, meta parses,
# optional spaCy model loaded, argostranslate importable). The body
# details which checks passed so an operator can tell a 503 from a
# warm-up state at a glance. Gated behind the LB's readiness probe
# so a slow spaCy model install does not leak requests to a partly-
# ready worker.
@app.get("/health/ready", include_in_schema=False)
def health_ready():
    report = _obs.readiness_probe()
    payload = {"status": "ok" if report.ok else "not_ready", "checks": report.checks}
    if report.detail:
        payload["detail"] = report.detail
    return JSONResponse(payload, status_code=200 if report.ok else 503)


# Polish-13 §3: Prometheus text exposition. ``PlainTextResponse``
# avoids any JSON middleware touching the body; the exposition format
# is a plain ``text/plain`` protocol. The endpoint is intentionally
# open (no token) because operators scrape it from the local network
# — nothing privacy-sensitive is here. Label sets are fixed and
# bounded (see ``obs.OUTCOMES``), so the series cardinality stays
# manageable.
@app.get("/metrics", include_in_schema=False)
def metrics():
    return PlainTextResponse(
        _obs.registry.render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


# Polish-13 §4: admin ops dashboard. Token-gated via the
# ``KOZHA_ADMIN_TOKEN`` env var. If that is empty, the route returns
# 404 — we do not want to leak that the page exists on a server that
# isn't configured to serve it. Token is compared with a constant-time
# helper to avoid leaking length through timing.
import secrets as _secrets  # noqa: E402


@app.get("/admin/ops", include_in_schema=False)
@app.get("/admin/ops/", include_in_schema=False)
def admin_ops(token: Optional[str] = None):
    expected = os.environ.get("KOZHA_ADMIN_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=404, detail="not found")
    if not token or not _secrets.compare_digest(token, expected):
        raise HTTPException(status_code=404, detail="not found")
    html_path = PUBLIC_DIR / "admin" / "ops.html"
    if not html_path.exists():
        return HTMLResponse(
            "<h1>admin/ops</h1><p>dashboard template missing</p>",
            status_code=500,
        )
    return FileResponse(html_path)

_SUPPORTED_SIGN_LANGS = {
    "bsl", "asl", "dgs", "lsf", "lse", "pjm", "gsl", "rsl",
    "algerian", "bangla", "ngt", "fsl", "isl", "kurdish", "vsl",
}
# Argos packages pre-warmed in _ensure_argos(). Any source_lang or target_lang
# outside this set is either unsupported (soft error) or a pass-through.
_SUPPORTED_BASE_LANGS = {"en", "fr", "de", "es", "pl", "nl", "el", "ru", "ar"}


def _translation_route_supported(src: str, tgt: str) -> bool:
    if src == tgt:
        return True
    return src in _SUPPORTED_BASE_LANGS and tgt in _SUPPORTED_BASE_LANGS


@app.post("/api/translate-text")
def api_translate_text(req: TranslateTextRequest, request: Request):
    # Prompt 4: accept either `text` or `source_text`; the JS client sends
    # both, but older extension builds still only send `text`.
    text = (req.text or req.source_text or "").strip()
    tgt_sign = req.target_sign_lang or "-"
    with _instrument_translation(request, req.source_lang or "", tgt_sign, text) as ctx:
        if not text or req.source_lang == req.target_lang:
            ctx["outcome"] = "success" if text else "validation_error"
            return {"translated": text}
        # Refuse unknown target sign languages explicitly rather than silently
        # falling through to argos (which would succeed but produce output the
        # client cannot render into sign).
        if req.target_sign_lang and req.target_sign_lang not in _SUPPORTED_SIGN_LANGS:
            logger.warning(
                "translate-text: unknown target_sign_lang",
                extra={"ctx": {"target_sign_lang": req.target_sign_lang}},
            )
            ctx["outcome"] = "validation_error"
            return {
                "translated": text,
                "error": f"unknown target sign language: {req.target_sign_lang}",
            }
        # Refuse source/target pairs with no argos route rather than silently
        # returning the original text and leaving the client unsure whether
        # translation ran.
        if not _translation_route_supported(req.source_lang, req.target_lang):
            ctx["outcome"] = "validation_error"
            return {
                "translated": text,
                "error": (
                    f"no translation route from {req.source_lang} to {req.target_lang}"
                ),
            }
        try:
            _ensure_argos()
            from argostranslate import translate
            result = translate.translate(text, req.source_lang, req.target_lang)
            if not isinstance(result, str):
                for attr in ("translatedText", "translated", "text", "translation"):
                    v = getattr(result, attr, None)
                    if isinstance(v, str):
                        result = v
                        break
                else:
                    logger.error(
                        "argostranslate returned non-string type",
                        extra={"ctx": {
                            "source_lang": req.source_lang,
                            "target_lang": req.target_lang,
                            "result_type": type(result).__name__,
                        }},
                    )
                    ctx["outcome"] = "server_error"
                    return {
                        "translated": text,
                        "error": f"translation returned unexpected type {type(result).__name__}",
                    }
            return {"translated": result}
        except Exception as e:
            # Explicit outcome + re-raise so the global handler can
            # format a 5xx body with the request_id; we still want the
            # metric to count this attempt as a server_error.
            logger.error("Translation error: %s", e)
            ctx["outcome"] = "server_error"
            return {"translated": text, "error": str(e)}

@app.post("/api/translate")
def api_translate(req: TranslateRequest, request: Request):
    with _instrument_translation(request, req.source_lang or "", "-", req.text) as ctx:
        glosses = text_to_glosses(req.text, req.source_lang)
        if not glosses:
            ctx["outcome"] = "missing_gloss" if (req.text or "").strip() else "validation_error"
        return {"glosses": glosses, "raw": req.text}

@app.post("/api/translate/batch")
def api_translate_batch(req: BatchTranslateRequest, request: Request):
    # Batch is a single instrumented request — counting each segment
    # separately would inflate the request count metric and break the
    # p50/p95 derived from it (those are per-request, not per-segment).
    total_text = " ".join((s.text or "") for s in req.segments)
    with _instrument_translation(request, req.source_lang or "", "-", total_text) as ctx:
        results = []
        for seg in req.segments:
            glosses = text_to_glosses(seg.text, req.source_lang)
            results.append({"glosses": glosses, "start": seg.start, "duration": seg.duration})
        if results and not any(r["glosses"] for r in results):
            ctx["outcome"] = "missing_gloss"
        return {"results": results}

@app.post("/api/plan")
def api_plan(req: TextRequest, request: Request):
    with _instrument_translation(request, req.language or "", req.sign_language or "", req.text) as ctx:
        result = plan_from_text(
            req.text,
            req.language,
            req.sign_language,
            reviewed_only=req.reviewed_only,
        )
        if isinstance(result, dict) and "error" in result:
            ctx["outcome"] = "validation_error"
            return result
        final = (result.get("final") or "").strip() if isinstance(result, dict) else ""
        if not final:
            ctx["outcome"] = "missing_gloss"
        # Per-token metrics: count words the target language's database
        # does not cover (unknown_word), and count how often we render a
        # fingerspelling fallback for a token (force_fingerspell == True
        # OR the token is not in the database). The labels are target
        # language only — we intentionally avoid labelling by the token
        # itself, which would be unbounded and a metrics anti-pattern.
        per_tok = result.get("per_token_review") if isinstance(result, dict) else None
        if per_tok:
            target = req.sign_language or ""
            for tok in per_tok:
                if not isinstance(tok, dict):
                    continue
                in_db = bool(tok.get("in_database"))
                force_fs = bool(tok.get("force_fingerspell"))
                if not in_db:
                    _obs.UNKNOWN_WORD_TOTAL.inc(target_sign_lang=target)
                if force_fs or not in_db:
                    _obs.FINGERSPELL_FALLBACK_TOTAL.inc(target_sign_lang=target)
        return result


@app.get("/api/review-meta/{sign_language}")
def api_review_meta(sign_language: str):
    """Public read-only view of a language's review-metadata summary.

    Used by the progress dashboard (prompt 8) and by the translator chip
    detail panel to look up gloss-level review state without serving the
    full meta.json payload. Returns the merged default + per-sign overrides
    for the requested sign language.
    """
    lang = (sign_language or "").strip().lower()
    if not lang:
        return {"error": "missing sign_language"}
    index = _review_meta.get_index()
    summary = index.language_summary(lang)
    return {"sign_language": lang, **summary}


@app.post("/api/review-meta/{sign_language}/lookup")
def api_review_meta_lookup(sign_language: str, glosses: Dict[str, List[str]]):
    """Bulk lookup for a list of glosses. Input body: ``{"glosses": [...]}``.

    Used by the translator when a user expands a chip detail panel for an
    arbitrary token; the server resolves the review status using the same
    precedence as ``/api/plan``. This avoids duplicating the precedence
    logic in the browser.
    """
    lang = (sign_language or "").strip().lower()
    index = _review_meta.get_index()
    out: Dict[str, dict] = {}
    for g in glosses.get("glosses") or []:
        rec = index.get(lang, g)
        info = rec.to_dict()
        info["gloss"] = g
        out[g] = info
    return {"sign_language": lang, "results": out}


# ---------------------------------------------------------------------
# /api/transcribe — Gladia fallback for the in-browser Whisper pipeline.
#
# The translator ships an on-device Whisper ASR via transformers.js +
# onnxruntime-web (public/app.html, public/index.html). On browsers
# where the local pipeline can't initialise (no SIMD, blocked CDN,
# unsupported WebAssembly surface, etc.), the client POSTs the raw
# audio blob here and we proxy through Gladia's pre-recorded API.
#
# The TRANSCRIBE_KEY env var is the only secret this route uses.
# When it's missing the route returns 503 so the client can show a
# clean error rather than receiving HTML — matches the same graceful-
# degradation contract the OPENAI_API_KEY path uses (deploy.yml gates
# the env-file write on the secret being non-empty).
# ---------------------------------------------------------------------

# Cap the proxy at 25 MB. The client-side surfaces already enforce a
# 50 MB upload cap and a 60 s duration cap, but a raw WAV at 16 kHz mono
# 16-bit comes out around 2 MB/min, so the practical Gladia payload is
# ~2 MB; the 25 MB ceiling is a defence-in-depth budget that keeps a
# malformed client from posting arbitrary blobs.
_TRANSCRIBE_MAX_BYTES = 25 * 1024 * 1024

# Total time we'll wait for Gladia's pre-recorded job to move from
# "queued" → "done". For a 60 s clip the typical end-to-end is 3–6 s;
# 90 s gives headroom for queue spikes without holding the connection
# open indefinitely.
_TRANSCRIBE_POLL_TIMEOUT_S = 90.0
_TRANSCRIBE_POLL_INTERVAL_S = 1.0


def _gladia_request(
    method: str,
    url: str,
    api_key: str,
    *,
    json_body: Optional[dict] = None,
    raw_body: Optional[bytes] = None,
    raw_content_type: Optional[str] = None,
    timeout: float = 30.0,
) -> dict:
    """Stdlib-only HTTP helper for Gladia's REST endpoints.

    Returns the parsed JSON body. Raises ``RuntimeError`` with a short
    message on transport failures or non-2xx responses. Using urllib
    instead of httpx/requests keeps server/requirements.txt lean.
    """
    import urllib.request
    import urllib.error

    headers = {"x-gladia-key": api_key, "accept": "application/json"}
    data: Optional[bytes] = None
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        headers["content-type"] = "application/json"
    elif raw_body is not None:
        data = raw_body
        headers["content-type"] = raw_content_type or "application/octet-stream"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
    except urllib.error.HTTPError as e:
        # Drain the body so the server-side message reaches our log,
        # but cap at 1 KB so a multi-megabyte HTML error page can't
        # bloat the JSON we re-raise into.
        try:
            detail = e.read().decode("utf-8", errors="replace")[:1024]
        except Exception:
            detail = ""
        raise RuntimeError(f"gladia HTTP {e.code}: {detail or e.reason}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"gladia network error: {e.reason}")
    except Exception as e:
        raise RuntimeError(f"gladia request failed: {e}")

    try:
        return json.loads(payload.decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"gladia returned non-JSON: {e}")


def _gladia_upload_audio(audio: bytes, content_type: str, api_key: str) -> str:
    """Upload the raw audio bytes via Gladia's multipart /v2/upload.

    Returns the ``audio_url`` Gladia hosts the file under (used as the
    input for the transcribe job). We construct the multipart body by
    hand so the server stays free of python-multipart at the proxy
    layer.
    """
    boundary = "----kozha-gladia-" + _secrets.token_hex(16)
    # Pick a filename extension Gladia recognises from the client's
    # content-type; it's only used to hint the format and Gladia
    # detects from the bytes anyway, so a wrong guess only affects
    # logs on their side.
    ext = "webm"
    ct = (content_type or "").lower()
    if "wav" in ct:
        ext = "wav"
    elif "mp4" in ct or "m4a" in ct:
        ext = "mp4"
    elif "mpeg" in ct or "mp3" in ct:
        ext = "mp3"
    elif "ogg" in ct:
        ext = "ogg"

    crlf = b"\r\n"
    parts = [
        f"--{boundary}".encode("ascii"),
        f'Content-Disposition: form-data; name="audio"; filename="clip.{ext}"'.encode("ascii"),
        f"Content-Type: {content_type or 'application/octet-stream'}".encode("ascii"),
        b"",
        audio,
        f"--{boundary}--".encode("ascii"),
        b"",
    ]
    body = crlf.join(parts)
    payload = _gladia_request(
        "POST",
        "https://api.gladia.io/v2/upload",
        api_key,
        raw_body=body,
        raw_content_type=f"multipart/form-data; boundary={boundary}",
        timeout=60.0,
    )
    audio_url = payload.get("audio_url")
    if not audio_url:
        raise RuntimeError(f"gladia upload returned no audio_url: {payload}")
    return audio_url


def _gladia_transcribe(audio_url: str, api_key: str, language: Optional[str]) -> str:
    """Submit a pre-recorded transcription job and poll until done.

    ``language`` accepts a 2-letter code (en/de/pl/...) which Gladia
    uses as a hint; ``None`` lets Gladia auto-detect.
    """
    body: dict = {"audio_url": audio_url, "diarization": False}
    if language:
        body["language_config"] = {"languages": [language]}
    submit = _gladia_request(
        "POST",
        "https://api.gladia.io/v2/pre-recorded",
        api_key,
        json_body=body,
        timeout=30.0,
    )
    result_url = submit.get("result_url") or submit.get("resultUrl")
    if not result_url:
        raise RuntimeError(f"gladia submit returned no result_url: {submit}")
    deadline = time.monotonic() + _TRANSCRIBE_POLL_TIMEOUT_S
    last_status = ""
    while time.monotonic() < deadline:
        poll = _gladia_request("GET", result_url, api_key, timeout=15.0)
        status = (poll.get("status") or "").lower()
        last_status = status
        if status == "done":
            result = poll.get("result") or {}
            transcription = result.get("transcription") or {}
            text = transcription.get("full_transcript")
            if text is None:
                # Some Gladia plans return per-utterance only; flatten.
                utterances = transcription.get("utterances") or []
                text = " ".join((u.get("text") or "").strip() for u in utterances).strip()
            return (text or "").strip()
        if status == "error":
            raise RuntimeError(f"gladia job errored: {poll.get('error_code') or poll}")
        time.sleep(_TRANSCRIBE_POLL_INTERVAL_S)
    raise RuntimeError(f"gladia poll timeout after {_TRANSCRIBE_POLL_TIMEOUT_S:.0f}s (last status: {last_status or 'unknown'})")


@app.post("/api/transcribe")
async def api_transcribe(request: Request):
    api_key = (os.environ.get("TRANSCRIBE_KEY") or "").strip()
    if not api_key:
        # 503 is the right shape: the route exists, it's just not
        # configured. The client falls back to its own error copy
        # rather than treating a missing key as a transient network
        # blip worth retrying.
        raise HTTPException(status_code=503, detail="Transcription fallback not configured")

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty audio body")
    if len(body) > _TRANSCRIBE_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio body exceeds {_TRANSCRIBE_MAX_BYTES // (1024 * 1024)} MB limit",
        )

    content_type = request.headers.get("content-type", "audio/webm")
    # Optional input-language hint forwarded to Gladia's language_config.
    # Sent as a header so the body stays raw audio (no multipart on the
    # client side either).
    language = (request.headers.get("x-input-language") or "").strip().lower() or None
    if language and not re.match(r"^[a-z]{2}$", language):
        language = None

    try:
        audio_url = _gladia_upload_audio(body, content_type, api_key)
        text = _gladia_transcribe(audio_url, api_key, language)
    except Exception as e:
        logger.error(
            "transcribe via gladia failed",
            extra={"ctx": {"bytes": len(body), "content_type": content_type, "lang": language, "err": str(e)[:240]}},
        )
        raise HTTPException(status_code=502, detail="Transcription service failed")

    return {"text": text, "provider": "gladia"}


try:
    from api import create_app as _create_chat2hamnosys_app
    _chat2hamnosys_sub_app = _create_chat2hamnosys_app()
    app.mount("/api/chat2hamnosys", _chat2hamnosys_sub_app)
except Exception as _c2h_mount_err:
    logger.warning("chat2hamnosys API not mounted: %s", _c2h_mount_err)

if DATA_DIR.exists():
    app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

@app.get("/", include_in_schema=False)
def serve_index():
    return FileResponse(PUBLIC_DIR / "index.html")


# Permanent per-submission status URL. Prompt 10 ships this as the link
# the contribute flow hands back after /accept — the path carries the
# session id so an author (or a reviewer) can bookmark or share it.
# The actual fetch goes to /api/chat2hamnosys/sessions/{id}/status; this
# route just serves the static HTML shell. A trailing segment is tolerated
# so /contribute/status/<id>/ also resolves.
@app.get("/contribute/status/{session_id}", include_in_schema=False)
@app.get("/contribute/status/{session_id}/", include_in_schema=False)
def serve_contribute_status(session_id: str):  # noqa: ARG001 — id read by client JS
    return FileResponse(PUBLIC_DIR / "contribute-status.html")


# Public progress dashboard (prompt 8). The extensionless /progress URL
# serves the same static HTML that the static mount would serve at
# /progress.html — having an explicit route lets us keep the URL stable
# even if we ever swap the file name.
@app.get("/progress", include_in_schema=False)
@app.get("/progress/", include_in_schema=False)
def serve_progress():
    return FileResponse(PUBLIC_DIR / "progress.html")


# Public credits page (prompt 9). Extensionless /credits URL parallels
# /progress — a stable path for citation-quality links from the README,
# the main footer, and the per-language rows on the progress dashboard.
@app.get("/credits", include_in_schema=False)
@app.get("/credits/", include_in_schema=False)
def serve_credits():
    return FileResponse(PUBLIC_DIR / "credits.html")


# Progress snapshot JSON with an explicit 15-minute cache window.
# Leaving this to the static mount would emit no Cache-Control header,
# pushing visitors to re-download the blob on every navigation. The
# snapshot is regenerated on deploy at most, so a 900-second cache is
# safe and keeps the dashboard under the 2s-on-3G budget the prompt
# calls for. Public/same-origin only — no credentials involved.
@app.get("/progress_snapshot.json", include_in_schema=False)
def serve_progress_snapshot():
    snapshot_path = PUBLIC_DIR / "progress_snapshot.json"
    if not snapshot_path.exists():
        return Response(
            content='{"error": "snapshot not yet generated"}',
            status_code=503,
            media_type="application/json",
            headers={"Cache-Control": "no-store"},
        )
    return FileResponse(
        snapshot_path,
        media_type="application/json",
        headers={"Cache-Control": "public, max-age=900"},
    )


# Contributor's session dashboard (prompt 14 step 8). Lists every
# session this browser has a token for; pure client-side aggregation
# reading sessionStorage and hitting /sessions/{id}/status for each.
# No login, no server-side identity — if the browser is cleared, the
# list is empty but individual status URLs still resolve.
@app.get("/contribute/me", include_in_schema=False)
@app.get("/contribute/me/", include_in_schema=False)
def serve_contribute_me():
    return FileResponse(PUBLIC_DIR / "contribute-me.html")


# HTTP 404 handler: browser navigations get the designed 404.html (same
# header/footer as every other page). API clients and asset requests
# continue to receive plain JSON — we gate on path prefix and Accept.
@app.exception_handler(StarletteHTTPException)
async def _kozha_http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        path = request.url.path or ""
        accept = request.headers.get("accept", "")
        wants_html = "text/html" in accept or accept == ""
        is_api = path.startswith("/api/")
        is_asset = path.startswith(("/data/", "/cwa/", "/styles/", "/fonts/"))
        if wants_html and not is_api and not is_asset:
            return FileResponse(PUBLIC_DIR / "404.html", status_code=404)
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


# Polish-13 §5: 5xx surface. Any uncaught exception returns a small
# JSON body with a request_id the user can quote when reporting an
# error. The stack trace is written to the server log (where operators
# can correlate by request_id) but never echoed to the client —
# responses can carry contextual data from locals, and we don't want
# to leak that across the trust boundary.
@app.exception_handler(Exception)
async def _kozha_generic_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "") if hasattr(request, "state") else ""
    logger.error(
        "unhandled server error",
        extra={"ctx": {
            "request_id": request_id,
            "path": request.url.path,
            "exc_type": type(exc).__name__,
        }},
        exc_info=True,
    )
    return JSONResponse(
        {
            "error": "internal server error",
            "request_id": request_id,
            "detail": "An unexpected error occurred. Please include this request_id when reporting the issue.",
        },
        status_code=500,
        headers={"X-Request-ID": request_id} if request_id else None,
    )


# Seed per-language database-size and reviewed-signs gauges from the
# current progress snapshot. If the snapshot doesn't exist yet (first
# boot before the cron runs), the gauges stay at zero — the next
# snapshot writes populate them. A malformed snapshot is logged at
# INFO and ignored; the server still boots.
_SNAPSHOT_JSON = PUBLIC_DIR / "progress_snapshot.json"
if _SNAPSHOT_JSON.exists():
    try:
        _obs.refresh_database_gauges_from_snapshot(
            json.loads(_SNAPSHOT_JSON.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError) as _snap_err:
        logger.info("startup snapshot gauges skipped: %s", _snap_err)


app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")
