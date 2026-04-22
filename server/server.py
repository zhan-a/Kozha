import sys
from pathlib import Path

# chat2hamnosys uses flat imports (``from session import ...``); make its
# package directory importable before anything inside it is referenced.
_CHAT2HAMNOSYS_ROOT = Path(__file__).resolve().parent.parent / "backend" / "chat2hamnosys"
if str(_CHAT2HAMNOSYS_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHAT2HAMNOSYS_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List, Optional
from collections import OrderedDict
from threading import Lock
import json
import re
import logging
import spacy
from spacy.matcher import PhraseMatcher

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

@app.get("/api/health")
def health():
    return {"ok": True}

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
def api_translate_text(req: TranslateTextRequest):
    # Prompt 4: accept either `text` or `source_text`; the JS client sends
    # both, but older extension builds still only send `text`.
    text = (req.text or req.source_text or "").strip()
    if not text or req.source_lang == req.target_lang:
        return {"translated": text}
    # Refuse unknown target sign languages explicitly rather than silently
    # falling through to argos (which would succeed but produce output the
    # client cannot render into sign).
    if req.target_sign_lang and req.target_sign_lang not in _SUPPORTED_SIGN_LANGS:
        logger.warning(
            "translate-text: unknown target_sign_lang %r", req.target_sign_lang,
        )
        return {
            "translated": text,
            "error": f"unknown target sign language: {req.target_sign_lang}",
        }
    # Refuse source/target pairs with no argos route rather than silently
    # returning the original text and leaving the client unsure whether
    # translation ran.
    if not _translation_route_supported(req.source_lang, req.target_lang):
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
                    "argostranslate returned non-string type %s for %s→%s: %r",
                    type(result).__name__, req.source_lang, req.target_lang, result,
                )
                return {
                    "translated": text,
                    "error": f"translation returned unexpected type {type(result).__name__}",
                }
        return {"translated": result}
    except Exception as e:
        logger.error("Translation error: %s", e)
        return {"translated": text, "error": str(e)}

@app.post("/api/translate")
def api_translate(req: TranslateRequest):
    glosses = text_to_glosses(req.text, req.source_lang)
    return {"glosses": glosses, "raw": req.text}

@app.post("/api/translate/batch")
def api_translate_batch(req: BatchTranslateRequest):
    results = []
    for seg in req.segments:
        glosses = text_to_glosses(seg.text, req.source_lang)
        results.append({"glosses": glosses, "start": seg.start, "duration": seg.duration})
    return {"results": results}

@app.post("/api/plan")
def api_plan(req: TextRequest):
    return plan_from_text(
        req.text,
        req.language,
        req.sign_language,
        reviewed_only=req.reviewed_only,
    )


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


app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")
