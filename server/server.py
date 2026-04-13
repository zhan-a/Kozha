from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from typing import Dict, List, Optional
from collections import OrderedDict
from threading import Lock
import json
import re
import spacy
from spacy.matcher import PhraseMatcher

class TextRequest(BaseModel):
    text: str
    language: str = "en"
    sign_language: str = "bsl"

APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parent
PUBLIC_DIR = REPO_ROOT / "public"
DATA_DIR = REPO_ROOT / "data"

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

def plan_from_text(text: str, language: str = "en", sign_language: str = "bsl") -> Dict[str, object]:
    text = (text or "").strip()
    if not text:
        return {"error": "Empty text."}
    rewritten = process_text(text, language, sign_language)
    return {"allowed": [], "raw": text, "final": rewritten, "language": language, "sign_language": sign_language}

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
    return plan_from_text(req.text, req.language, req.sign_language)

if DATA_DIR.exists():
    app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

@app.get("/", include_in_schema=False)
def serve_index():
    return FileResponse(PUBLIC_DIR / "index.html")

app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")
