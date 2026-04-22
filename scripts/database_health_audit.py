"""Database health audit, quarantine, and auto-repair pass.

Scope (prompt 7):
  * Parse every ``data/*.sigml`` file and classify each sign entry into
    one of: ok, repaired-in-place, or quarantine.
  * Apply conservative auto-repairs for well-documented tag fixes
    (``hamupperarm`` -> ``hamUpperarm``, ``hamindxfinger`` ->
    ``hamindexfinger``) and NFC-normalize gloss attributes.
  * Move unambiguously broken entries out of the active file into a
    sidecar ``data/<stem>_quarantine.sigml`` with the failure reason
    preserved as an XML comment. Entries are never deleted.
  * Emit per-file health reports, a cross-language gloss coverage
    matrix, a suspected-duplicate HamNoSys clusters report, an
    alphabet coverage report, an auto-repair log, and a
    dashboard-friendly statistics JSON.

The script is idempotent: running it twice produces the same output.
The set of "safe repair" rules is intentionally small; adding new
rules must be accompanied by expert review.

Usage::

    python3 scripts/database_health_audit.py --write
    python3 scripts/database_health_audit.py --dry-run  # default

Outputs (relative to repo root)::

    docs/polish/07-database-health/<stem>.md   per-file report
    docs/polish/07-coverage-matrix.md          cross-language gloss matrix
    docs/polish/07-suspected-duplicates.md     shared-HamNoSys clusters
    docs/polish/07-alphabet-coverage.md        alphabet coverage
    docs/polish/07-auto-repairs.md             auto-repair log
    docs/polish/07-summary.md                  roll-up stats
    docs/polish/07-database-stats.json         dashboard-consumable
    data/<stem>_quarantine.sigml               quarantined entries
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DOCS_DIR = REPO_ROOT / "docs" / "polish"
PER_FILE_DIR = DOCS_DIR / "07-database-health"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.scan_unknown_hns_tags import (  # noqa: E402
    CONTAINER_TAGS,
    extract_known_tags,
)

# -- Safe repairs ------------------------------------------------------------
# Only tag names whose correct CWASA spelling is unambiguous belong here.
# Every entry is: unknown -> canonical. The rewrite is a pure alias and
# does not alter handshape or motion semantics.
SAFE_TAG_REPAIRS: dict[str, str] = {
    "hamupperarm": "hamUpperarm",
    "hamindxfinger": "hamindexfinger",
}

# -- Language metadata (for alphabet coverage) -------------------------------
# Each entry describes the letters a fingerspelling alphabet is expected to
# cover. The coverage check compares these against the gloss set of the
# alphabet .sigml. Letters missing from the database surface in the report
# so the community can prioritize contributions.
_BASE_LATIN = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
EXPECTED_ALPHABETS: dict[str, dict] = {
    "asl": {
        "file": "asl_alphabet_sigml.sigml",
        "letters": list(_BASE_LATIN),
        "label": "American Sign Language",
    },
    "bsl": {
        "file": "bsl_alphabet_sigml.sigml",
        "letters": list(_BASE_LATIN),
        "label": "British Sign Language",
    },
    "dgs": {
        "file": "dgs_alphabet_sigml.sigml",
        "letters": list(_BASE_LATIN) + ["Ä", "Ö", "Ü", "ß"],
        "label": "German Sign Language",
    },
    "lsf": {
        "file": "lsf_alphabet_sigml.sigml",
        "letters": list(_BASE_LATIN),
        "label": "French Sign Language",
    },
    "pjm": {
        "file": "pjm_alphabet_sigml.sigml",
        # Polish alphabet: 32 letters, no Q/V/X in native alphabet but
        # included for loanword fingerspelling.
        "letters": list(_BASE_LATIN)
        + ["Ą", "Ć", "Ę", "Ł", "Ń", "Ó", "Ś", "Ź", "Ż"],
        "label": "Polish Sign Language",
    },
    "ngt": {
        "file": "ngt_alphabet_sigml.sigml",
        "letters": list(_BASE_LATIN),
        "label": "Dutch Sign Language (NGT)",
    },
}

# Sign-language database files (non-alphabet) and their sign-language codes.
# The codes align with ``_SUPPORTED_SIGN_LANGS`` in ``server/server.py`` so
# the coverage matrix can cross-reference.
LANG_FILES: list[tuple[str, str, str]] = [
    # (file, lang_code, display_name)
    ("hamnosys_bsl_version1.sigml", "bsl", "British"),
    ("French_SL_LSF.sigml", "lsf", "French (LSF)"),
    ("German_SL_DGS.sigml", "dgs", "German (DGS)"),
    ("Greek_SL_GSL.sigml", "gsl", "Greek"),
    ("Polish_SL_PJM.sigml", "pjm", "Polish (PJM)"),
    ("Indian_SL.sigml", "isl", "Indian"),
    ("Dutch_SL_NGT.sigml", "ngt", "Dutch (NGT)"),
    ("Vietnamese_SL.sigml", "vsl", "Vietnamese"),
    ("Kurdish_SL.sigml", "kurdish", "Kurdish"),
    ("Algerian_SL.sigml", "algerian", "Algerian"),
    ("Bangla_SL.sigml", "bangla", "Bangla"),
    ("Filipino_SL.sigml", "fsl", "Filipino"),
]

# Top-500 English word frequency list — a lightly curated subset of the
# Google-books / subtlex unigram frequencies restricted to concrete nouns,
# common verbs, pronouns, and function words that a fingerspelling or gloss
# dictionary would plausibly cover. Embedded inline to keep the audit
# self-contained (no network access during CI).
TOP_500_EN: list[str] = """
the be to of and a in that have it for not on with he as you do at this
but his by from they we say her she or an will my one all would there
their what so up out if about who get which go me when make can like time
no just him know take people into year your good some could them see other
than then now look only come its over think also back after use two how our
work first well way even new want because any these give day most us is was
are were am being been has had having does did doing go went gone said come
came came went see saw seen got gotten hear heard here there where why what
when who whom whose how under above below near far water food fire earth air
wind sun moon star sky sea land mountain river tree grass flower leaf fruit
apple orange banana bread milk tea coffee sugar salt egg meat fish chicken
cow horse dog cat bird mouse lion tiger bear wolf elephant monkey snake
house home school church hospital shop store market bank office police
street road path bridge door window wall floor roof table chair bed desk
book paper pen pencil computer phone car bus train plane boat ship bicycle
man woman boy girl child baby father mother brother sister son daughter
uncle aunt grandfather grandmother family friend teacher doctor nurse farmer
name age birthday life death body head hair face eye ear nose mouth tooth
lip tongue neck shoulder arm hand finger thumb leg foot knee back chest
heart lung blood bone skin muscle sick hurt pain help sleep dream eat drink
cook wash dress wear speak talk listen read write learn teach play sing
dance draw paint run walk jump swim sit stand lie wake open close break
build make fix clean cut push pull hold carry throw catch give take buy
sell pay money rich poor big small tall short long wide narrow high low
fast slow hot cold warm cool wet dry clean dirty good bad right wrong easy
hard hard soft heavy light dark bright quiet loud strong weak old young new
happy sad angry scared tired hungry thirsty red blue green yellow black
white brown pink gray purple number one two three four five six seven eight
nine ten hundred thousand million today yesterday tomorrow morning afternoon
evening night week month year hour minute second always never sometimes
often rarely before after during until since while quickly slowly yes no
maybe please thank sorry hello goodbye because why how which question answer
country city village island country language english french german spanish
italian polish dutch greek russian arabic chinese japanese korean indian
sign deaf hearing
""".split()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EntryAudit:
    """Health audit of a single ``<hns_sign>`` or ``<hamgestural_sign>`` entry."""

    index: int  # position within the parent file (0-based)
    tag: str  # "hns_sign" or "hamgestural_sign"
    original_gloss: str | None
    normalized_gloss: str | None
    unknown_tags: list[str] = field(default_factory=list)
    repair_counts: dict[str, int] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)  # e.g. "empty_gloss"
    hamnosys_manual_tag_count: int = 0

    @property
    def has_safe_repairs(self) -> bool:
        return bool(self.repair_counts)

    @property
    def quarantine_reason(self) -> str | None:
        if not self.issues:
            return None
        # Ordering: most informative first. "has_unresolved_unknown_tags" is
        # implied by "unknown_tags_after_repair" but we surface both names.
        return "; ".join(self.issues)

    @property
    def must_quarantine(self) -> bool:
        # These issue codes indicate the entry can never be served.
        # "duplicate_gloss" is flagged but not quarantined (first-wins).
        quarantining = {
            "empty_gloss",
            "no_hamnosys_content",
            "object_literal_in_entry",
            "malformed_xml",
            "gestural_sign_no_gloss",
            "unknown_tags_after_repair",
        }
        return any(code in quarantining for code in self.issues)


@dataclass
class FileAudit:
    path: Path
    language_code: str
    total_entries: int = 0
    kept_entries: int = 0
    repaired_entries: int = 0
    quarantined_entries: int = 0
    duplicate_entries: int = 0
    glosses_active: set[str] = field(default_factory=set)
    glosses_active_lower: set[str] = field(default_factory=set)
    audits: list[EntryAudit] = field(default_factory=list)
    issue_counts: Counter = field(default_factory=Counter)
    repair_totals: Counter = field(default_factory=Counter)


# ---------------------------------------------------------------------------
# Audit logic
# ---------------------------------------------------------------------------


_TAG_REGEX = re.compile(r"<(ham[A-Za-z][\w]*)")


def _element_xml(elem: ET.Element) -> str:
    return ET.tostring(elem, encoding="unicode")


def _scan_tags(xml_text: str) -> list[str]:
    return _TAG_REGEX.findall(xml_text)


def _apply_safe_repairs(xml_text: str) -> tuple[str, dict[str, int]]:
    """Rewrite known-safe unknown tags to their canonical spelling.

    Returns the rewritten XML and a mapping of repair-name -> count.
    """
    repairs: dict[str, int] = {}
    out = xml_text
    for bad, good in SAFE_TAG_REPAIRS.items():
        # Match the tag whether self-closing (``<bad/>``) or paired
        # (``<bad></bad>``). Use word-boundary style lookahead to avoid
        # partial matches such as ``<hamupperarmsomething>``.
        pattern = re.compile(rf"<{bad}\b")
        if pattern.search(out):
            count = len(pattern.findall(out))
            out = pattern.sub(f"<{good}", out)
            # Closing tags ``</bad>`` rarely occur (HamNoSys tags are
            # self-closing) but rewrite them for symmetry.
            close = re.compile(rf"</{bad}>")
            out = close.sub(f"</{good}>", out)
            repairs[bad] = repairs.get(bad, 0) + count
    return out, repairs


def _normalize_gloss(raw: str | None) -> tuple[str | None, bool, bool]:
    """Return (normalized, changed_by_nfc, stripped_whitespace)."""
    if raw is None:
        return None, False, False
    nfc = unicodedata.normalize("NFC", raw)
    stripped = nfc.strip()
    return stripped, (nfc != raw), (stripped != nfc)


def _audit_entry(
    index: int,
    elem: ET.Element,
    known_tags: set[str],
    seen_glosses: set[str],
    seen_glosses_lower: set[str],
) -> tuple[EntryAudit, ET.Element | None]:
    """Audit one entry. Returns (audit_record, possibly-repaired-element).

    The returned element is the same object as ``elem`` if no in-place
    repairs applied; otherwise it's a fresh parse of the repaired XML.
    """
    tag = elem.tag
    raw_gloss = elem.get("gloss")
    norm_gloss, nfc_changed, whitespace_stripped = _normalize_gloss(raw_gloss)

    audit = EntryAudit(
        index=index,
        tag=tag,
        original_gloss=raw_gloss,
        normalized_gloss=norm_gloss,
    )

    # ---- entry-level structural checks ----
    if tag == "hamgestural_sign" and not raw_gloss:
        # hamgestural_sign entries in the wild carry no gloss; they cannot
        # be looked up by the translator regardless.
        audit.issues.append("gestural_sign_no_gloss")
    elif tag == "hns_sign" and (raw_gloss is None or not raw_gloss.strip()):
        audit.issues.append("empty_gloss")

    xml_text = _element_xml(elem)
    if "[object Object]" in xml_text:
        audit.issues.append("object_literal_in_entry")

    # Child check: must have hamnosys_manual with at least one tag, OR be
    # a hamgestural_sign with a sign_manual child that has tags. Entries
    # with only hamnosys_nonmanual and no manual content cannot render.
    manual_children = list(elem.findall("hamnosys_manual"))
    if not manual_children:
        manual_children = list(elem.findall("sign_manual"))
    manual_tag_count = 0
    for mc in manual_children:
        manual_tag_count += sum(
            1 for _ in mc.iter() if _.tag.startswith("ham")
        )
    audit.hamnosys_manual_tag_count = manual_tag_count
    if manual_tag_count == 0:
        audit.issues.append("no_hamnosys_content")

    # ---- lexical scan + safe repair ----
    repaired_xml, repair_counts = _apply_safe_repairs(xml_text)
    audit.repair_counts = repair_counts

    tags_seen = _scan_tags(repaired_xml)
    unknown = sorted({t for t in tags_seen if t not in CONTAINER_TAGS and t not in known_tags})
    audit.unknown_tags = unknown
    if unknown:
        audit.issues.append("unknown_tags_after_repair")

    # ---- XML well-formedness of the (possibly repaired) block ----
    repaired_elem: ET.Element | None = elem
    if repair_counts or nfc_changed or whitespace_stripped:
        # Rebuild from repaired XML to ensure changes survive write-back.
        try:
            repaired_elem = ET.fromstring(repaired_xml)
            if nfc_changed or whitespace_stripped:
                if norm_gloss is not None:
                    repaired_elem.set("gloss", norm_gloss)
        except ET.ParseError:
            audit.issues.append("malformed_xml")
            repaired_elem = None

    # ---- duplicate detection ----
    # First-wins. Second occurrence of the same gloss is flagged but still
    # kept in the active file (the downstream loader enforces first-wins
    # on lookup; removing duplicates from disk would lose data).
    if norm_gloss and norm_gloss.strip():
        key = norm_gloss.strip()
        if key.lower() in seen_glosses_lower:
            audit.issues.append("duplicate_gloss")
        else:
            seen_glosses.add(key)
            seen_glosses_lower.add(key.lower())

    if nfc_changed:
        audit.issues.append("gloss_not_nfc")
    if whitespace_stripped:
        audit.issues.append("gloss_leading_trailing_whitespace")

    return audit, repaired_elem


def _extract_hns_tag_sequence(elem: ET.Element) -> tuple[str, ...]:
    """Return the ordered sequence of ham* tag names under <hamnosys_manual>.

    Used to compare signs across languages for suspected copy-paste.
    """
    tags: list[str] = []
    manual = elem.find("hamnosys_manual")
    if manual is None:
        manual = elem.find("sign_manual")
    if manual is None:
        return tuple()
    for e in manual.iter():
        if e is manual:
            continue
        if e.tag.startswith("ham"):
            tags.append(e.tag)
    return tuple(tags)


def audit_file(path: Path, known_tags: set[str]) -> tuple[FileAudit, list[tuple[EntryAudit, ET.Element]], list[tuple[EntryAudit, ET.Element]]]:
    """Audit one .sigml file.

    Returns ``(file_audit, keep_list, quarantine_list)`` where the keep/
    quarantine lists contain ``(audit, element)`` pairs ready to be
    written back to their respective files. The elements in keep_list
    reflect any safe in-place repairs.
    """
    audit = FileAudit(path=path, language_code=path.stem)
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError as exc:
        audit.issue_counts["file_parse_error"] += 1
        audit.audits.append(
            EntryAudit(
                index=-1,
                tag="file",
                original_gloss=None,
                normalized_gloss=None,
                issues=[f"file_parse_error: {exc}"],
            )
        )
        return audit, [], []

    seen_glosses: set[str] = set()
    seen_glosses_lower: set[str] = set()
    keep_list: list[tuple[EntryAudit, ET.Element]] = []
    quarantine_list: list[tuple[EntryAudit, ET.Element]] = []

    for idx, child in enumerate(root):
        audit.total_entries += 1
        entry_audit, repaired_elem = _audit_entry(
            idx, child, known_tags, seen_glosses, seen_glosses_lower
        )
        audit.audits.append(entry_audit)
        for code in entry_audit.issues:
            audit.issue_counts[code] += 1
        for rname, rcount in entry_audit.repair_counts.items():
            audit.repair_totals[rname] += rcount

        if entry_audit.must_quarantine or repaired_elem is None:
            quarantine_list.append((entry_audit, child))
            audit.quarantined_entries += 1
        else:
            keep_list.append((entry_audit, repaired_elem))
            audit.kept_entries += 1
            if entry_audit.has_safe_repairs or "gloss_not_nfc" in entry_audit.issues or "gloss_leading_trailing_whitespace" in entry_audit.issues:
                audit.repaired_entries += 1
            if entry_audit.normalized_gloss and entry_audit.normalized_gloss.strip():
                audit.glosses_active.add(entry_audit.normalized_gloss.strip())
                audit.glosses_active_lower.add(entry_audit.normalized_gloss.strip().lower())
            if "duplicate_gloss" in entry_audit.issues:
                audit.duplicate_entries += 1

    return audit, keep_list, quarantine_list


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


_XML_DECL = '<?xml version="1.0" encoding="utf-8"?>\n'


def _ensure_xml_decl(path: Path) -> bool:
    """Return True if the file starts with an XML declaration."""
    head = path.read_text(encoding="utf-8", errors="replace")[:256]
    return head.lstrip().startswith("<?xml")


def _serialize_elements(
    root_tag: str,
    root_attrib: dict[str, str],
    entries: Iterable[tuple[EntryAudit, ET.Element]],
    *,
    with_comments: bool,
    include_xml_decl: bool,
) -> str:
    """Serialize a set of entries back into a SiGML document.

    ``with_comments`` adds per-entry XML comments listing the quarantine
    reasons before each element — used for the sidecar file.
    """
    # Build a fresh root so we don't mutate the original tree.
    new_root = ET.Element(root_tag, root_attrib or {})
    # Track number of entries for the ``count`` attribute on
    # ``sigml_collection``. We don't emit count on ``sigml`` roots
    # (alphabet files) since the original never carries one.
    count = 0
    body_parts: list[str] = []
    for audit, elem in entries:
        if with_comments:
            reason = audit.quarantine_reason or "unspecified"
            # XML comments cannot contain '--' so escape conservatively.
            safe = reason.replace("--", "- -")
            body_parts.append(f"  <!-- quarantined: {safe} -->\n")
        # ElementTree.tostring may append a trailing newline; strip it so
        # we control the inter-entry spacing and don't produce double
        # blank lines in the output.
        frag = ET.tostring(elem, encoding="unicode").rstrip() + "\n"
        body_parts.append("  " + frag + "\n")
        count += 1

    if root_tag == "sigml_collection":
        new_root.set("language", root_attrib.get("language", ""))
        new_root.set("count", str(count))

    # Hand-assemble the document to preserve the structure contributors
    # are used to reading (one entry per line-ish, root on its own line).
    head = _XML_DECL if include_xml_decl else ""
    if root_tag == "sigml_collection":
        lang = new_root.get("language") or ""
        opener = f'<sigml_collection language="{lang}" count="{count}">\n\n'
    else:
        opener = "<sigml>\n"
    closer = "\n</sigml_collection>\n" if root_tag == "sigml_collection" else "\n</sigml>\n"
    return head + opener + "".join(body_parts) + closer


def write_cleaned_file(path: Path, keep: list[tuple[EntryAudit, ET.Element]]) -> None:
    tree = ET.parse(path)
    root = tree.getroot()
    include_decl = _ensure_xml_decl(path)
    serialized = _serialize_elements(
        root.tag,
        dict(root.attrib),
        keep,
        with_comments=False,
        include_xml_decl=include_decl,
    )
    path.write_text(serialized, encoding="utf-8")


def write_quarantine_file(
    source: Path,
    quarantine: list[tuple[EntryAudit, ET.Element]],
) -> Path | None:
    """Write broken entries to ``data/<stem>_quarantine.sigml``.

    If there's nothing to quarantine, the sidecar is removed so reruns
    don't leave stale sidecars around.
    """
    sidecar = source.with_name(f"{source.stem}_quarantine.sigml")
    if not quarantine:
        if sidecar.exists():
            sidecar.unlink()
        return None
    tree = ET.parse(source)
    root = tree.getroot()
    include_decl = _ensure_xml_decl(source)
    serialized = _serialize_elements(
        root.tag,
        dict(root.attrib),
        quarantine,
        with_comments=True,
        include_xml_decl=include_decl,
    )
    sidecar.write_text(serialized, encoding="utf-8")
    return sidecar


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


_ISSUE_LABELS = {
    "empty_gloss": "Empty gloss",
    "no_hamnosys_content": "No HamNoSys manual tags",
    "object_literal_in_entry": "Contains literal `[object Object]`",
    "malformed_xml": "Entry XML is malformed",
    "gestural_sign_no_gloss": "`hamgestural_sign` without a gloss",
    "unknown_tags_after_repair": "Unknown HamNoSys tags (post-repair)",
    "duplicate_gloss": "Duplicate gloss within file (first-wins)",
    "gloss_not_nfc": "Gloss required NFC normalization",
    "gloss_leading_trailing_whitespace": "Gloss had leading/trailing whitespace",
    "file_parse_error": "File-level XML parse error",
}


def write_per_file_report(audit: FileAudit, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{audit.path.stem}.md"

    lines = [
        f"# Database health — `data/{audit.path.name}`",
        "",
        "Generated by `scripts/database_health_audit.py` (prompt 7).",
        "",
        "## Summary",
        "",
        f"| metric | value |",
        f"|---|---|",
        f"| total entries | {audit.total_entries} |",
        f"| kept in active file | {audit.kept_entries} |",
        f"| quarantined | {audit.quarantined_entries} |",
        f"| repaired in place | {audit.repaired_entries} |",
        f"| duplicate glosses flagged | {audit.duplicate_entries} |",
        f"| unique glosses (active) | {len(audit.glosses_active)} |",
        "",
    ]

    if audit.issue_counts:
        lines.extend(["## Issues", "", "| code | count | meaning |", "|---|---|---|"])
        for code, n in sorted(audit.issue_counts.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"| `{code}` | {n} | {_ISSUE_LABELS.get(code, '—')} |")
        lines.append("")
    else:
        lines.extend(["## Issues", "", "_No issues detected._", ""])

    if audit.repair_totals:
        lines.extend(
            [
                "## Auto-repairs applied",
                "",
                "Safe tag renames that do not alter handshape or motion semantics.",
                "",
                "| old tag | new tag | occurrences |",
                "|---|---|---|",
            ]
        )
        for tag, count in sorted(audit.repair_totals.items(), key=lambda x: -x[1]):
            lines.append(f"| `<{tag}/>` | `<{SAFE_TAG_REPAIRS[tag]}/>` | {count} |")
        lines.append("")

    # Sample quarantined entries (up to 20) for orientation.
    quarantined = [a for a in audit.audits if a.must_quarantine]
    if quarantined:
        lines.extend(
            [
                "## Quarantined entries (sample)",
                "",
                f"Full list in `data/{audit.path.stem}_quarantine.sigml`. First 20 shown:",
                "",
                "| # | gloss | tag | reason |",
                "|---|---|---|---|",
            ]
        )
        for a in quarantined[:20]:
            gloss = (a.normalized_gloss or "(none)").replace("|", "\\|")
            lines.append(
                f"| {a.index} | `{gloss}` | `{a.tag}` | {a.quarantine_reason or '—'} |"
            )
        if len(quarantined) > 20:
            lines.append(f"| ... | +{len(quarantined) - 20} more | | |")
        lines.append("")

    # Surface unresolved unknown tags (after safe repair). These are the
    # remaining expert-review candidates.
    unknown_counter: Counter = Counter()
    for a in audit.audits:
        for t in a.unknown_tags:
            unknown_counter[t] += 1
    if unknown_counter:
        lines.extend(
            [
                "## Unknown HamNoSys tags (post-repair)",
                "",
                "Tags not recognised by CWASA `tokenNameMap` after safe renames.",
                "Expert review required — semantics could not be inferred.",
                "",
                "| tag | occurrences |",
                "|---|---|",
            ]
        )
        for tag, count in sorted(unknown_counter.items(), key=lambda x: -x[1]):
            lines.append(f"| `<{tag}/>` | {count} |")
        lines.append("")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Cross-language reports
# ---------------------------------------------------------------------------


def _lookup_gloss_base(g: str) -> str:
    """Rough base-word key for cross-language matching.

    Aligns with ``server/review_metadata.py:gloss_base`` so the matrix
    reflects what the translator would actually resolve.
    """
    s = g.strip().lower()
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"#\d+$", "", s)
    s = re.sub(r"\d+[a-z]?\^?$", "", s)
    s = re.sub(r"^_num-", "", s)
    s = re.sub(r"_\(.*?\)", "", s)
    # Collapse punctuation to spaces, trim.
    s = re.sub(r"[^a-z0-9]+", " ", s, flags=re.IGNORECASE)
    return s.strip()


def write_coverage_matrix(
    per_language_bases: dict[str, set[str]],
    out_path: Path,
) -> None:
    """Produce a table of top-500 English concepts vs language coverage."""
    words = [w.lower() for w in TOP_500_EN if w]
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique_words: list[str] = []
    for w in words:
        if w in seen:
            continue
        seen.add(w)
        unique_words.append(w)

    lang_codes = sorted(per_language_bases.keys())

    # Header.
    lines = [
        "# Cross-language gloss coverage matrix",
        "",
        "For each of the top-500 common English concepts, shows which sign",
        "languages have a matching active-database entry. Matching uses the",
        "same base-form rule (`gloss_base`) the translator uses, so a word",
        "like `fruit` also matches `fruit(n)#1`.",
        "",
        f"Languages analysed: {len(lang_codes)} ({', '.join(lang_codes)}).",
        "",
        "| concept | " + " | ".join(lang_codes) + " | total |",
        "|---|" + "|".join(["---"] * (len(lang_codes) + 1)) + "|",
    ]

    total_hits = Counter()
    for concept in unique_words:
        row = []
        count = 0
        for code in lang_codes:
            bases = per_language_bases[code]
            if concept in bases:
                row.append("y")
                count += 1
            else:
                row.append("")
        if count == 0:
            # Hide rows where no language has it — keeps the matrix usable.
            continue
        total_hits[concept] = count
        lines.append(f"| {concept} | " + " | ".join(row) + f" | {count} |")

    lines.extend(
        [
            "",
            "## Coverage rollup",
            "",
            f"Of the top-500 concepts, {sum(1 for c in unique_words if c in total_hits)}"
            f" have at least one language with a matching gloss.",
            "",
            "| language | concepts covered (of 500) | total active glosses |",
            "|---|---|---|",
        ]
    )
    for code in lang_codes:
        bases = per_language_bases[code]
        covered = sum(1 for c in unique_words if c in bases)
        lines.append(f"| {code} | {covered} | {len(bases)} |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_suspected_duplicates(
    per_language_signs: dict[str, dict[str, tuple[str, ...]]],
    out_path: Path,
) -> None:
    """Find clusters of 3+ languages sharing identical HamNoSys for a gloss.

    A shared HamNoSys across two languages for a gloss like ``"ONE"`` is
    expected; identical sequences across many languages for a concept
    like ``"FRUIT"`` is suspicious and hints at copy-paste drift.
    """
    # Invert: gloss_base -> {language -> hamnosys_tuple}
    by_gloss: dict[str, dict[str, tuple[str, ...]]] = defaultdict(dict)
    for lang, signs in per_language_signs.items():
        for gloss, hns in signs.items():
            if not hns:
                continue
            base = _lookup_gloss_base(gloss)
            if not base:
                continue
            # First-wins inside a language: if we've already recorded a
            # hns_tuple for this base, keep the earlier one.
            by_gloss[base].setdefault(lang, hns)

    # For each concept, find the largest cluster of languages sharing the
    # same sequence.
    rows: list[tuple[int, str, tuple[str, ...], list[str]]] = []
    for base, lang_map in by_gloss.items():
        if len(lang_map) < 2:
            continue
        by_seq: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for lang, seq in lang_map.items():
            by_seq[seq].append(lang)
        for seq, langs in by_seq.items():
            if len(langs) >= 3:
                rows.append((len(langs), base, seq, sorted(langs)))

    rows.sort(key=lambda r: (-r[0], r[1]))

    lines = [
        "# Suspected cross-language HamNoSys duplicates",
        "",
        "Concepts where the exact same HamNoSys tag sequence appears in",
        "three or more sign languages. A low edit distance across two",
        "languages is normal (shared iconicity or borrowed signs); three",
        "or more identical sequences usually indicates copy-paste between",
        "databases rather than genuine cross-language alignment.",
        "",
        f"Clusters found: {len(rows)}.",
        "",
    ]
    if rows:
        lines.extend(
            [
                "| concept | shared across | languages | sequence length |",
                "|---|---|---|---|",
            ]
        )
        for size, base, seq, langs in rows[:200]:
            lines.append(
                f"| `{base}` | {size} | {', '.join(langs)} | {len(seq)} |"
            )
        if len(rows) > 200:
            lines.append(f"\n_+{len(rows) - 200} more clusters omitted._\n")
    else:
        lines.append("_No suspicious clusters found._")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_alphabet_coverage(
    file_audits: dict[str, FileAudit],
    out_path: Path,
) -> None:
    lines = [
        "# Alphabet coverage per language",
        "",
        "Fingerspelling alphabets must cover the full writing system of the",
        "target language or the translator cannot fall through to letter-by-",
        "letter spelling for unknown words.",
        "",
        "| language | expected letters | present | missing | notes |",
        "|---|---|---|---|---|",
    ]

    # Languages with no alphabet file at all.
    alphabet_files_present = {
        audit.path.name for audit in file_audits.values()
        if "_alphabet_" in audit.path.name
    }

    for code, meta in EXPECTED_ALPHABETS.items():
        expected = meta["letters"]
        file_name = meta["file"]
        audit = file_audits.get(file_name)
        if audit is None or audit.total_entries == 0:
            present = set()
            missing = expected
            notes = f"file missing or empty: `{file_name}`"
        else:
            present_upper = {g.upper() for g in audit.glosses_active}
            present = sorted(set(expected) & present_upper)
            missing = sorted(set(expected) - present_upper)
            notes = f"`{file_name}`"
            quarantined = audit.quarantined_entries
            if quarantined:
                notes += f" ({quarantined} letter(s) quarantined)"
        lines.append(
            f"| {meta['label']} ({code}) | {len(expected)} | {len(present)} | "
            f"{', '.join(missing) if missing else 'none'} | {notes} |"
        )

    # Languages that should have an alphabet but don't have an entry in
    # EXPECTED_ALPHABETS.
    sl_codes_in_data = {code for _, code, _ in LANG_FILES}
    known_alphabets = set(EXPECTED_ALPHABETS.keys())
    missing_alphabets = sorted(sl_codes_in_data - known_alphabets)
    if missing_alphabets:
        lines.extend(
            [
                "",
                "## Languages without a fingerspelling alphabet",
                "",
                "These languages have a sign database but no alphabet file.",
                "Words that fall through the translator's vocabulary lookup",
                "cannot be fingerspelled and will be dropped.",
                "",
                "| language | note |",
                "|---|---|",
            ]
        )
        for code in missing_alphabets:
            lines.append(f"| {code} | no `{code}_alphabet_sigml.sigml` in `data/` |")

    # Stray alphabet files not listed in EXPECTED_ALPHABETS (sanity check
    # against a surprise new alphabet file that the audit hasn't been
    # updated for).
    expected_filenames = {meta["file"] for meta in EXPECTED_ALPHABETS.values()}
    stray = sorted(alphabet_files_present - expected_filenames)
    if stray:
        lines.extend(
            [
                "",
                "## Unrecognised alphabet files",
                "",
                "These files look like alphabets but are not in the",
                "expected-alphabets table. Update `EXPECTED_ALPHABETS` in",
                "`scripts/database_health_audit.py` to bring them in scope.",
                "",
                *(f"- `data/{name}`" for name in stray),
            ]
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_auto_repairs_log(
    file_audits: dict[str, FileAudit],
    out_path: Path,
) -> None:
    lines = [
        "# Auto-repairs applied",
        "",
        "The audit applies only repairs whose correctness is independently",
        "verifiable (i.e. the unknown tag has a single canonical CWASA",
        "name that differs from the misspelt form by case or a clear",
        "typographical error). Anything ambiguous is quarantined instead.",
        "",
        "| repair | description |",
        "|---|---|",
    ]
    repair_descriptions = {
        "hamupperarm": "Case-sensitive alias for `hamUpperarm` (CWASA's `tokenNameMap` has capital `U`).",
        "hamindxfinger": "Typo for `hamindexfinger` (missing `e`).",
    }
    for old, new in SAFE_TAG_REPAIRS.items():
        desc = repair_descriptions.get(old, "see commit history")
        lines.append(f"| `<{old}/>` → `<{new}/>` | {desc} |")
    lines.append("")

    # Per-file totals.
    total_per_repair: Counter = Counter()
    lines.extend(["## Per-file occurrences", "", "| file | repair | count |", "|---|---|---|"])
    had_any = False
    for fname, audit in sorted(file_audits.items()):
        for rname, rcount in sorted(audit.repair_totals.items()):
            lines.append(f"| `{fname}` | `<{rname}/>` → `<{SAFE_TAG_REPAIRS[rname]}/>` | {rcount} |")
            total_per_repair[rname] += rcount
            had_any = True
    if not had_any:
        lines.append("| _(no auto-repairs applied)_ |  |  |")
    lines.append("")

    # Unicode-level repairs (NFC, whitespace).
    nfc_total = sum(a.issue_counts.get("gloss_not_nfc", 0) for a in file_audits.values())
    ws_total = sum(a.issue_counts.get("gloss_leading_trailing_whitespace", 0) for a in file_audits.values())
    lines.extend(
        [
            "## Gloss-level repairs",
            "",
            f"- Entries whose gloss required NFC Unicode normalization: **{nfc_total}**",
            f"- Entries whose gloss had leading/trailing whitespace stripped: **{ws_total}**",
            "",
        ]
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _alphabet_coverage_for_code(
    code: str,
    file_audits: dict[str, FileAudit],
) -> dict:
    """Return alphabet-coverage numbers for one sign-language code.

    ``prompt 8``'s dashboard wants per-language presence of letters so it
    can render a "X / N letters covered" chip without reparsing the
    report markdown.
    """
    meta = EXPECTED_ALPHABETS.get(code)
    if meta is None:
        return {"expected": 0, "present": 0, "missing": []}
    audit = file_audits.get(meta["file"])
    if audit is None:
        return {"expected": len(meta["letters"]), "present": 0, "missing": list(meta["letters"])}
    present_upper = {g.upper() for g in audit.glosses_active}
    expected = meta["letters"]
    present = [letter for letter in expected if letter in present_upper]
    missing = [letter for letter in expected if letter not in present_upper]
    return {
        "expected": len(expected),
        "present": len(present),
        "missing": missing,
    }


def write_summary_and_stats(
    file_audits: dict[str, FileAudit],
    per_language_bases: dict[str, set[str]],
    summary_path: Path,
    stats_path: Path,
) -> None:
    total = sum(a.total_entries for a in file_audits.values())
    kept = sum(a.kept_entries for a in file_audits.values())
    quarantined = sum(a.quarantined_entries for a in file_audits.values())
    repaired = sum(a.repaired_entries for a in file_audits.values())

    lines = [
        "# Database health — rollup",
        "",
        "Feeds into the progress dashboard (prompt 8).",
        "",
        f"- Total sign entries across all databases: **{total}**",
        f"- Kept in active files: **{kept}**",
        f"- Quarantined (moved to sidecar): **{quarantined}**",
        f"- Repaired in place: **{repaired}**",
        "",
        "## Per-file breakdown",
        "",
        "| file | total | kept | quarantined | repaired | unique glosses |",
        "|---|---|---|---|---|---|",
    ]
    stats_entries: dict[str, dict] = {}
    for fname, audit in sorted(file_audits.items()):
        lines.append(
            f"| `{fname}` | {audit.total_entries} | {audit.kept_entries} | "
            f"{audit.quarantined_entries} | {audit.repaired_entries} | "
            f"{len(audit.glosses_active)} |"
        )
        stats_entries[fname] = {
            "language_code": audit.language_code,
            "total": audit.total_entries,
            "kept": audit.kept_entries,
            "quarantined": audit.quarantined_entries,
            "repaired": audit.repaired_entries,
            "duplicate_flagged": audit.duplicate_entries,
            "unique_glosses_active": len(audit.glosses_active),
            "issues": dict(audit.issue_counts),
            "repair_counts": dict(audit.repair_totals),
        }

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Machine-consumable version for prompt 8's dashboard. Kept intentionally
    # small so the dashboard can pull it over the wire on a page load.
    # Aggregate per sign-language code (so a dashboard chip for "BSL" can
    # read a single record that covers both the DictaSign corpus file and
    # the alphabet, rather than joining multiple file rows).
    by_language: dict[str, dict] = {}
    for fname, audit in file_audits.items():
        code = _file_to_lang_code(fname)
        if code is None:
            continue
        bucket = by_language.setdefault(
            code,
            {
                "total": 0,
                "kept": 0,
                "quarantined": 0,
                "repaired": 0,
                "duplicate_flagged": 0,
                "files": [],
            },
        )
        bucket["total"] += audit.total_entries
        bucket["kept"] += audit.kept_entries
        bucket["quarantined"] += audit.quarantined_entries
        bucket["repaired"] += audit.repaired_entries
        bucket["duplicate_flagged"] += audit.duplicate_entries
        bucket["files"].append(fname)
    for code, bucket in by_language.items():
        bucket["alphabet"] = _alphabet_coverage_for_code(code, file_audits)

    stats_payload = {
        "generated_by": "scripts/database_health_audit.py",
        "prompt": 7,
        "totals": {
            "entries": total,
            "kept": kept,
            "quarantined": quarantined,
            "repaired": repaired,
        },
        "files": stats_entries,
        "by_language": by_language,
        "languages_covered": sorted(per_language_bases.keys()),
    }
    stats_path.write_text(
        json.dumps(stats_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _count_existing_quarantine(path: Path) -> int:
    """Count entries already sitting in the sidecar from a previous run.

    When the audit is re-run on a cleaned corpus there are zero new
    entries to quarantine, but the sidecar from the first run still
    holds the historical offenders. Including that count in the report
    keeps the summary stable across repeated runs — otherwise rerunning
    the audit would erase its own history from the dashboard.
    """
    sidecar = path.with_name(f"{path.stem}_quarantine.sigml")
    if not sidecar.exists():
        return 0
    try:
        tree = ET.parse(sidecar)
    except ET.ParseError:
        return 0
    return sum(1 for _ in tree.getroot())


def run(*, write: bool) -> dict[str, FileAudit]:
    known_tags = extract_known_tags()
    file_audits: dict[str, FileAudit] = {}
    per_language_bases: dict[str, set[str]] = {}
    per_language_signs: dict[str, dict[str, tuple[str, ...]]] = {}

    sigml_paths = sorted(p for p in DATA_DIR.glob("*.sigml") if "_quarantine" not in p.name)
    for path in sigml_paths:
        audit, keep_list, quarantine_list = audit_file(path, known_tags)
        file_audits[path.name] = audit

        # Fold in entries already quarantined by a previous run so the
        # summary is stable across repeated invocations.
        historical_quarantine = _count_existing_quarantine(path)
        if historical_quarantine and not quarantine_list:
            audit.quarantined_entries = historical_quarantine
            audit.total_entries = audit.kept_entries + historical_quarantine
            audit.issue_counts["previously_quarantined"] = historical_quarantine

        if write:
            write_cleaned_file(path, keep_list)
            if quarantine_list:
                write_quarantine_file(path, quarantine_list)

        # Language accumulators are keyed by sign-language code, not file
        # name, so BSL (which has an alphabet + main corpus) merges into
        # a single "bsl" entry.
        code = _file_to_lang_code(path.name)
        if code is None:
            continue
        bases = per_language_bases.setdefault(code, set())
        sign_hns = per_language_signs.setdefault(code, {})
        for eaudit, elem in keep_list:
            g = eaudit.normalized_gloss
            if not g:
                continue
            base = _lookup_gloss_base(g)
            if base:
                bases.add(base)
            hns_seq = _extract_hns_tag_sequence(elem)
            if hns_seq and g not in sign_hns:
                sign_hns[g] = hns_seq

    if write:
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        PER_FILE_DIR.mkdir(parents=True, exist_ok=True)
        for audit in file_audits.values():
            write_per_file_report(audit, PER_FILE_DIR)
        write_coverage_matrix(per_language_bases, DOCS_DIR / "07-coverage-matrix.md")
        write_suspected_duplicates(per_language_signs, DOCS_DIR / "07-suspected-duplicates.md")
        write_alphabet_coverage(file_audits, DOCS_DIR / "07-alphabet-coverage.md")
        write_auto_repairs_log(file_audits, DOCS_DIR / "07-auto-repairs.md")
        write_summary_and_stats(
            file_audits,
            per_language_bases,
            DOCS_DIR / "07-summary.md",
            DOCS_DIR / "07-database-stats.json",
        )

    return file_audits


def _file_to_lang_code(filename: str) -> str | None:
    """Map a data file name to a sign-language code for language-level stats."""
    for fname, code, _ in LANG_FILES:
        if filename == fname:
            return code
    # Alphabets roll up into their language code.
    for code, meta in EXPECTED_ALPHABETS.items():
        if filename == meta["file"]:
            return code
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write reports, quarantine sidecars, and in-place repairs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan only — print summary without writing files.",
    )
    args = parser.parse_args()
    do_write = args.write and not args.dry_run

    audits = run(write=do_write)
    total = sum(a.total_entries for a in audits.values())
    kept = sum(a.kept_entries for a in audits.values())
    quarantined = sum(a.quarantined_entries for a in audits.values())
    repaired = sum(a.repaired_entries for a in audits.values())
    mode = "wrote" if do_write else "scanned (dry-run)"
    print(
        f"{mode}: {len(audits)} files, {total} entries — "
        f"{kept} kept, {quarantined} quarantined, {repaired} repaired"
    )


if __name__ == "__main__":
    main()
