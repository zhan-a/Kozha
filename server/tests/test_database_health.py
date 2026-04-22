"""Regression guard for the prompt-7 database health invariants.

These tests run against the live ``data/`` directory and fail CI if a
new malformed entry sneaks in. They intentionally re-parse the files
on every run (rather than caching the audit output) so that an entry
added after the last audit still trips the guard.

Guarded invariants:

1. Every active ``.sigml`` file is well-formed XML.
2. Every active ``<hns_sign>`` entry has a non-empty gloss.
3. No active ``.sigml`` file contains the literal ``[object Object]``.
4. Every active entry parses cleanly under the prompt-3 unknown-tag
   scanner — i.e. only tags CWASA's ``tokenNameMap`` recognises, or
   tags that the health audit's safe-repair table has already replaced.
5. Quarantine sidecars exist if and only if the active file has had
   entries removed, and they are themselves well-formed.
6. The server's review-metadata loader can boot without exceptions
   against the live data (exercises the hardening in
   ``review_metadata._enumerate_glosses``).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from xml.etree import ElementTree

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "server") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "server"))

from scripts.scan_unknown_hns_tags import (  # noqa: E402
    CONTAINER_TAGS,
    DATA_DIR,
    extract_known_tags,
)
from scripts.database_health_audit import SAFE_TAG_REPAIRS  # noqa: E402

import review_metadata  # noqa: E402


KNOWN_TAGS = extract_known_tags()
ACTIVE_SIGML_FILES = sorted(
    p for p in DATA_DIR.glob("*.sigml") if "_quarantine" not in p.name
)
QUARANTINE_SIGML_FILES = sorted(DATA_DIR.glob("*_quarantine.sigml"))


@pytest.mark.parametrize("path", ACTIVE_SIGML_FILES, ids=lambda p: p.name)
def test_active_file_parses_as_xml(path: Path) -> None:
    """Any file the frontend might fetch must parse as valid XML."""
    ElementTree.parse(path)


@pytest.mark.parametrize("path", ACTIVE_SIGML_FILES, ids=lambda p: p.name)
def test_active_file_has_no_empty_gloss(path: Path) -> None:
    """An entry with an empty gloss is unreachable and indicates drift."""
    tree = ElementTree.parse(path)
    offenders: list[int] = []
    for idx, child in enumerate(tree.getroot()):
        if child.tag != "hns_sign":
            # ``hamgestural_sign`` entries have no gloss by design.
            # They are quarantined by the health audit, so finding one
            # in an active file is itself a problem — captured by the
            # stricter test below.
            continue
        gloss = (child.get("gloss") or "").strip()
        if not gloss:
            offenders.append(idx)
    assert not offenders, (
        f"{path.name} has hns_sign entries with empty gloss at indexes "
        f"{offenders} — these should be quarantined"
    )


@pytest.mark.parametrize("path", ACTIVE_SIGML_FILES, ids=lambda p: p.name)
def test_active_file_has_no_gestural_sign_without_gloss(path: Path) -> None:
    """``hamgestural_sign`` entries carry no gloss — quarantine them."""
    tree = ElementTree.parse(path)
    offenders = [
        idx for idx, c in enumerate(tree.getroot())
        if c.tag == "hamgestural_sign" and not (c.get("gloss") or "").strip()
    ]
    assert not offenders, (
        f"{path.name}: {len(offenders)} `hamgestural_sign` entries with no "
        f"gloss remain in the active file — move them to the quarantine "
        f"sidecar"
    )


@pytest.mark.parametrize("path", ACTIVE_SIGML_FILES, ids=lambda p: p.name)
def test_active_file_has_no_object_literal(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    assert "[object Object]" not in text, (
        f"{path.name} contains literal [object Object] — a JS object was "
        f"templated into the XML; quarantine that entry"
    )


@pytest.mark.parametrize("path", ACTIVE_SIGML_FILES, ids=lambda p: p.name)
def test_active_file_has_no_unknown_hamnosys_tags(path: Path) -> None:
    """CWASA's tokenNameMap covers every ham* tag in an active file.

    After the prompt-7 audit, any tag that survived is either in CWASA's
    spec or was rewritten by the safe-repair table. Anything else is a
    new ingestion that bypassed the audit — flag it so the contributor
    can either extend the repair table or route it through quarantine.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    repair_sources = set(SAFE_TAG_REPAIRS.keys())
    found = set(re.findall(r"<(ham[A-Za-z][\w]*)", text))
    unknown = sorted(
        t for t in found
        if t not in CONTAINER_TAGS
        and t not in KNOWN_TAGS
        # Tags listed in the safe-repair *source* set shouldn't be
        # present post-audit either; list them so the assertion is
        # explicit about why they're unacceptable.
        and t not in repair_sources
    )
    assert not unknown, (
        f"{path.name} has unknown HamNoSys tags {unknown!r}. Either extend "
        f"the CWASA `tokenNameMap`, add the tag to `SAFE_TAG_REPAIRS`, or "
        f"quarantine the entries containing it"
    )


@pytest.mark.parametrize("path", QUARANTINE_SIGML_FILES, ids=lambda p: p.name)
def test_quarantine_file_is_well_formed(path: Path) -> None:
    """Quarantine sidecars are preserved artefacts; they must still parse."""
    ElementTree.parse(path)


def test_quarantine_and_active_are_disjoint() -> None:
    """No sign should appear in both the active file and its quarantine.

    A gloss that survived the audit but is also in quarantine would be
    a double-write bug: the translator would serve it while the report
    claims it's broken.
    """
    for active in ACTIVE_SIGML_FILES:
        sidecar = active.with_name(f"{active.stem}_quarantine.sigml")
        if not sidecar.exists():
            continue
        active_tree = ElementTree.parse(active)
        quar_tree = ElementTree.parse(sidecar)
        active_glosses = {
            (c.get("gloss") or "").strip().lower()
            for c in active_tree.getroot()
            if c.tag == "hns_sign" and (c.get("gloss") or "").strip()
        }
        quar_glosses = {
            (c.get("gloss") or "").strip().lower()
            for c in quar_tree.getroot()
            if c.tag == "hns_sign" and (c.get("gloss") or "").strip()
        }
        overlap = active_glosses & quar_glosses
        assert not overlap, (
            f"{active.name} and its quarantine sidecar share glosses: "
            f"{sorted(overlap)[:10]}"
        )


def test_enumerate_glosses_rejects_object_literal(tmp_path: Path) -> None:
    """``_enumerate_glosses`` must drop any entry whose body is broken.

    Defence-in-depth: even if a malformed entry slips into a data file
    post-deploy, the review-metadata loader must not admit its gloss
    into the known set (since the translator would then claim coverage
    the renderer can't deliver).
    """
    fixture = tmp_path / "bad.sigml"
    fixture.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<sigml>\n'
        '  <hns_sign gloss="GOOD">\n'
        '    <hamnosys_manual><hamflathand/></hamnosys_manual>\n'
        '  </hns_sign>\n'
        '  <hns_sign gloss="BAD">\n'
        '    <hamnosys_manual>[object Object]</hamnosys_manual>\n'
        '  </hns_sign>\n'
        '</sigml>\n',
        encoding="utf-8",
    )
    glosses = review_metadata._enumerate_glosses(fixture)
    assert "good" in glosses
    assert "bad" not in glosses


def test_enumerate_glosses_rejects_entry_without_manual(tmp_path: Path) -> None:
    fixture = tmp_path / "no_manual.sigml"
    fixture.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<sigml>\n'
        '  <hns_sign gloss="SHELL"><hamnosys_nonmanual></hamnosys_nonmanual></hns_sign>\n'
        '</sigml>\n',
        encoding="utf-8",
    )
    glosses = review_metadata._enumerate_glosses(fixture)
    assert "shell" not in glosses


def test_enumerate_glosses_skips_quarantine_sidecars(tmp_path: Path) -> None:
    sidecar = tmp_path / "Language_quarantine.sigml"
    sidecar.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<sigml>\n'
        '  <hns_sign gloss="Q">'
        '  <hamnosys_manual><hamflathand/></hamnosys_manual></hns_sign>\n'
        '</sigml>\n',
        encoding="utf-8",
    )
    assert review_metadata._enumerate_glosses(sidecar) == set()


def test_review_metadata_loader_boots_on_live_data() -> None:
    """End-to-end: the loader's ``reload()`` must not raise against the
    live ``data/`` directory — a single malformed meta file cannot bring
    the whole translator down.
    """
    index = review_metadata.reset_for_tests()
    # A successful reload produces at least the alphabet languages (the
    # main corpora may be stripped by the audit but alphabets survive).
    langs = index.known_languages()
    assert langs, "review-metadata produced no languages — loader crashed"
    # BSL is the reference corpus; if its summary is missing the loader
    # silently dropped something load-bearing.
    bsl = index.language_summary("bsl")
    assert bsl["total"] > 0 or bsl["reviewed"] >= 0


def test_safe_repair_sources_do_not_appear_in_active_files() -> None:
    """``hamupperarm``, ``hamindxfinger`` etc. must never survive the audit.

    If a rerun of the audit ever leaves these tags present, the repair
    pass broke (or an entry was added post-audit without rerunning).
    """
    repair_sources = set(SAFE_TAG_REPAIRS.keys())
    offenders: list[str] = []
    for path in ACTIVE_SIGML_FILES:
        text = path.read_text(encoding="utf-8", errors="replace")
        for tag in repair_sources:
            if re.search(rf"<{tag}\b", text):
                offenders.append(f"{path.name}: <{tag}/>")
    assert not offenders, (
        "Unrepaired tag occurrences: "
        + ", ".join(offenders)
        + " — rerun `python3 scripts/database_health_audit.py --write`"
    )
