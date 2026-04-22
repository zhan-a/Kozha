"""Regression tests for the translation pipeline and SiGML data integrity.

These tests pin the fix shipped in prompt 3 (``fix(translator): resolve '[object Object]'
HamNoSys emission``). They cover three layers:

1. The ``/api/translate-text`` endpoint must always return a JSON body whose
   ``translated`` field is a string — never an object that would later
   serialise as ``[object Object]`` on the client.
2. The ``/api/plan`` endpoint must always return a ``final`` string for a
   small grid of language pairs.
3. The SiGML data files must not contain the literal ``[object Object]``, and
   each sign entry targeted by known regression cases (`fruit` in LSF/BSL/DGS)
   must either parse as well-formed XML or be flagged by the unknown-tag
   scanner.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from xml.etree import ElementTree

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "server") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "server"))

from scripts.scan_unknown_hns_tags import (  # noqa: E402
    CONTAINER_TAGS,
    DATA_DIR,
    extract_known_tags,
    scan_file,
)
from server import app  # noqa: E402


client = TestClient(app)
KNOWN_TAGS = extract_known_tags()


@pytest.mark.parametrize(
    "src,tgt,text",
    [
        ("en", "en", "fruit"),
        ("en", "fr", "fruit"),
        ("en", "de", "fruit"),
        ("en", "pl", "fruit"),
        ("en", "el", "fruit"),
        ("fr", "fr", "fruit"),
    ],
)
def test_translate_text_returns_string(src: str, tgt: str, text: str) -> None:
    """Every language pair must return a JSON string for ``translated``.

    The bug this test pins: if argostranslate ever returns a non-string, our
    server would have shipped that object straight through to the client,
    which would then template-literal it into SiGML as ``[object Object]``.
    """
    resp = client.post(
        "/api/translate-text",
        json={"text": text, "source_lang": src, "target_lang": tgt},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "translated" in data
    assert isinstance(data["translated"], str), f"non-string translated: {data['translated']!r}"
    assert "[object Object]" not in data["translated"]


@pytest.mark.parametrize(
    "text,language,sign_language",
    [
        ("fruit", "en", "bsl"),
        ("fruit", "fr", "lsf"),
        ("fruit", "en", "asl"),
        ("fruit", "en", "dgs"),
        ("zzqqxxyy", "en", "bsl"),
    ],
)
def test_plan_returns_string_final(text: str, language: str, sign_language: str) -> None:
    """/api/plan must always return a ``final`` string field."""
    resp = client.post(
        "/api/plan",
        json={"text": text, "language": language, "sign_language": sign_language},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "final" in data
    assert isinstance(data["final"], str), f"non-string final: {data['final']!r}"
    assert "[object Object]" not in data["final"]


def _extract_sign_block(sigml_path: Path, *gloss_candidates: str) -> str | None:
    text = sigml_path.read_text()
    for gloss in gloss_candidates:
        pattern = re.compile(
            r'<hns_sign\s+gloss="' + re.escape(gloss) + r'">.*?</hns_sign>',
            re.DOTALL,
        )
        m = pattern.search(text)
        if m:
            return m.group(0)
    return None


def _validate_sign_block(block: str) -> tuple[bool, list[str]]:
    """Returns (is_parseable, unknown_tags)."""
    try:
        ElementTree.fromstring(block)
    except ElementTree.ParseError:
        return False, []
    tags = set(re.findall(r"<(ham[A-Za-z][\w]*)", block))
    unknown = sorted(t for t in tags if t not in CONTAINER_TAGS and t not in KNOWN_TAGS)
    return True, unknown


def test_fruit_bsl_exists_and_is_parseable() -> None:
    block = _extract_sign_block(
        DATA_DIR / "hamnosys_bsl_version1.sigml",
        "FRUIT", "fruit", "fruit(n)#1",
    )
    assert block is not None, "FRUIT entry missing from BSL file"
    parseable, unknown = _validate_sign_block(block)
    assert parseable, "FRUIT in BSL is not well-formed XML"
    assert "[object Object]" not in block
    assert not unknown, f"FRUIT in BSL has unknown tags {unknown}"


def test_fruit_lsf_was_quarantined_by_prompt_7() -> None:
    """Closure of the prompt-3 bug: FRUIT in LSF used ``<hampalmud/>``,
    an unknown CWASA tag that the client-side validator rejects. Prompt 7
    quarantined every entry whose HamNoSys could not be safely auto-
    repaired, so FRUIT is now in the sidecar, not the active file.

    Asserting both halves pins the guarantee the health audit makes:
    no unknown tags in the live file, the failing entry preserved in
    quarantine for community review.
    """
    active = _extract_sign_block(DATA_DIR / "French_SL_LSF.sigml", "FRUIT")
    assert active is None, (
        "FRUIT is still in the active LSF file — the health audit should "
        "have moved it to `French_SL_LSF_quarantine.sigml`"
    )
    sidecar = DATA_DIR / "French_SL_LSF_quarantine.sigml"
    assert sidecar.exists(), "LSF quarantine sidecar is missing"
    block = _extract_sign_block(sidecar, "FRUIT")
    assert block is not None, "FRUIT missing from LSF quarantine sidecar"
    parseable, unknown = _validate_sign_block(block)
    assert parseable, "FRUIT in LSF quarantine is not well-formed XML"
    assert "[object Object]" not in block
    assert "hampalmud" in unknown, (
        "FRUIT in LSF quarantine no longer contains hampalmud — either the "
        "sidecar was hand-edited or the quarantine decision changed; review "
        "`docs/polish/07-database-health/French_SL_LSF.md`"
    )


def test_fruit_dgs_is_supported_or_flagged() -> None:
    """DGS should either have FRUIT or not. If it does, it should validate clean."""
    block = _extract_sign_block(
        DATA_DIR / "German_SL_DGS.sigml",
        "FRUIT", "fruit", "FRUIT_A", "FRUIT#1",
    )
    if block is None:
        pytest.skip("FRUIT not present in DGS file")
    parseable, unknown = _validate_sign_block(block)
    assert parseable
    assert "[object Object]" not in block
    # DGS has `hamupperarm` cases; assert FRUIT specifically does not
    assert not unknown, f"FRUIT in DGS has unknown tags {unknown}"


def test_missing_word_returns_empty_or_original() -> None:
    """A word guaranteed to be missing from all databases must not crash.

    The plan endpoint reduces unknown words aggressively; depending on the
    spaCy pipeline it may be kept (as fingerspellable) or dropped. Either is
    acceptable, but the response must still be a JSON string.
    """
    resp = client.post(
        "/api/plan",
        json={"text": "zzqqxxyy", "language": "en", "sign_language": "bsl"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data.get("final"), str)
    assert "[object Object]" not in data["final"]


def test_no_sigml_file_contains_object_literal() -> None:
    """Defence in depth: no .sigml file should literally contain [object Object]."""
    for path in sorted(DATA_DIR.glob("*.sigml")):
        text = path.read_text()
        assert "[object Object]" not in text, (
            f"{path.name} contains literal [object Object] — someone serialised "
            "a JS object into a database entry"
        )


def test_scanner_flags_no_unknown_tags_in_active_lsf() -> None:
    """Scanner sanity post-prompt-7: the active LSF file contains only
    tags CWASA recognises. All ``hampalmud`` occurrences live in the
    quarantine sidecar now — scanning the live file returns nothing.
    """
    unknown = scan_file(DATA_DIR / "French_SL_LSF.sigml", KNOWN_TAGS)
    assert not unknown, (
        f"LSF still has unknown tags in the active file: {dict(unknown)}. "
        f"Rerun `python3 scripts/database_health_audit.py --write`"
    )


def test_scanner_still_sees_hampalmud_in_quarantine() -> None:
    """The broken entries are preserved, not deleted: the quarantine
    sidecar still surfaces ``hampalmud`` so community reviewers can find
    them.
    """
    sidecar = DATA_DIR / "French_SL_LSF_quarantine.sigml"
    if not sidecar.exists():
        pytest.skip("no LSF quarantine sidecar — nothing to review")
    unknown = scan_file(sidecar, KNOWN_TAGS)
    assert "hampalmud" in unknown, (
        "quarantine sidecar lost hampalmud — someone deleted the failing "
        "entries; restore them from git so reviewers can see what went wrong"
    )
