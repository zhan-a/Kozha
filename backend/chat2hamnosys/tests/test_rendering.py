"""Tests for :mod:`backend.chat2hamnosys.rendering`.

Four contract areas:

* **Round-trip** — every HamNoSys codepoint we feed into :func:`to_sigml`
  must come out the other side as a self-closing ``<short_name/>`` tag
  with no re-ordering or dropped characters.
* **DTD validation** — both our converter's own output *and* a 20-sign
  sample from the real ``data/hamnosys_bsl_version1.sigml`` corpus must
  satisfy the bundled SiGML DTD. If either stops validating, something
  has drifted between our inventory and the upstream schema.
* **Cache** — :class:`PreviewCache` is content-addressable; a second
  ``render_preview`` call with the same SiGML must hit the sidecar
  instead of re-running the renderer.
* **Graceful degradation** — when no renderer is configured the
  wrapper must return :attr:`PreviewStatus.RENDERER_NOT_AVAILABLE`
  with the documented message, never raise.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from lxml import etree

from hamnosys import SYMBOLS
from models import NonManualFeatures
from rendering.hamnosys_to_sigml import (
    HamNoSysConversionError,
    SigmlValidationError,
    to_sigml,
    validate_sigml,
)
from rendering.preview import (
    PreviewResult,
    PreviewStatus,
    _parse_hamnosys_input,
    render_preview,
)
from rendering.preview_cache import (
    CacheEntry,
    CacheMetadata,
    PreviewCache,
    sigml_content_hash,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
BSL_CORPUS = REPO_ROOT / "data" / "hamnosys_bsl_version1.sigml"

# The ABROAD BSL sign — also used by the main conftest fixture.
ABROAD_HAMNOSYS = (
    "\uE001\uE00C\uE011\uE0E6"
    "\uE001\uE00C\uE020\uE03C"
    "\uE040\uE059\uE089\uE0C6\uE0D8"
)


# ---------------------------------------------------------------------------
# Round-trip: every codepoint must appear as the matching element
# ---------------------------------------------------------------------------


def _manual_tag_names(xml: str) -> list[str]:
    root = etree.fromstring(xml.encode())
    manual = root.find(".//hamnosys_manual")
    assert manual is not None, "output is missing <hamnosys_manual>"
    return [child.tag for child in manual]


def _expected_tag_names(hamnosys: str) -> list[str]:
    return [SYMBOLS[ord(c)].short_name for c in hamnosys]


def test_to_sigml_roundtrip_preserves_order_and_content():
    xml = to_sigml(ABROAD_HAMNOSYS, gloss="ABROAD")
    assert _manual_tag_names(xml) == _expected_tag_names(ABROAD_HAMNOSYS)


def test_to_sigml_every_codepoint_translates_to_its_short_name():
    # Exercise every codepoint in the inventory in a single assembled
    # string — if the converter adds a spurious prefix/suffix, the
    # element list will diverge from the short-name list.
    # Skip bracket/alt punctuation codepoints that would produce
    # structurally impossible strings on their own; we only need a
    # manual block full of EMPTY tags for this test.
    codepoints = sorted(SYMBOLS.keys())
    hns = "".join(chr(cp) for cp in codepoints)
    xml = to_sigml(hns, gloss="ALL_TAGS", validate=False)
    assert _manual_tag_names(xml) == _expected_tag_names(hns)


def test_to_sigml_rejects_unknown_codepoint():
    bad = "\uE001\uF000\uE040"  # U+F000 is outside our table
    with pytest.raises(HamNoSysConversionError) as ei:
        to_sigml(bad, gloss="BAD")
    assert "U+F000" in str(ei.value)


def test_to_sigml_rejects_empty_input():
    with pytest.raises(HamNoSysConversionError):
        to_sigml("", gloss="EMPTY")


def test_to_sigml_rejects_blank_gloss():
    with pytest.raises(ValueError):
        to_sigml(ABROAD_HAMNOSYS, gloss="   ")


def test_to_sigml_escapes_special_chars_in_gloss():
    # lxml escapes attribute values, so ampersands/quotes are safe.
    xml = to_sigml(ABROAD_HAMNOSYS, gloss='A & B "quoted"')
    root = etree.fromstring(xml.encode())
    assert root.find("hns_sign").attrib["gloss"] == 'A & B "quoted"'


# ---------------------------------------------------------------------------
# Non-manual features
# ---------------------------------------------------------------------------


def test_to_sigml_omits_nonmanual_block_when_empty():
    xml = to_sigml(ABROAD_HAMNOSYS, gloss="A", non_manual=None)
    root = etree.fromstring(xml.encode())
    assert root.find(".//hamnosys_nonmanual") is None


def test_to_sigml_omits_nonmanual_block_when_fields_blank():
    nm = NonManualFeatures(mouth_picture="", eye_gaze="looks around")
    xml = to_sigml(ABROAD_HAMNOSYS, gloss="A", non_manual=nm)
    # Only mouth_picture has a canonical encoding; with it blank and
    # the prose fields skipped, the block must be omitted entirely.
    root = etree.fromstring(xml.encode())
    assert root.find(".//hamnosys_nonmanual") is None


def test_to_sigml_emits_mouth_picture_when_set():
    nm = NonManualFeatures(mouth_picture="bi:")
    xml = to_sigml(ABROAD_HAMNOSYS, gloss="B", non_manual=nm)
    root = etree.fromstring(xml.encode())
    nonmanual = root.find(".//hamnosys_nonmanual")
    assert nonmanual is not None
    mouth = nonmanual.find("hnm_mouthpicture")
    assert mouth is not None
    assert mouth.attrib["picture"] == "bi:"
    # nonmanual must precede manual per the DTD content model
    kids = list(root.find("hns_sign"))
    assert kids[0].tag == "hamnosys_nonmanual"
    assert kids[1].tag == "hamnosys_manual"


# ---------------------------------------------------------------------------
# DTD validation
# ---------------------------------------------------------------------------


def test_validate_sigml_accepts_to_sigml_output():
    xml = to_sigml(ABROAD_HAMNOSYS, gloss="ABROAD")
    ok, errors = validate_sigml(xml)
    assert ok, f"our own output failed DTD: {errors}"


def test_validate_sigml_rejects_unknown_element():
    xml = to_sigml(ABROAD_HAMNOSYS, gloss="X")
    bad = xml.replace("<hamflathand/>", "<hamflathand/><hambogustag/>", 1)
    ok, errors = validate_sigml(bad)
    assert not ok
    joined = "\n".join(errors)
    assert "hambogustag" in joined


def test_to_sigml_raises_if_dtd_validation_fails(monkeypatch):
    # Swap in a stricter DTD that disallows any <hns_sign> children, so
    # our normal output fails validation; this proves validate=True
    # raises rather than returning silently.
    import io as _io

    from rendering import hamnosys_to_sigml as mod

    tiny = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<!ELEMENT sigml EMPTY>\n"
    )
    monkeypatch.setattr(mod, "_DTD_CACHE", None)
    monkeypatch.setattr(
        mod, "_get_dtd", lambda: etree.DTD(_io.BytesIO(tiny.encode()))
    )
    with pytest.raises(SigmlValidationError):
        to_sigml(ABROAD_HAMNOSYS, gloss="ABROAD")


@pytest.mark.skipif(
    not BSL_CORPUS.exists(),
    reason="BSL corpus not checked out",
)
def test_twenty_real_bsl_signs_validate():
    """Sanity-check our DTD against the upstream DictaSign BSL corpus.

    Pull the first 20 signs whose manual block only uses tags in our
    inventory; each one, rewrapped as a standalone ``<sigml>`` doc,
    must satisfy the DTD. If this fails it almost always means we
    removed a tag from the inventory that the corpus still uses.
    """
    known = {s.short_name for s in SYMBOLS.values()}
    tree = etree.parse(str(BSL_CORPUS))
    signs = tree.getroot().findall("hns_sign")

    picked: list[etree._Element] = []
    for s in signs:
        manual = s.find("hamnosys_manual")
        if manual is None:
            continue
        if all(c.tag in known for c in manual):
            picked.append(s)
        if len(picked) >= 20:
            break
    assert len(picked) == 20, "corpus should contain >=20 signs with known tags"

    for s in picked:
        wrapper = etree.Element("sigml")
        # Deep-copy the sign into the new root
        wrapper.append(etree.fromstring(etree.tostring(s)))
        xml = etree.tostring(wrapper, xml_declaration=True, encoding="UTF-8")
        ok, errors = validate_sigml(xml.decode())
        assert ok, (
            f"real BSL sign {s.attrib.get('gloss')!r} failed DTD validation: "
            f"{errors[:1]}"
        )


# ---------------------------------------------------------------------------
# Preview cache — content-hash lookups, sidecar correctness
# ---------------------------------------------------------------------------


def _write_dummy_video(path: Path, body: bytes = b"\x00\x00\x00 ftypmp42") -> Path:
    path.write_bytes(body)
    return path


def test_cache_miss_then_hit(tmp_path: Path):
    cache = PreviewCache(tmp_path / "cache")
    sigml = to_sigml(ABROAD_HAMNOSYS, gloss="ABROAD")

    assert cache.get(sigml) is None
    src = _write_dummy_video(tmp_path / "render.mp4")
    entry = cache.put(sigml, src, gloss="ABROAD")
    assert entry.video_path.exists()
    assert entry.sidecar_path.exists()

    hit = cache.get(sigml)
    assert hit is not None
    assert hit.video_path == entry.video_path
    assert hit.metadata.sigml_hash == sigml_content_hash(sigml)
    assert hit.metadata.gloss == "ABROAD"


def test_cache_keyed_by_content_not_gloss(tmp_path: Path):
    """Identical SiGML → single cache slot, regardless of metadata."""
    cache = PreviewCache(tmp_path)
    sigml = to_sigml(ABROAD_HAMNOSYS, gloss="ABROAD")
    src = _write_dummy_video(tmp_path / "v.mp4")

    entry1 = cache.put(sigml, src, gloss="ONE")
    entry2 = cache.put(sigml, src, gloss="TWO")
    assert entry1.video_path == entry2.video_path
    # sidecar should have been overwritten with the latest gloss
    assert cache.get(sigml).metadata.gloss == "TWO"


def test_cache_treats_partial_entries_as_miss(tmp_path: Path):
    cache = PreviewCache(tmp_path)
    sigml = to_sigml(ABROAD_HAMNOSYS, gloss="ABROAD")
    h = sigml_content_hash(sigml)
    # Create just the video, no sidecar.
    _write_dummy_video(tmp_path / f"{h}.mp4")
    assert cache.get(sigml) is None
    # And just the sidecar:
    (tmp_path / f"{h}.mp4").unlink()
    (tmp_path / f"{h}.json").write_text(
        json.dumps({"sigml_hash": h, "gloss": "X"}),
        encoding="utf-8",
    )
    assert cache.get(sigml) is None


def test_cache_invalidate_removes_both_files(tmp_path: Path):
    cache = PreviewCache(tmp_path)
    sigml = to_sigml(ABROAD_HAMNOSYS, gloss="ABROAD")
    cache.put(sigml, _write_dummy_video(tmp_path / "r.mp4"))
    assert cache.invalidate(sigml) is True
    assert cache.get(sigml) is None
    assert cache.invalidate(sigml) is False


def test_cache_integrity_flags_dangling_sidecar(tmp_path: Path):
    cache = PreviewCache(tmp_path)
    (tmp_path / "dead.json").write_text(
        json.dumps({"sigml_hash": "dead", "gloss": "X"}),
        encoding="utf-8",
    )
    complaints = cache.integrity()
    assert any("sidecar without video" in c for c in complaints)


# ---------------------------------------------------------------------------
# render_preview — graceful degradation + cache hit behaviour
# ---------------------------------------------------------------------------


def test_render_preview_degrades_cleanly_when_no_renderer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Ensure no env-var renderer or on-PATH renderer is picked up.
    monkeypatch.delenv("KOZHA_RENDERER_CMD", raising=False)
    monkeypatch.setattr(
        "rendering.preview.shutil.which",
        lambda _name: None,
    )

    cache = PreviewCache(tmp_path)
    sigml = to_sigml(ABROAD_HAMNOSYS, gloss="ABROAD")

    result = render_preview(sigml, gloss="ABROAD", cache=cache)
    assert result.status == PreviewStatus.RENDERER_NOT_AVAILABLE
    assert not result.ok
    assert result.video_path is None
    assert "SiGML produced successfully" in result.message
    assert result.sigml == sigml


def test_render_preview_hits_cache_on_duplicate_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache = PreviewCache(tmp_path)
    sigml = to_sigml(ABROAD_HAMNOSYS, gloss="ABROAD")

    # Fake renderer command: a `cp` that uses both placeholders so the
    # template helper doesn't append a trailing sigml_path. Copying the
    # sigml into the output slot produces a non-empty file at {output},
    # which is all the cache layer actually cares about.
    cmd = ["cp", "{sigml}", "{output}"]

    first = render_preview(sigml, gloss="ABROAD", cache=cache, renderer_cmd=cmd)
    assert first.status == PreviewStatus.OK, first.message
    assert first.video_path is not None and first.video_path.exists()

    # Track subprocess calls on the second invocation — it must be zero.
    calls = {"n": 0}

    def _should_not_run(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("subprocess invoked on a cache hit")

    monkeypatch.setattr(subprocess, "run", _should_not_run)
    second = render_preview(sigml, gloss="ABROAD", cache=cache, renderer_cmd=cmd)
    assert second.status == PreviewStatus.CACHED
    assert second.video_path == first.video_path
    assert calls["n"] == 0


def test_render_preview_reports_failure_if_renderer_exits_nonzero(tmp_path: Path):
    cache = PreviewCache(tmp_path)
    sigml = to_sigml(ABROAD_HAMNOSYS, gloss="ABROAD")
    # `false` always exits non-zero and produces no file.
    result = render_preview(
        sigml, gloss="X", cache=cache, renderer_cmd=["false"], timeout=5.0
    )
    assert result.status == PreviewStatus.RENDERER_FAILED
    assert result.video_path is None
    assert result.sigml == sigml


def test_render_preview_reports_failure_if_renderer_missing(tmp_path: Path):
    cache = PreviewCache(tmp_path)
    sigml = to_sigml(ABROAD_HAMNOSYS, gloss="ABROAD")
    result = render_preview(
        sigml,
        gloss="X",
        cache=cache,
        renderer_cmd=["/definitely/not/a/real/binary/xyzzy"],
        timeout=5.0,
    )
    assert result.status == PreviewStatus.RENDERER_FAILED
    assert "FileNotFoundError" in (
        result.renderer_stderr or ""
    ) or "renderer invocation failed" in (result.renderer_stderr or "")


# ---------------------------------------------------------------------------
# CLI input parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("E001 E020 E03C E040", "\uE001\uE020\uE03C\uE040"),
        ("U+E001 U+E020", "\uE001\uE020"),
        ("\uE001\uE020\uE03C", "\uE001\uE020\uE03C"),  # raw PUA passes through
    ],
)
def test_parse_hamnosys_input_accepts_multiple_forms(raw: str, expected: str):
    assert _parse_hamnosys_input(raw) == expected


def test_cli_prints_sigml_for_valid_input(tmp_path: Path):
    # End-to-end CLI: invoke the module as a subprocess.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "rendering.preview",
            "E001 E020 E03C E040",
            "--gloss",
            "ELECTRON",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=10.0,
    )
    assert proc.returncode == 0, proc.stderr
    assert "<hns_sign gloss=\"ELECTRON\">" in proc.stdout
    assert "<hamflathand/>" in proc.stdout
    assert "<hamextfingeru/>" in proc.stdout
