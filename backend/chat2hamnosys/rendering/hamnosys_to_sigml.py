"""HamNoSys PUA string → SiGML XML document.

Takes the Unicode Private-Use-Area HamNoSys strings produced by
:mod:`backend.chat2hamnosys.generator` (see prompt 7) and wraps them in
the ``<sigml><hns_sign><hamnosys_manual>…</hamnosys_manual></hns_sign></sigml>``
shape that the Kozha CWASA renderer expects. Each PUA codepoint is
resolved against :data:`hamnosys.SYMBOLS` and emitted as a self-closing
element named after the symbol's canonical ``short_name``
(``U+E001 → <hamflathand/>``, etc.).

Validation
----------
Every document produced by :func:`to_sigml` is validated against the
bundled :file:`sigml.dtd` (modeled on the UEA Virtual Humans SiGML
schema). Validation failure raises :class:`SigmlValidationError` so the
generator can never ship XML that would crash or silently mis-play in
the renderer. The DTD is loaded once and cached.

Non-manual features
-------------------
:class:`backend.chat2hamnosys.models.NonManualFeatures` has one field
that has a canonical SiGML representation today (``mouth_picture`` →
``<hnm_mouthpicture picture="…"/>``). The other fields store prose
descriptions for the authoring dialogue — they have no tag encoding
yet, so we skip them rather than round-trip them as comments that the
renderer would ignore. If every non-manual field is empty we omit the
``<hamnosys_nonmanual>`` block entirely; CWASA treats that as "no
non-manual overlay".
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from lxml import etree

from hamnosys import SYMBOLS

if TYPE_CHECKING:  # pragma: no cover - typing only
    from models import NonManualFeatures


DTD_PATH: Path = Path(__file__).with_name("sigml.dtd")


class HamNoSysConversionError(ValueError):
    """The HamNoSys input contains a codepoint we cannot map to a SiGML tag."""


class SigmlValidationError(ValueError):
    """A generated SiGML document failed validation against the bundled DTD."""

    def __init__(self, message: str, errors: list[str]) -> None:
        super().__init__(message)
        self.errors = errors


_DTD_CACHE: etree.DTD | None = None


def _get_dtd() -> etree.DTD:
    """Load the bundled SiGML DTD, caching the parsed copy."""
    global _DTD_CACHE
    if _DTD_CACHE is None:
        with DTD_PATH.open("rb") as f:
            _DTD_CACHE = etree.DTD(f)
    return _DTD_CACHE


def _manual_children(hamnosys: str) -> list[etree._Element]:
    """Map each HamNoSys codepoint to a self-closing XML element."""
    if not hamnosys:
        raise HamNoSysConversionError(
            "hamnosys must be a non-empty string of HamNoSys codepoints"
        )
    elements: list[etree._Element] = []
    for i, ch in enumerate(hamnosys):
        sym = SYMBOLS.get(ord(ch))
        if sym is None:
            raise HamNoSysConversionError(
                f"position {i}: U+{ord(ch):04X} is not a known HamNoSys 4.0 "
                f"codepoint; cannot emit a SiGML tag"
            )
        elements.append(etree.Element(sym.short_name))
    return elements


def _nonmanual_children(
    non_manual: "NonManualFeatures | None",
) -> list[etree._Element]:
    """Translate ``NonManualFeatures`` into SiGML ``<hnm_*>`` elements.

    Only ``mouth_picture`` has a canonical SiGML encoding today; the
    other fields are prose notes consumed by the authoring dialogue and
    have no renderer-visible tag. Returning an empty list signals that
    the ``<hamnosys_nonmanual>`` block should be omitted entirely.
    """
    if non_manual is None:
        return []
    children: list[etree._Element] = []
    mouth = (non_manual.mouth_picture or "").strip() if non_manual.mouth_picture else ""
    if mouth:
        el = etree.Element("hnm_mouthpicture")
        el.set("picture", mouth)
        children.append(el)
    return children


def to_sigml(
    hamnosys: str,
    gloss: str = "SIGN",
    *,
    non_manual: "NonManualFeatures | None" = None,
    validate: bool = True,
    xml_declaration: bool = True,
) -> str:
    """Convert a HamNoSys PUA string to a CWASA-ready SiGML document.

    Parameters
    ----------
    hamnosys:
        Unicode string of HamNoSys 4.0 PUA codepoints (typically
        ``U+E000..U+E0F1`` plus ASCII brackets ``{``, ``|``, ``}``).
        Each codepoint must appear in :data:`hamnosys.SYMBOLS`.
    gloss:
        The gloss attribute written onto ``<hns_sign>``. Defaults to
        ``"SIGN"`` to match the prompt contract. The gloss is inserted
        as an XML attribute value and will be escaped if it contains
        markup characters.
    non_manual:
        Optional :class:`NonManualFeatures`. If provided and at least
        one field has a canonical SiGML encoding (currently only
        ``mouth_picture``), a ``<hamnosys_nonmanual>`` block is emitted
        before the manual block. If no usable field is set the block
        is omitted — CWASA treats that as "no non-manual overlay".
    validate:
        When ``True`` (default), the assembled document is validated
        against the bundled SiGML DTD and :class:`SigmlValidationError`
        is raised on any violation. Disable only for ad-hoc diagnostics.
    xml_declaration:
        Whether to prepend ``<?xml version="1.0" encoding="UTF-8"?>``.
        The corpus files under ``data/*.sigml`` all have one; keep this
        on unless you're embedding the fragment in a larger document.

    Returns
    -------
    str
        A serialised, pretty-printed SiGML XML document.

    Raises
    ------
    HamNoSysConversionError
        If ``hamnosys`` is empty or contains a codepoint that is not a
        known HamNoSys 4.0 symbol.
    SigmlValidationError
        If ``validate=True`` and the output does not satisfy the DTD.
    """
    if not gloss or not gloss.strip():
        raise ValueError("gloss must be a non-empty string")

    root = etree.Element("sigml")
    sign = etree.SubElement(root, "hns_sign", gloss=gloss)

    nonmanual_kids = _nonmanual_children(non_manual)
    if nonmanual_kids:
        nm = etree.SubElement(sign, "hamnosys_nonmanual")
        for kid in nonmanual_kids:
            nm.append(kid)

    manual = etree.SubElement(sign, "hamnosys_manual")
    for kid in _manual_children(hamnosys):
        manual.append(kid)

    if validate:
        dtd = _get_dtd()
        if not dtd.validate(root):
            errors = [str(e) for e in dtd.error_log]
            raise SigmlValidationError(
                f"generated SiGML failed DTD validation ({len(errors)} error(s)): "
                f"{errors[0] if errors else '<no details>'}",
                errors=errors,
            )

    xml_bytes = etree.tostring(
        root,
        pretty_print=True,
        xml_declaration=xml_declaration,
        encoding="UTF-8",
    )
    return xml_bytes.decode("utf-8")


def validate_sigml(sigml: str) -> tuple[bool, list[str]]:
    """Validate an existing SiGML document against the bundled DTD.

    Returns ``(ok, errors)``. Useful for round-tripping or for tests
    that exercise arbitrary XML from the data/ corpus rather than our
    own :func:`to_sigml` output.
    """
    try:
        doc = etree.fromstring(sigml.encode("utf-8"))
    except etree.XMLSyntaxError as exc:
        return False, [f"XML syntax error: {exc}"]
    dtd = _get_dtd()
    if dtd.validate(doc):
        return True, []
    return False, [str(e) for e in dtd.error_log]


__all__ = [
    "DTD_PATH",
    "HamNoSysConversionError",
    "SigmlValidationError",
    "to_sigml",
    "validate_sigml",
]
