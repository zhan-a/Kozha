"""Rendering — HamNoSys → SiGML → avatar video.

Thin layer that takes the Unicode-PUA HamNoSys strings produced by
:mod:`backend.chat2hamnosys.generator` and packages them as the SiGML
XML documents the Kozha frontend (``public/cwa/allcsa.js``) hands to
CWASA/JASigning for avatar playback. Validation is anchored to the
bundled :file:`sigml.dtd` so the generator can't ship a document the
renderer would silently mis-interpret.

Import public symbols directly from their submodules::

    from rendering.hamnosys_to_sigml import to_sigml
    from rendering.preview import render_preview, PreviewResult
    from rendering.preview_cache import PreviewCache

The submodule-level imports are deliberate: they keep ``python -m
rendering.preview`` (the CLI in :mod:`.preview`) free of the
double-initialisation warning that ``from .preview import …`` in this
file would otherwise trigger.
"""
