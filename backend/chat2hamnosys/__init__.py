"""chat2hamnosys — LLM-driven natural language to HamNoSys transcription.

See hamnosys/ for the HamNoSys 4.0 grammar, symbol table, and validator.

Modules inside this package use *flat* imports (e.g. ``from models import
SignEntry``), assuming this directory is on ``sys.path``. The main
:mod:`server.server` boots that way explicitly. When importing through
the dotted path (e.g. ``python -m backend.chat2hamnosys.review.admin``)
we bootstrap sys.path here so the flat imports keep working.
"""

import sys as _sys
from pathlib import Path as _Path

_SELF_DIR = str(_Path(__file__).resolve().parent)
if _SELF_DIR not in _sys.path:
    _sys.path.insert(0, _SELF_DIR)
