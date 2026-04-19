"""Module entry point so ``python -m eval`` (or
``python -m backend.chat2hamnosys.eval``) invokes the CLI.
"""

from __future__ import annotations

import sys

from .cli import main


if __name__ == "__main__":  # pragma: no cover — passthrough
    raise SystemExit(main(sys.argv[1:]))
