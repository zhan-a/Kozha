#!/usr/bin/env python3
"""Dump the public-facing site as one copy-pastable plain-text file.

Walks ``public/*.html`` (and ``public/admin/*.html``), strips scripts /
styles / hidden nodes, and joins the visible text from each page into
``site-text.txt`` at the repo root with clear page separators.

Intended for offline evaluation (LLM-as-judge, manual review, copy
into a doc) where pasting the rendered page text is simpler than
running the deploy + a browser. Not committed — see .gitignore.

Run from the repo root: ``python3 scripts/dump_site_text.py``.
"""

from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
PUBLIC_DIR = REPO_ROOT / "public"
OUTPUT_FILE = REPO_ROOT / "site-text.txt"

# HTML elements whose contents shouldn't be rendered as user-visible
# text. ``template`` is included because contribute.html stores
# off-screen markup chunks there.
SKIP_TAGS = frozenset({"script", "style", "noscript", "template", "svg"})

# Block-level tags get a trailing newline to keep paragraphs apart in
# the dump. Headings get an extra blank line + a marker so the
# hierarchy survives plain text.
BLOCK_TAGS = frozenset({
    "p", "div", "section", "article", "header", "footer", "main",
    "ul", "ol", "li", "dl", "dt", "dd", "table", "tr", "form",
    "fieldset", "details", "summary", "blockquote", "figure",
    "figcaption", "hr",
})
HEADING_TAGS = {"h1": "#", "h2": "##", "h3": "###", "h4": "####", "h5": "#####", "h6": "######"}

# Pages to dump in this order. ``index.html`` is the landing page; the
# others follow the visible nav order so a reader walks the site
# top-to-bottom the way a visitor would.
PAGE_ORDER = [
    ("index.html",            "/"),
    ("app.html",              "/app.html"),
    ("contribute.html",       "/contribute.html"),
    ("contribute-status.html", "/contribute-status.html"),
    ("contribute-me.html",    "/contribute-me.html"),
    ("progress.html",         "/progress"),
    ("credits.html",          "/credits"),
    ("governance.html",       "/governance.html"),
    ("whats-new.html",        "/whats-new.html"),
    ("404.html",              "/404.html"),
    ("admin/ops.html",        "/admin/ops.html"),
]


class TextExtractor(HTMLParser):
    """Tiny HTML→plain-text walker.

    Skips ``SKIP_TAGS``, drops nodes carrying ``hidden`` /
    ``aria-hidden="true"`` / ``style="display:none"``, and adds line
    breaks around block-level boundaries so the dump retains some of
    the page's structure.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0          # inside a skipped subtree
        self._hidden_depth = 0        # inside a hidden subtree
        self._heading_marker: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in SKIP_TAGS:
            self._skip_depth += 1
            return
        attr_map = {k: (v or "") for k, v in attrs}
        if (
            "hidden" in attr_map
            or attr_map.get("aria-hidden") == "true"
            or "display:none" in attr_map.get("style", "").replace(" ", "")
        ):
            self._hidden_depth += 1
            return
        if tag in HEADING_TAGS:
            self._heading_marker = HEADING_TAGS[tag]
            self._chunks.append("\n\n" + self._heading_marker + " ")
        elif tag == "br":
            self._chunks.append("\n")
        elif tag == "li":
            self._chunks.append("\n- ")
        elif tag in BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if self._hidden_depth > 0:
            # End-tag bookkeeping is approximate — HTMLParser doesn't
            # tell us which open tag we just left. Decrement once per
            # closing tag while we believe we're inside hidden; this
            # holds for well-formed markup which is what we ship.
            self._hidden_depth -= 1
            return
        if tag in HEADING_TAGS and self._heading_marker:
            self._chunks.append("\n")
            self._heading_marker = None
        elif tag in BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0 or self._hidden_depth > 0:
            return
        text = data.strip("\n\r")
        if not text:
            return
        self._chunks.append(text)

    def get_text(self) -> str:
        joined = "".join(self._chunks)
        # Collapse runs of >2 newlines and runs of trailing whitespace
        # so the output stays readable when pasted somewhere.
        lines = [line.rstrip() for line in joined.splitlines()]
        out: list[str] = []
        blank = 0
        for line in lines:
            if not line.strip():
                blank += 1
                if blank <= 1:
                    out.append("")
                continue
            blank = 0
            out.append(line)
        return "\n".join(out).strip() + "\n"


def extract_page(html_path: Path) -> str:
    parser = TextExtractor()
    parser.feed(html_path.read_text(encoding="utf-8"))
    return parser.get_text()


def collect_pages() -> Iterable[tuple[str, Path]]:
    """Yield ``(url_path, file_path)`` in PAGE_ORDER, then any other
    HTML files found under ``public/`` in alphabetical order so newly
    added pages still get included without being silently dropped."""
    seen: set[Path] = set()
    for rel, url in PAGE_ORDER:
        path = PUBLIC_DIR / rel
        if path.is_file():
            seen.add(path)
            yield url, path
    extra = sorted(set(PUBLIC_DIR.rglob("*.html")) - seen)
    for path in extra:
        rel = path.relative_to(PUBLIC_DIR).as_posix()
        yield "/" + rel, path


def main() -> int:
    if not PUBLIC_DIR.is_dir():
        print(f"public/ not found at {PUBLIC_DIR}", file=sys.stderr)
        return 1
    sections: list[str] = []
    for url, path in collect_pages():
        rel = path.relative_to(REPO_ROOT).as_posix()
        text = extract_page(path)
        header = (
            "=" * 78 + "\n"
            f"PAGE: {url}\n"
            f"FILE: {rel}\n"
            + "=" * 78 + "\n\n"
        )
        sections.append(header + text.rstrip() + "\n")
    body = "\n\n".join(sections)
    preamble = (
        "Kozha — full site text dump\n"
        f"Generated by scripts/dump_site_text.py\n"
        f"Pages: {len(sections)}\n\n"
    )
    OUTPUT_FILE.write_text(preamble + body, encoding="utf-8")
    print(f"Wrote {OUTPUT_FILE.relative_to(REPO_ROOT)} ({OUTPUT_FILE.stat().st_size} bytes, {len(sections)} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
