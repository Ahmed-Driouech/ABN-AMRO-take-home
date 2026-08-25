"""Measures whether parsed tables reproduce the figures printed in the PDF.

This is evaluation, not runtime. Nothing in ``app/`` imports it and it never
edits a parsed document -- it exists to answer "did the parser actually get
the numbers right?" with a number instead of an assurance.

It earns its place twice. It is the evidence behind reading tables from the
element's ``text`` rather than its ``text_as_html`` (96.79% -> 98.86% of
figures accounted for across the five reports). And because the application
accepts uploads, it is the only check available on a report nobody has seen
before.

    python scripts/parse_reports.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pdfplumber
from langchain_core.documents import Document

NUMERIC = re.compile(r"^\(?-?[\d,]+(?:\.\d+)?\)?%?$")
FOOTNOTE = re.compile(r"\[[A-Za-z0-9]{1,2}\]")


def normalise_figure(value: str) -> str:
    """Strip presentation from a figure so two spellings of the same number
    compare equal: the text layer prints "11.3" and "%" as separate tokens
    where a cell holds "11.3%", and accounting parentheses move around.
    Digits themselves are never touched -- a trailing zero is meaningful."""
    value = FOOTNOTE.sub("", str(value).strip())
    value = value.strip("()[]% \t").replace("$", "").replace("€", "").replace("£", "")
    return value.strip().rstrip(".,").lstrip("+")


def page_figures(path: str | Path) -> dict[int, set[str]]:
    """Every figure printed on each page, read from the PDF's own text layer.

    The text layer is exact glyph data rather than OCR, which is what makes it
    usable as ground truth.
    """
    out: dict[int, set[str]] = {}
    with pdfplumber.open(path) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            out[number] = {
                figure
                for token in (page.extract_text() or "").split()
                if (figure := normalise_figure(token)) and any(c.isdigit() for c in figure)
            }
    return out


def audit_numbers(docs: list[Document], figures: dict[int, set[str]]) -> tuple[int, int]:
    """Count table cells whose figure is not printed on the source page.

    Only whole cells of a markdown row are examined. Running a regex over the
    raw text instead pulls "172" out of "Section 172(1)" and "3" out of "Nm3"
    and reports them as corruption -- which understated the true rate by six
    points when this was first written.
    """
    total = unmatched = 0
    for doc in docs:
        if doc.metadata.get("content_type") != "table":
            continue
        printed = figures.get(doc.metadata.get("page") or 0, set())
        if not printed:
            continue
        for line in doc.page_content.splitlines():
            line = line.strip()
            if not line.startswith("|"):
                continue
            for cell in line.strip("|").split("|"):
                cell = cell.strip().strip("*")
                if not cell or not NUMERIC.match(cell):
                    continue
                total += 1
                if normalise_figure(cell) not in printed:
                    unmatched += 1
    return total, unmatched
