#!/usr/bin/env python
"""Check the eval set points at the right pages before anything is scored.

An eval set with a wrong expected page silently reports a working retriever
as broken, and is worse than no eval set at all. Every answerable question
declares a string that must appear on its expected page; this asserts it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]

from app.store import get_store  # noqa: E402

QUESTIONS = ROOT / "eval" / "questions.yaml"


def load_questions() -> list[dict]:
    return yaml.safe_load(QUESTIONS.read_text())


def page_text(company: str, page: int) -> str:
    records = get_store().get(
        where={"$and": [{"company": company}, {"page": page}]}, include=["documents"]
    )
    return "\n".join(records.get("documents") or [])


def main() -> int:
    failures = []
    questions = load_questions()
    for q in questions:
        if q.get("answerable") is False:
            continue
        found = any(q["contains"] in page_text(q["company"], p) for p in q["pages"])
        status = "ok" if found else "MISSING"
        if not found:
            failures.append(q["id"])
        print(f"  {status:>8}  {q['id']:<28} {q['company']} p{q['pages']} <- {q['contains']!r}")

    answerable = sum(1 for q in questions if q.get("answerable") is not False)
    print(f"\n{answerable} answerable, {len(questions) - answerable} unanswerable")
    if failures:
        print(f"FAILED: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("every expected page contains its expected string")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
