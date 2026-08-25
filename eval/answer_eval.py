#!/usr/bin/env python
"""Score the generated answers, separately from retrieval.

A RAG failure is either a retrieval failure or a generation failure. This
measures the second, and reports three things:

* **verbatim** -- for questions with a figure verified by reading the source
  page, does that figure appear character-for-character in the answer? The
  brief asks for exact datapoints, and this is the cheapest honest test of it:
  a string check, no LLM judge and none of its biases.
* **cited** -- did the answer come back with at least one page reference?
* **abstained** -- on questions the corpus cannot answer, did the system say
  so instead of producing something plausible? For a bank this is the most
  important column on the page.

    python eval/answer_eval.py
"""

from __future__ import annotations

import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]

warnings.filterwarnings("ignore")

from app.answer import answer_question  # noqa: E402

from eval.validate_questions import load_questions  # noqa: E402


def run(question: dict) -> dict:
    result = answer_question(question["question"])
    return {"q": question, "r": result}


def main() -> int:
    questions = load_questions()
    with ThreadPoolExecutor(max_workers=6) as pool:
        outcomes = list(pool.map(run, questions))

    answerable = [o for o in outcomes if o["q"].get("answerable") is not False]
    unanswerable = [o for o in outcomes if o["q"].get("answerable") is False]

    print(f"{'id':<34}{'found':>7}{'cited':>7}{'verbatim':>10}")
    print("-" * 58)
    verbatim_total = verbatim_hit = cited = 0
    for o in answerable:
        q, r = o["q"], o["r"]
        expected = q.get("expected_answer")
        mark = "-"
        if expected:
            # A figure is often printed in more than one place and unit, so
            # any of the accepted spellings counts.
            accepted = expected if isinstance(expected, list) else [expected]
            verbatim_total += 1
            ok = any(a in r.answer for a in accepted)
            verbatim_hit += ok
            mark = "yes" if ok else "NO"
        cited += bool(r.citations)
        print(f"{q['id']:<34}{str(r.found):>7}{len(r.citations):>7}{mark:>10}")

    print(f"\n{'unanswerable':<34}{'abstained':>12}")
    print("-" * 46)
    abstained = 0
    for o in unanswerable:
        ok = not o["r"].found
        abstained += ok
        print(f"{o['q']['id']:<34}{('yes' if ok else 'NO — ' + o['r'].answer[:40]):>12}")

    print(
        f"\nanswerable={len(answerable)}"
        f" | cited={cited}/{len(answerable)}"
        f" | verbatim={verbatim_hit}/{verbatim_total}"
        f" | abstained={abstained}/{len(unanswerable)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
