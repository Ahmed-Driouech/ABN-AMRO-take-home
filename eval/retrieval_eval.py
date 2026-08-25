#!/usr/bin/env python
"""Score retrieval against the eval set, so its knobs are set from data.

A RAG failure is either a retrieval failure or a generation failure, and the
two have different fixes. This measures the first in isolation: did the page
holding the answer come back at all? No LLM is involved, so a regression here
is unambiguous.

**recall@k** is the headline metric -- the share of questions whose answer
page appears in the top k. Precision matters much less here: passing a few
extra chunks to generation costs tokens, whereas missing the right chunk makes
a correct answer impossible.

Results are also split by question type. A set made only of "find this exact
figure" questions is inherently BM25-friendly and would argue for dropping
dense retrieval on biased evidence, so the paraphrased questions -- whose
wording does not appear on the answer page -- are scored separately.

    python eval/retrieval_eval.py
"""

from __future__ import annotations

import hashlib
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]

warnings.filterwarnings("ignore")

from app.store import CANDIDATES, _bm25, get_store  # noqa: E402

from eval.validate_questions import load_questions  # noqa: E402

KS = (1, 3, 5, 10)

# (label, dense weight, sparse weight, RRF damping)
CONFIGS = [
    ("dense only", 1.0, 0.0, 60),
    ("sparse only", 0.0, 1.0, 60),
    ("hybrid 50/50 k=60", 1.0, 1.0, 60),
    ("hybrid 50/50 k=10", 1.0, 1.0, 10),
    ("hybrid 30/70 k=10", 0.3, 0.7, 10),
    ("hybrid 20/80 k=10", 0.2, 0.8, 10),
]


def fuse(rankings: list[tuple[list, float]], rrf_k: int) -> list:
    """Weighted Reciprocal Rank Fusion.

    Fusing on rank rather than score means the dense and sparse scores, which
    live on unrelated scales, never have to be made comparable.
    """
    scores: dict[str, float] = {}
    best: dict[str, object] = {}
    for ranking, weight in rankings:
        for position, doc in enumerate(ranking):
            key = hashlib.sha1(doc.page_content.encode()).hexdigest()
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + position + 1)
            best.setdefault(key, doc)
    return [best[key] for key in sorted(scores, key=lambda key: scores[key], reverse=True)]


def rank_pages(question: dict, dense_w: float, sparse_w: float, rrf_k: int) -> list[int]:
    """The pages retrieval returns for one question, best first."""
    where = {"company": question["company"]} if question.get("company") else None
    dense = (
        get_store().similarity_search(question["question"], k=CANDIDATES, filter=where)
        if dense_w
        else []
    )
    retriever = _bm25(question.get("company")) if sparse_w else None
    sparse = retriever.invoke(question["question"]) if retriever else []
    return [d.metadata.get("page") for d in fuse([(dense, dense_w), (sparse, sparse_w)], rrf_k)]


def recall(questions: list[dict], config: tuple) -> dict[int, float]:
    _, dense_w, sparse_w, rrf_k = config
    hits = dict.fromkeys(KS, 0)
    for q in questions:
        pages = rank_pages(q, dense_w, sparse_w, rrf_k)
        for k in KS:
            if set(pages[:k]) & set(q["pages"]):
                hits[k] += 1
    return {k: hits[k] / len(questions) for k in KS}


def report(label: str, questions: list[dict]) -> None:
    header = f"{label:<22}" + "".join(f"{'R@' + str(k):>8}" for k in KS)
    print(f"\n{header}\n" + "-" * len(header))
    for config in CONFIGS:
        scores = recall(questions, config)
        print(f"{config[0]:<22}" + "".join(f"{scores[k]:>7.0%} " for k in KS), flush=True)


def main() -> int:
    questions = [q for q in load_questions() if q.get("answerable") is not False]
    lexical = [q for q in questions if not q["id"].startswith("paraphrase")]
    paraphrased = [q for q in questions if q["id"].startswith("paraphrase")]

    report(f"all ({len(questions)})", questions)
    report(f"lexical ({len(lexical)})", lexical)
    report(f"paraphrased ({len(paraphrased)})", paraphrased)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
