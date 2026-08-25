"""Tests for indexing and hybrid retrieval.

These cover the pure logic -- fusion and tokenisation -- rather than Chroma
or the embedding API. Retrieval quality itself is measured by
``eval/retrieval_eval.py`` against a labelled question set, which is the only
honest way to judge it.
"""

from __future__ import annotations

from app.store import _rrf, _tokenise
from langchain_core.documents import Document


def _doc(text: str, page: int = 1) -> Document:
    return Document(page_content=text, metadata={"doc_id": "d", "page": page})


class TestTokenise:
    """LangChain's BM25Retriever defaults to ``text.split()``. That single
    default kept the Shell goodwill table out of the top 30 for a question
    about Shell's goodwill, because the chunk holds "goodwill," and the
    question holds "goodwill". Fixing it lifted recall@10 from 71% to 93%."""

    def test_trailing_punctuation_is_stripped(self):
        assert "goodwill" in _tokenise("carrying value of goodwill, other assets")

    def test_a_question_and_a_chunk_produce_the_same_term(self):
        assert set(_tokenise("total goodwill?")) & set(_tokenise("Goodwill, total"))

    def test_case_is_normalised(self):
        assert _tokenise("Integrated Gas") == ["integrated", "gas"]

    def test_decimal_figures_stay_whole(self):
        # Matching a figure exactly is the point of the sparse half.
        assert "60.0" in _tokenise("| Integrated Gas | 60.0 |")

    def test_thousands_separators_stay_whole(self):
        assert "5,001.2" in _tokenise("Wages and salaries 5,001.2")

    def test_percentages_are_preserved(self):
        assert "11.3%" in _tokenise("Gearing 11.3%")


class TestReciprocalRankFusion:
    def test_a_document_ranked_by_both_retrievers_wins(self):
        a, b, c = _doc("alpha"), _doc("beta"), _doc("gamma")
        # b is mid-table in each ranking but is the only one in both.
        fused = _rrf([[a, b], [c, b]], k=3)
        assert fused[0].page_content == "beta"

    def test_the_same_chunk_from_two_retrievers_appears_once(self):
        a = _doc("alpha")
        assert len(_rrf([[a], [a]], k=5)) == 1

    def test_parts_of_a_split_table_are_kept_distinct(self):
        """Every part repeats the same description and header row, so keying
        on a prefix would merge them and silently drop all but the first."""
        head = "Segment values.\n| Segment | Value |\n|---|---|\n"
        parts = [_doc(head + "| Upstream | 5.3 |"), _doc(head + "| Marketing | 4.3 |")]
        assert len(_rrf([parts], k=5)) == 2

    def test_k_limits_the_result_size(self):
        docs = [_doc(f"chunk {i}") for i in range(20)]
        assert len(_rrf([docs], k=5)) == 5

    def test_an_empty_ranking_is_harmless(self):
        a = _doc("alpha")
        assert [d.page_content for d in _rrf([[a], []], k=5)] == ["alpha"]
