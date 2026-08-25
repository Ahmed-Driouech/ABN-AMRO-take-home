"""Tests for answer assembly.

These cover the guards, not the model: that a plan can only name a company we
actually hold, and that a citation can only point at a chunk retrieval really
returned. Answer quality itself is measured by ``eval/answer_eval.py``.
"""

from __future__ import annotations

from app.answer import Answer, Citation, _format_context, _resolve_citations
from langchain_core.documents import Document


def _chunk(page: int, company: str = "Shell", **meta) -> Document:
    return Document(
        page_content=f"content for page {page}",
        metadata={
            "company": company,
            "report_year": 2024,
            "page": page,
            "page_end": page,
            "section": "Notes",
            "content_type": "table",
            **meta,
        },
    )


class TestFormatContext:
    def test_excerpts_are_numbered_so_the_model_can_cite_by_number(self):
        text = _format_context([_chunk(10), _chunk(20)])
        assert "[1]" in text and "[2]" in text

    def test_each_excerpt_names_its_company_year_and_page(self):
        text = _format_context([_chunk(258)])
        assert "Shell" in text and "2024" in text and "p. 258" in text

    def test_a_chunk_spanning_pages_shows_a_range(self):
        text = _format_context([_chunk(258, page_end=259)])
        assert "pp. 258-259" in text


class TestResolveCitations:
    """The model cites excerpt numbers; pages are looked up here. If it wrote
    page numbers itself it could invent one that looks entirely plausible."""

    def test_a_cited_number_maps_to_the_real_chunk(self):
        cites = _resolve_citations(
            Answer(found=True, answer="x", sources=[2]), [_chunk(10), _chunk(258)]
        )
        assert cites == [
            Citation(company="Shell", page=258, page_end=258, section="Notes", content_type="table")
        ]

    def test_an_out_of_range_number_is_discarded(self):
        answer = Answer(found=True, answer="x", sources=[9])
        assert _resolve_citations(answer, [_chunk(10)]) == []

    def test_a_repeated_citation_appears_once(self):
        assert (
            len(_resolve_citations(Answer(found=True, answer="x", sources=[1, 1]), [_chunk(10)]))
            == 1
        )

    def test_no_sources_means_no_citations(self):
        assert (
            _resolve_citations(Answer(found=False, answer="not found", sources=[]), [_chunk(10)])
            == []
        )
