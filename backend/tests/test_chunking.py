"""Tests for chunking.

Parsing emits one Document per layout element, and those are far too fine to
retrieve -- the median prose element is 7 tokens. So the job here is merging
first and splitting second, and the tests exist to keep that from silently
regressing into "run a splitter over things already smaller than chunk_size",
which is a no-op.
"""

from __future__ import annotations

from app.chunking import chunk_documents
from app.config import settings
from langchain_core.documents import Document


def _prose(text: str, page: int = 1, section: str = "Strategic Report > Outlook"):
    return Document(
        page_content=text,
        metadata={
            "company": "Shell",
            "report_year": 2024,
            "doc_id": "abc",
            "page": page,
            "section": section,
            "content_type": "prose",
        },
    )


def _table(text: str, page: int = 25):
    return Document(
        page_content=text,
        metadata={
            "company": "Shell",
            "report_year": 2024,
            "doc_id": "abc",
            "page": page,
            "section": "Notes",
            "content_type": "table",
        },
    )


class TestProseMerging:
    def test_tiny_elements_merge_into_one_chunk(self):
        """Six fragments of one paragraph must not become six chunks."""
        chunks = chunk_documents([_prose(f"Sentence number {i}.") for i in range(6)])
        assert len(chunks) == 1
        assert "Sentence number 5." in chunks[0].page_content

    def test_different_sections_are_never_merged(self):
        chunks = chunk_documents(
            [_prose("Revenue rose.", section="A"), _prose("Emissions fell.", section="B")]
        )
        assert len(chunks) == 2

    def test_heading_path_is_prepended_to_the_chunk(self):
        """Self-containment and retrieval both need the topic in the text."""
        chunk = chunk_documents([_prose("Shell operates wind capacity.")])[0]
        assert chunk.page_content.startswith("Strategic Report > Outlook")

    def test_a_long_section_is_split(self):
        long_text = " ".join(f"word{i}" for i in range(4000))
        chunks = chunk_documents([_prose(long_text)])
        assert len(chunks) > 1

    def test_chunks_respect_the_configured_size(self):
        long_text = " ".join(f"word{i}" for i in range(4000))
        chunks = chunk_documents([_prose(long_text)])
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        headroom = settings.chunk_size + settings.chunk_overlap + 50
        assert all(len(enc.encode(c.page_content)) <= headroom for c in chunks)


class TestCitationPages:
    def test_page_range_reflects_the_pages_actually_covered(self):
        chunks = chunk_documents(
            [_prose("First page text.", page=10), _prose("Second page text.", page=11)]
        )
        assert (chunks[0].metadata["page"], chunks[0].metadata["page_end"]) == (10, 11)

    def test_single_page_section_reports_one_page(self):
        chunk = chunk_documents([_prose("Only here.", page=7)])[0]
        assert chunk.metadata["page"] == chunk.metadata["page_end"] == 7


class TestTables:
    GRID = "Segment carrying values.\n\n| Segment | Goodwill |\n|---|---|\n| Upstream | 5.3 |"

    def test_a_normal_table_passes_through_whole(self):
        """Splitting a grid strands rows from the header that gives them meaning."""
        chunks = chunk_documents([_table(self.GRID)])
        assert len(chunks) == 1
        assert chunks[0].page_content == self.GRID

    def test_a_table_is_never_merged_with_prose(self):
        chunks = chunk_documents([_prose("Intro."), _table(self.GRID), _prose("Outro.")])
        assert sum(1 for c in chunks if c.metadata["content_type"] == "table") == 1

    def test_document_order_is_preserved(self):
        chunks = chunk_documents(
            [_prose("Intro.", section="A"), _table(self.GRID), _prose("Outro.", section="B")]
        )
        assert [c.metadata["content_type"] for c in chunks] == ["prose", "table", "prose"]

    def test_an_oversized_table_splits_and_repeats_its_header(self):
        rows = "\n".join(f"| Row{i} | {i}.5 |" for i in range(900))
        chunks = chunk_documents([_table(f"Big table.\n\n| Segment | Value |\n|---|---|\n{rows}")])
        assert len(chunks) > 1
        # Every part must carry the header, or it is a grid of bare numbers.
        assert all("| Segment | Value |" in c.page_content for c in chunks)
        assert all(c.metadata["parts"] == len(chunks) for c in chunks)

    def test_no_rows_are_lost_when_a_table_splits(self):
        rows = [f"| Row{i} | {i}.5 |" for i in range(900)]
        chunks = chunk_documents(
            [_table("Big table.\n\n| Segment | Value |\n|---|---|\n" + "\n".join(rows))]
        )
        emitted = {
            ln for c in chunks for ln in c.page_content.splitlines() if ln.startswith("| Row")
        }
        assert emitted == set(rows)
