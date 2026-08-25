"""Tests for the parser audit.

The audit is the evidence behind the parser's design, so it has to be right
about what counts as a mismatch -- an earlier version regexed raw text and
pulled "172" out of "Section 172(1)", understating fidelity by six points.
"""

from __future__ import annotations

from langchain_core.documents import Document

from eval.parser_audit import audit_numbers, normalise_figure


class TestNormaliseFigure:
    """The same number is decorated differently in the two sources, and
    comparing decorated strings reports identical digits as a mismatch."""

    def test_percent_suffix_is_stripped(self):
        # The text layer prints "11.3" and "%" as separate tokens.
        assert normalise_figure("11.3%") == "11.3"

    def test_accounting_parentheses_are_stripped(self):
        assert normalise_figure("(4.9)") == "4.9"

    def test_currency_symbol_is_stripped(self):
        assert normalise_figure("$1,234") == "1,234"

    def test_footnote_marker_is_stripped(self):
        assert normalise_figure("73.9[A]") == "73.9"

    def test_significant_digits_are_never_altered(self):
        # A trailing zero is meaningful in a financial statement.
        assert normalise_figure("60.0") == "60.0"
        assert normalise_figure("2.10") == "2.10"


class TestAuditNumbers:
    def _table(self, content: str, page: int = 1):
        return Document(page_content=content, metadata={"content_type": "table", "page": page})

    def test_figures_printed_on_the_page_are_accounted_for(self):
        assert audit_numbers(
            [self._table("| Integrated Gas | 4.9 | 60.0 |")], {1: {"4.9", "60.0"}}
        ) == (2, 0)

    def test_a_figure_absent_from_the_page_is_counted(self):
        """49 is what OCR damage of 4.9 looks like when it leaks through."""
        assert audit_numbers([self._table("| Integrated Gas | 49 |")], {1: {"4.9"}}) == (1, 1)

    def test_decoration_does_not_count_as_a_mismatch(self):
        _, unmatched = audit_numbers([self._table("| Gearing | 11.3% |")], {1: {"11.3"}})
        assert unmatched == 0

    def test_digits_inside_words_are_not_treated_as_figures(self):
        """ "Section 172(1)" is a cross-reference, not two damaged numbers."""
        assert audit_numbers(
            [self._table("| Principal decisions (Section 172(1)) | 4.9 |")], {1: {"4.9"}}
        ) == (1, 0)

    def test_prose_chunks_are_not_audited(self):
        prose = Document(
            page_content="Revenue rose 12 percent", metadata={"content_type": "prose", "page": 1}
        )
        assert audit_numbers([prose], {1: set()}) == (0, 0)

    def test_the_audit_never_edits_the_document(self):
        doc = self._table("| Integrated Gas | 49 |")
        audit_numbers([doc], {1: {"4.9"}})
        assert doc.page_content == "| Integrated Gas | 49 |"
