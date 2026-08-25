"""Tests for PDF parsing.

The theme is verbatim fidelity. hi_res gives two views of every table and
both are wrong in different ways -- the HTML has damaged numbers, the text
has scrambled order -- so the pieces that guard against a wrong figure
reaching an answer are pinned here.
"""

from __future__ import annotations

from app.parsing import TABLE_PROMPT, section_paths, tidy_markdown


def _el(kind: str, text: str, eid: str | None = None, **meta):
    return {
        "type": kind,
        "text": text,
        "element_id": eid or text[:12],
        "metadata": {"page_number": 25, **meta},
    }


class TestSectionPaths:
    """Unstructured nests content under a Title, and a Title under the running
    Header. Getting this wrong is not cosmetic: an earlier version treated an
    unresolved parent as "under the previous heading", which chained every
    heading onto the last one and produced paths 1,009 deep on the Shell
    report -- poisoning the metadata and embedding text of every chunk."""

    def test_content_nests_under_header_and_title(self):
        paths = section_paths(
            [
                _el("Header", "Financial Statements", eid="h1"),
                _el("Title", "4. Climate change", eid="t1", parent_id="h1"),
                _el("NarrativeText", "Shell launched its strategy.", parent_id="t1"),
            ]
        )
        assert paths[-1] == ["Financial Statements", "4. Climate change"]

    def test_sibling_headings_replace_rather_than_chain(self):
        paths = section_paths(
            [
                _el("Title", "Section A", eid="a"),
                _el("Title", "Section B", eid="b"),
                _el("NarrativeText", "body", parent_id="b"),
            ]
        )
        assert paths[-1] == ["Section B"]

    def test_depth_stays_bounded_over_many_headings(self):
        elements = [_el("Title", f"Heading {i}", eid=f"h{i}") for i in range(50)]
        assert max(len(p) for p in section_paths(elements)) <= 2

    def test_unresolved_parent_falls_back_to_the_latest_heading(self):
        paths = section_paths(
            [
                _el("Title", "4. Climate change", eid="t1"),
                _el("Table", "Goodwill 4.9", parent_id="does-not-exist"),
            ]
        )
        assert paths[-1] == ["4. Climate change"]


class TestTablePrompt:
    """The prompt is load-bearing: it is the only thing stopping the model
    from 'helpfully' repairing a damaged number into a plausible wrong one."""

    def test_forbids_the_model_repairing_numbers_itself(self):
        assert "character-for-character" in TABLE_PROMPT
        assert "Never repair" in TABLE_PROMPT

    def test_preserves_em_dash_nils(self):
        assert "—" in TABLE_PROMPT

    def test_supplies_both_views(self):
        assert "{layout}" in TABLE_PROMPT and "{values}" in TABLE_PROMPT


class TestTidyMarkdown:
    """The model's raw output needs two things removed before it is stored."""

    def test_code_fence_is_stripped(self):
        # Every single response comes back wrapped in one.
        assert "```" not in tidy_markdown("desc\n\n```markdown\n| a |\n|---|\n```")

    def test_runaway_separator_row_is_collapsed(self):
        """One table returned a separator row a million characters long -- the
        model hit its output limit mid-line, so it never closed the row."""
        out = tidy_markdown("| a | b |\n|" + "-" * 200_000)
        assert len(out) < 100
        assert "| a | b |" in out

    def test_data_rows_are_never_altered(self):
        grid = "| Integrated Gas | 4.9 | 60.0 |"
        assert grid in tidy_markdown(f"desc\n\n| A | B | C |\n|---|---|---|\n{grid}")
