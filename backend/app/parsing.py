"""PDF -> LangChain Documents, one per report element.

Parsing runs through Unstructured's hosted ``hi_res`` API. Annual-report
tables are frequently borderless, the narrative is laid out in magazine
columns, and five issuers means five typesetting pipelines, so a layout model
earns its keep. Running it as an API keeps torch out of the image, and maps
onto Azure AI Document Intelligence in ABN's own environment.

Tables need care, because ``hi_res`` returns two views of each one and both
are wrong in different ways:

* ``text_as_html`` has the right *structure* but damaged *numbers* -- it is
  re-OCRed from the rendered grid, so decimal separators are lost and Shell's
  goodwill of 4.9 comes back as 49. Corpus-wide, 2-3% of its figures do not
  appear in the PDF at all.
* ``text`` has the right *numbers* -- it is lifted from the embedded text
  layer, and only 0.01-0.33% of its figures are unaccounted for -- but the
  reading order is scrambled, with a column often emitted after all the others.

So the two are handed to a model together and reconciled: structure from the
HTML, every figure from the text. The prompt forbids repairing a damaged
number, so the model transcribes and never does arithmetic.
"""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_core.documents import Document

from app.config import settings

# Running headers, footers and page furniture repeat on every page.
DROP = {"Header", "Footer", "PageBreak", "PageNumber", "Image", "Formula"}
# Both carry section structure. A Header is the running section title
# ("Financial Statements | Notes to the Consolidated Financial Statements")
# and is what Titles hang off, so it belongs in the path even though it is
# never emitted as a chunk of its own.
HEADINGS = {"Title", "Header"}
PROSE = {"NarrativeText", "ListItem", "UncategorizedText", "FigureCaption", "Address"}

TABLE_PROMPT = """\
Rebuild this table from a company annual report as a markdown table.

You are given two imperfect views of the SAME table:

1. LAYOUT - correct row/column structure, but numbers are damaged: decimal
   points and thousands separators were lost by OCR (4.9 appears as 49).
2. VALUES - every figure exactly as printed, but the reading order is
   scrambled: a column is often emitted after all the others.

Method:
- Take the grid shape, row order and headers from LAYOUT.
- Take every figure from VALUES. Match each cell by its damaged counterpart
  in LAYOUT: 49 in LAYOUT is the figure in VALUES whose digits are 49.
- HARD RULE: every figure you output must appear character-for-character in
  VALUES. Never repair a LAYOUT number yourself by inserting a decimal point,
  and never round, recompute or invent one. If a figure is not in VALUES,
  leave the cell blank.
- Copy dashes exactly: an em dash (\u2014) means nil and must stay an em dash.

Start with one sentence describing the table, then the markdown table,
including the unit and reporting year if the context gives them.

CONTEXT:
{context}

LAYOUT:
{layout}

VALUES:
{values}
"""


# A markdown separator row: pipes, dashes, colons and space. It may be missing
# its closing pipe when the model was cut off mid-line.
SEPARATOR_ROW = re.compile(r"^\s*\|[\s:|-]*$")


def tidy_markdown(text: str) -> str:
    """Clean up the model's markdown.

    Every response arrives wrapped in a ``` fence, and a separator row is
    sometimes padded to the width of its widest cell -- in one case to a
    million characters, which is 16k tokens of nothing.
    """
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
    lines = [re.sub(r"-{4,}", "---", ln) if SEPARATOR_ROW.match(ln) else ln for ln in lines]
    return "\n".join(lines).strip()


def doc_id_for(path: str | Path) -> str:
    """SHA-256 of the PDF bytes: identity is content, not filename, so the
    same report under two names ingests once and a re-issue is a new document."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _elements(path: Path, doc_id: str, refresh: bool) -> list[dict]:
    """Raw Unstructured elements, cached under the PDF's content hash.

    Partitioning is the only metered step, so it is paid once; re-ingesting is
    free and the app still demos if the API is unreachable.
    """
    cache = settings.parse_cache_dir / f"{doc_id}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    if not settings.unstructured_api_key:
        raise RuntimeError(f"{path.name} is not cached and UNSTRUCTURED_API_KEY is empty.")

    from langchain_unstructured import UnstructuredLoader

    loader = UnstructuredLoader(
        file_path=str(path),
        strategy=settings.unstructured_strategy,
        partition_via_api=True,
        api_key=settings.unstructured_api_key,
        # A 500-page report is only tractable if pages go up in parallel.
        split_pdf_page=True,
        split_pdf_concurrency_level=settings.unstructured_concurrency,
    )
    elements = [
        {
            "type": d.metadata.get("category", ""),
            "text": d.page_content,
            "element_id": d.metadata.get("element_id"),
            "metadata": d.metadata,
        }
        for d in loader.load()
    ]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(elements))
    return elements


def _context_for(elements: list[dict], i: int, heading: list[str]) -> str:
    """A few lines around the grid: the caption, the floating unit and the
    detached year all live outside the table in the source, and the model
    cannot rebuild the header row without them."""
    page = (elements[i].get("metadata") or {}).get("page_number")
    before = [
        (e.get("text") or "").strip()
        for e in elements[max(0, i - 4) : i]
        if (e.get("metadata") or {}).get("page_number") == page
    ]
    # Surrounding text first: it carries the caption, unit and year, and must
    # survive the length cap ahead of the section label.
    return "\n".join(filter(None, [*before, " > ".join(heading)]))[:1500]


def _rewrite_tables(jobs: list[tuple[str, str, str, str]], doc_id: str) -> dict[str, str]:
    """Reconcile each table's two views into one markdown grid, cached per
    document so the rewrite is paid once and is a fixed artefact after that."""
    cache_path = settings.table_cache_dir / f"{doc_id}.json"
    cached: dict[str, str] = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [j for j in jobs if j[0] not in cached]
    if not todo:
        return cached

    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)

    def one(job: tuple[str, str, str, str]) -> tuple[str, str]:
        element_id, values, layout, context = job
        prompt = TABLE_PROMPT.format(context=context, layout=layout, values=values)
        response = client.chat.completions.create(
            model=settings.chat_model,
            temperature=0,
            max_tokens=settings.table_rewrite_max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return element_id, tidy_markdown(response.choices[0].message.content or "")

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    def flush() -> None:
        cache_path.write_text(json.dumps(cached))

    # A report is hundreds of calls; checkpoint as they land so an interrupted
    # run resumes instead of paying for the same tables again.
    with ThreadPoolExecutor(max_workers=settings.table_rewrite_workers) as pool:
        for done, (element_id, markdown) in enumerate(pool.map(one, todo), start=1):
            cached[element_id] = markdown
            if done % 25 == 0:
                flush()
    flush()
    return cached


def section_paths(elements: list[dict]) -> list[list[str]]:
    """The heading path each element sits under, in document order.

    Section context lets a chunk say where it came from, and improves
    retrieval by embedding the chunk with its topic. Unstructured nests
    elements via ``parent_id``: content hangs off a Title, and a Title hangs
    off the running Header. Both levels are needed -- resolving only Titles
    leaves every heading looking top-level.
    """
    by_id: dict[str, list[str]] = {}
    latest: list[str] = []
    out: list[list[str]] = []
    for el in elements:
        parent = (el.get("metadata") or {}).get("parent_id") or ""
        enclosing = by_id.get(parent)  # None when absent or pointing at a dropped element
        out.append(enclosing if enclosing is not None else latest)
        if el.get("type") in HEADINGS and (text := (el.get("text") or "").strip()):
            # A heading nests only under a parent we actually resolved. Treating
            # an unresolved parent as "under the previous heading" instead makes
            # every heading chain onto the last one, and the path grows without
            # bound -- measured at a mean depth of 1,009 on the Shell report.
            own = [*enclosing, text] if enclosing is not None else [text]
            if eid := el.get("element_id"):
                by_id[eid] = own
            latest = own
    return out


def load_report(
    path: str | Path, company: str, year: int, *, refresh: bool = False
) -> list[Document]:
    """Parse one annual report into Documents ready for chunking.

    ``company`` and ``year`` are supplied by the caller, never inferred from
    the page: a figure attributed to the wrong issuer is the one error a bank
    cannot tolerate.
    """
    path = Path(path)
    doc_id = doc_id_for(path)
    elements = _elements(path, doc_id, refresh)

    paths = section_paths(elements)

    jobs = [
        (
            el["element_id"],
            el.get("text") or "",
            (el.get("metadata") or {}).get("text_as_html") or "",
            _context_for(elements, i, paths[i]),
        )
        for i, el in enumerate(elements)
        if el.get("type") == "Table" and el.get("element_id") and (el.get("text") or "").strip()
    ]
    markdown_by_id = _rewrite_tables(jobs, doc_id)

    docs: list[Document] = []
    for i, el in enumerate(elements):
        category = el.get("type") or ""
        if category in DROP:
            continue
        text = (el.get("text") or "").strip()
        if not text:
            continue
        meta = el.get("metadata") or {}
        base = {
            "company": company,
            "report_year": year,
            "doc_id": doc_id,
            "page": meta.get("page_number"),
            "section": " > ".join(paths[i]),
            "content_type": "table" if category == "Table" else "prose",
        }
        if category == "Table":
            rendered = markdown_by_id.get(el["element_id"])
            docs.append(
                Document(
                    page_content=tidy_markdown(rendered) if rendered else text,
                    metadata=base,
                )
            )
        elif category in PROSE:
            docs.append(Document(page_content=text, metadata=base))

    return docs
