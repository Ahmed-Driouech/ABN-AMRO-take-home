"""Report elements -> retrievable chunks.

Parsing emits one Document per layout element, which is far too fine to
retrieve: the median prose element is **7 tokens** and 61% are under 20, so
handing them to a splitter does nothing at all -- everything is already below
any sane chunk size. The work here is the opposite of splitting. Elements are
regrouped into section-sized text first, and only then split.

Section boundaries come from the document's own heading structure, so the
splitter never has to rediscover where a topic starts; it only handles the
tail of sections that run long.

Tables are left whole. They are already self-contained markdown that opens
with a description sentence, and splitting a grid strands rows from the
header that gives them meaning.
"""

from __future__ import annotations

from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def _splitter() -> RecursiveCharacterTextSplitter:
    # Token-based rather than character-based: what matters downstream is the
    # generation context budget, which is denominated in tokens.
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _split_table(doc: Document) -> list[Document]:
    """Split an oversized table into row groups, repeating its description and
    header in each.

    Only about 1% of tables need this, but the tail is extreme -- the largest
    is 16k tokens, which alone would swamp the generation context. A row group
    that lost its header row would be a grid of bare numbers, so the header is
    carried into every piece.
    """
    lines = doc.page_content.splitlines()
    header = [ln for ln in lines if ln.strip().startswith("|")][:2]
    preamble = [ln for ln in lines if not ln.strip().startswith("|")]
    rows = [ln for ln in lines if ln.strip().startswith("|")][2:]
    if not rows or not header:
        return [doc]

    splitter = _splitter()
    fixed = "\n".join([*preamble, *header])
    budget = max(1, settings.table_max_tokens - splitter._length_function(fixed))

    out, current, size = [], [], 0
    for row in rows:
        n = splitter._length_function(row)
        if current and size + n > budget:
            out.append("\n".join([fixed, *current]))
            current, size = [], 0
        current.append(row)
        size += n
    if current:
        out.append("\n".join([fixed, *current]))

    total = len(out)
    return [
        Document(
            page_content=text,
            metadata={**doc.metadata, "part": i + 1, "parts": total},
        )
        for i, text in enumerate(out)
    ]


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Turn parsed elements into chunks ready to embed.

    Order is preserved so that a chunk's neighbours in the list are its
    neighbours in the report.
    """
    splitter = _splitter()

    # Group prose by section, keeping document order and remembering which
    # page each stretch of text came from.
    groups: dict[tuple[str, str], list[Document]] = defaultdict(list)
    order: list[tuple[str, tuple[str, str]] | tuple[str, Document]] = []
    seen: set[tuple[str, str]] = set()

    for doc in docs:
        if doc.metadata.get("content_type") == "table":
            order.append(("table", doc))
            continue
        key = (doc.metadata.get("doc_id", ""), doc.metadata.get("section", ""))
        groups[key].append(doc)
        if key not in seen:
            seen.add(key)
            order.append(("prose", key))

    out: list[Document] = []
    for kind, item in order:
        if kind == "table":
            doc = item  # type: ignore[assignment]
            if splitter._length_function(doc.page_content) > settings.table_max_tokens:
                out.extend(_split_table(doc))
            else:
                out.append(doc)
            continue
        out.extend(_chunk_section(groups[item], splitter))  # type: ignore[arg-type]
    return out


def _chunk_section(
    elements: list[Document], splitter: RecursiveCharacterTextSplitter
) -> list[Document]:
    """Join one section's elements, split it, and give each chunk back the
    pages it actually came from."""
    joined = ""
    spans: list[tuple[int, int, int]] = []
    for el in elements:
        start = len(joined)
        joined += el.page_content.strip() + "\n\n"
        if page := el.metadata.get("page"):
            spans.append((start, len(joined), page))
    joined = joined.strip()
    if not joined:
        return []

    base = dict(elements[0].metadata)
    section = base.get("section") or ""
    out: list[Document] = []
    cursor = 0
    for text in splitter.split_text(joined):
        # Locate the chunk so its citation names the pages it really spans,
        # rather than the first page of a section that may run across several.
        found = joined.find(text, cursor)
        start = found if found != -1 else cursor
        cursor = start + max(1, len(text) // 2)
        pages = [p for s, e, p in spans if not (e <= start or s >= start + len(text))]
        # The heading path both self-describes the chunk and gives the
        # embedding its topic context.
        body = f"{section}\n\n{text}" if section else text
        out.append(
            Document(
                page_content=body,
                metadata={
                    **base,
                    "page": min(pages) if pages else base.get("page"),
                    "page_end": max(pages) if pages else base.get("page"),
                },
            )
        )
    return out
