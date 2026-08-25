"""The index: embedding, persistence, and hybrid retrieval.

Chroma holds the chunks and their embeddings and persists to a local
directory, so data survives a restart with no server to run.

Retrieval is **hybrid**, because the two halves fail on different questions.
Dense embedding search handles loose phrasing ("what are Shell's climate
targets"), but embeddings are poor at exact terminology and hopeless at
numbers. Sparse BM25 handles the literal case -- "climate change adaptation"
has to match those words -- which is precisely the shape of the questions
this application exists to answer. The two rankings are fused with Reciprocal
Rank Fusion.

Filtering is deliberately asymmetric:

* **company is a hard filter.** Returning a Heineken figure for a question
  about Shell is the one error that is never acceptable, so other issuers are
  excluded before ranking rather than hoped away by it.
* **year is a soft signal.** A single table often spans 2016-2024, and a
  report discusses prior years throughout, so filtering on year drops correct
  chunks. It is left to ranking and to the answer step.
"""

from __future__ import annotations

import functools
import hashlib
import re
import threading

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.chunking import chunk_documents
from app.config import settings
from app.parsing import doc_id_for, load_report

COLLECTION = "annual_reports"
# Chroma rejects oversized batches; embedding calls are batched to stay under it.
BATCH = 2000
# How deep each retriever goes before fusion. Wider than the final k so a chunk
# ranked mid-table by one method can still be rescued by the other.
CANDIDATES = 30
# RRF damping. The textbook value is 60, which is tuned for large candidate
# pools; with 30 candidates per retriever it flattens the top ranks to near
# equality. Measured on the eval set, 10 lifts recall@10 from 80% to 85%.
RRF_K = 10

# Keeps a figure whole -- "16.0" and "1,234" must stay single tokens, since
# matching a number exactly is the point of the sparse half.
_TOKEN = re.compile(r"[a-z0-9][a-z0-9.,%-]*")


def _tokenise(text: str) -> list[str]:
    """Normalise text for BM25.

    LangChain's BM25Retriever defaults to ``text.split()``, which leaves
    punctuation attached: "goodwill," in a chunk and "goodwill" in a question
    are then different terms and never match. That single default put the
    Shell goodwill table outside the top 30 for a question about Shell's
    goodwill; normalising brings it to rank 1.
    """
    return [token.strip(".,-") for token in _TOKEN.findall(text.lower())]


# Chroma's client initialisation is not thread-safe: two threads opening the
# persistent client at once fail with "Could not connect to tenant". The API
# serves requests concurrently, so construction is guarded.
_STORE_LOCK = threading.Lock()


@functools.lru_cache(maxsize=1)
def _open_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=OpenAIEmbeddings(
            model=settings.embedding_model, api_key=settings.openai_api_key
        ),
        persist_directory=str(settings.chroma_dir),
    )


def get_store() -> Chroma:
    with _STORE_LOCK:
        return _open_store()


def ingested_documents() -> dict[str, dict]:
    """Which reports are already in the index, by content hash.

    Chroma is the ledger: the chunks carry their ``doc_id``, so there is no
    second store to keep in step with it.
    """
    records = get_store().get(include=["metadatas"])
    out: dict[str, dict] = {}
    for meta in records.get("metadatas") or []:
        entry = out.setdefault(
            meta["doc_id"],
            {"company": meta.get("company"), "year": meta.get("report_year"), "chunks": 0},
        )
        entry["chunks"] += 1
    return out


def ingest_report(path: str, company: str, year: int, *, force: bool = False) -> dict:
    """Parse, chunk and index one report, skipping work already done.

    Identity is the PDF's content hash, so re-uploading the same report under
    a different filename is a no-op, while a corrected re-issue is correctly
    treated as a new document.
    """
    doc_id = doc_id_for(path)
    existing = ingested_documents().get(doc_id)
    if existing and not force:
        return {"doc_id": doc_id, "status": "skipped", "chunks": existing["chunks"]}

    store = get_store()
    if existing:
        store.delete(where={"doc_id": doc_id})

    chunks = chunk_documents(load_report(path, company=company, year=year))
    # Chroma metadata values must be scalars, and a missing key breaks a
    # `where` clause on it, so every chunk gets the same flat shape.
    for chunk in chunks:
        chunk.metadata = {k: v for k, v in chunk.metadata.items() if v is not None}

    for i in range(0, len(chunks), BATCH):
        store.add_documents(chunks[i : i + BATCH])

    _load_corpus.cache_clear()
    _bm25.cache_clear()
    return {"doc_id": doc_id, "status": "indexed", "chunks": len(chunks)}


@functools.lru_cache(maxsize=1)
def _load_corpus() -> list[Document]:
    """Every indexed chunk, read back out of Chroma.

    BM25 needs the full text of the corpus in memory. Reading it from Chroma
    rather than keeping a second copy means there is exactly one place a chunk
    lives, and the sparse index is rebuilt from it on start-up.
    """
    records = get_store().get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(
            records.get("documents") or [], records.get("metadatas") or [], strict=False
        )
    ]


@functools.lru_cache(maxsize=16)
def _bm25(company: str | None):
    """A sparse index over one company's chunks, or the whole corpus.

    Building one index per company is what keeps the two halves of the hybrid
    comparable. BM25 cannot filter by metadata, so scoring it over everything
    and fusing that with a company-filtered dense ranking would let other
    issuers' chunks in through the sparse side.
    """
    from langchain_community.retrievers import BM25Retriever

    docs = [d for d in _load_corpus() if not company or d.metadata.get("company") == company]
    if not docs:
        return None
    retriever = BM25Retriever.from_documents(docs, preprocess_func=_tokenise)
    retriever.k = CANDIDATES
    return retriever


def _rrf(rankings: list[list[Document]], k: int) -> list[Document]:
    """Reciprocal Rank Fusion.

    Fuses on rank rather than score, so the dense and sparse scores -- which
    are on unrelated scales -- never have to be made comparable.
    """
    scores: dict[str, float] = {}
    best: dict[str, Document] = {}
    for ranking in rankings:
        for position, doc in enumerate(ranking):
            # Hash the whole chunk. Keying on a prefix merges the parts of a
            # split table, which all repeat the same description and header
            # row -- so every part but the first would be silently dropped.
            key = hashlib.sha1(doc.page_content.encode()).hexdigest()
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + position + 1)
            best.setdefault(key, doc)
    ranked = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [best[key] for key in ranked[:k]]


def search(query: str, *, company: str | None = None, k: int = 10) -> list[Document]:
    """Hybrid retrieval over the indexed reports.

    Dense and sparse are weighted equally. Measured on the eval set they win
    on different questions -- sparse reaches 93% recall@10 on figure lookups
    where dense manages 36%, and dense reaches 67% on paraphrased questions
    where sparse manages 50% -- so neither is worth dropping. Together they
    reach 85% recall@10, above either alone.
    """
    store = get_store()
    where = {"company": company} if company else None
    dense = store.similarity_search(query, k=CANDIDATES, filter=where)

    sparse_retriever = _bm25(company)
    sparse = sparse_retriever.invoke(query) if sparse_retriever else []

    return _rrf([dense, sparse], k)
