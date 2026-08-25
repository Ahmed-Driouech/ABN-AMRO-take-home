# Annual-Report RAG

A retrieval-augmented question-answering system over company annual reports
(Shell, ABN AMRO, ASML, Heineken, CM.com — 2024, ~1,900 pages). Python/FastAPI
backend, Vue frontend, runs locally.

```bash
cp .env.example .env      # add OPENAI_API_KEY and UNSTRUCTURED_API_KEY
make install
scripts/fetch_reports.sh  # downloads the five PDFs (~134 MB)
make parse                # PDF -> elements   (metered, cached by content hash)
make ingest               # chunk, embed, index
make extract              # pre-extract FTE + sustainability goals
make serve                # http://localhost:8000
```

Or in a container, which serves the same thing on port 8000:

```bash
docker compose up --build      # Compose V2 plugin
docker-compose up --build      # standalone binary (Docker Desktop installs this one)
```

`data/` is mounted as a volume, so the parse cache, Chroma index and SQLite
survive a rebuild — and the metered parsing step is never paid twice.

---

## What it does

- **Chat** over the reports, every answer carrying the page it came from.
- **Abstains** when the reports do not support an answer, rather than
  producing something plausible.
- **Pre-extracts** the FTE count and sustainability goals for each report at
  ingestion, stores them in SQLite, and shows them in the UI.
- **Persists** across restarts: Chroma and SQLite are files under `data/`.
- **Idempotent ingestion**: identity is the SHA-256 of the PDF bytes, so
  re-running ingest is free and a re-issued report is correctly treated as new.

## Results

Measured against a labelled set of 20 answerable and 3 unanswerable questions
(`eval/questions.yaml`). The set is self-validating — each question declares a
string that must appear on its expected page, asserted before scoring, because
an eval set pointing at the wrong page is worse than no eval set.

**Retrieval** (`make eval`):

| config | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|
| dense only | 15% | 30% | 30% | 45% |
| sparse only | 60% | 70% | 75% | 80% |
| **hybrid 50/50, RRF k=10** | 25% | 50% | 75% | **85%** |

Split by question type, the two halves win on opposite questions — which is
why both are kept:

| | dense R@10 | sparse R@10 |
|---|---|---|
| lexical, "find this figure" (14) | 36% | **93%** |
| paraphrased, wording absent from the page (6) | **67%** | 50% |

**Generation**: `cited 17/20`, `verbatim 7/7`, `abstained 3/3`.

*Verbatim* is a string check — does the figure verified by reading the source
page appear character-for-character in the answer? No LLM judge, so no judge
bias to argue about. *Abstained* is the column that matters most for a bank.

## How it works

```
PDF ──► Unstructured hi_res API ──► elements (Title / NarrativeText / Table …)
                                      │
        tables: reconcile two views ──┤
        prose: group by section    ───┤
                                      ▼
                                   chunks ──► Chroma (dense) + BM25 (sparse)
                                                  │
                        question ──► plan ──► hybrid search ──► answer + citation
                                      │
                        ingest ──► pre-extraction ──► SQLite ──► UI
```

| Layer | Choice | Why |
|---|---|---|
| Parsing | Unstructured `hi_res`, hosted API | Annual-report tables are often borderless and the narrative is multi-column. Running it as an API keeps torch and multi-GB weights out of the image. Maps onto **Azure AI Document Intelligence**. |
| Vector store | Chroma | Embedded, persists to a directory, no server. Maps onto **Azure AI Search**. |
| Sparse | `rank_bm25`, one index per company | BM25 cannot filter metadata; scoring it over the whole corpus and fusing with a company-filtered dense ranking would leak other issuers in through the sparse side. |
| Relational | SQLite | Pre-extracted datapoints are queried exactly, not by similarity. |
| LLM | OpenAI | Maps onto **Azure OpenAI**. |

**Filtering is deliberately asymmetric.** Company is a *hard* filter — a
Heineken figure returned for a Shell question is the one error that is never
acceptable. Year is a *soft* signal, because a single table often spans
2016–2024 and filtering on it drops correct chunks.


## Repository

```
backend/app/parsing.py     PDF -> Documents (Unstructured + table reconciliation)
backend/app/chunking.py    elements -> chunks
backend/app/store.py       Chroma + BM25 + weighted RRF
backend/app/answer.py      planning, grounded answering, citation resolution
backend/app/extraction.py  pre-extraction -> SQLite
backend/app/api.py         FastAPI
frontend/src/App.vue       chat + pre-extracted data
eval/                      labelled question set, retrieval and answer scoring
backend/tests/             51 unit tests
```

`make test` · `make eval` · `make lint`
