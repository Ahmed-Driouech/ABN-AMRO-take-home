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

## Three findings worth the detail

**1. Read tables from `text`, not `text_as_html`.** `hi_res` returns both.
The HTML is re-OCRed from the rendered grid and loses decimal separators —
Shell's goodwill of `4.9` comes back as `49`. Corpus-wide, 2–3% of figures in
`text_as_html` appear nowhere in the PDF, against 0.01–0.33% in `text`. But
`text` has no structure: rows and columns arrive interleaved. So both views go
to the model together — structure from the HTML, every figure from the text,
with an explicit rule never to repair a damaged number. Without that rule it
"helpfully" turned `5.7` into `57.0`.

`eval/parser_audit.py` measures the result: **98.86% of figures in the
rendered tables appear character-for-character on their source page.** It runs
outside the application and never edits anything.

**2. LangChain's `BM25Retriever` defaults to `text.split()`.** No lowercasing,
no punctuation stripping — so `"goodwill,"` in a chunk and `"goodwill"` in a
question are different terms. The Shell goodwill table sat outside the top 30
for a question about Shell's goodwill. One tokenizer took sparse recall@10
from **71% to 93%**. Only the labelled set surfaced it.

**3. The splitter is a guard, not the mechanism.** `hi_res` emits one element
per paragraph fragment — median **7 tokens**, 61% under 20 — so running a
splitter over them is a no-op. Grouping by the document's own heading
hierarchy does the work: 35,708 elements become 6,074 sections, and
`RecursiveCharacterTextSplitter` fires on only **1.8%** of them, the ones that
run long (largest: 18,214 tokens).

## Known limitations

**The example question in the brief currently returns a wrong answer.** Asked
what Shell spent on climate change adaptation in 2024, the system answers
`$1,849 million`. The correct answer is **nil** — the source row reads
`698 0.2% 698 0.2% — — 1,849 6.0% 1,849 6.0% — —`, and under the headers
`CCM+CCA | mitigation | adaptation` the adaptation column is an em-dash. The
1,849 is *mitigation*.

The fault is the table, not the model. That EU Taxonomy template has ~14
columns with **vertically-rotated headers**, which `hi_res` reads as
`t t t / n n n / u u u / o o o / m A % m A %`, and whose HTML headers come back
scrambled. Both views are unusable, so the reconstruction placed figures under
the wrong objective and dropped the em-dashes.

Unstructured's `vlm` strategy parses that page correctly in 50 seconds
(dashes intact). Switching the default is one line in `config.py`, and a
full re-parse is ~3–4 hours of background work, cached permanently. It was not
run for time.

Note also what this exposes about the audit: `1,849` **is** on the page, so a
presence check cannot detect a *placement* error. The audit measures
transcription, not column alignment.

Other limitations:

- **Retrieval is 85% recall@10**, so roughly one question in seven cannot be
  answered from what is retrieved. The next lever is a cross-encoder reranker
  over the fused top-30; the eval harness exists to prove whether it helps.
- **Generation eval is a string check.** Faithfulness scoring via LLM-as-judge
  is not built. It needs the judge itself validated, and position/verbosity
  bias controlled, before its numbers mean anything.
- **20 answerable questions is a small set.** Directionally sound, not tight.
- **Single-turn is well tested; multi-turn is not.** The planner rewrites
  follow-ups into standalone questions, but no eval covers it.

## Assumptions

- **Multi-document, cross-company** support via company/year metadata, even
  though the example question is single-company.
- **Company and year are supplied at upload**, never inferred from the page.
- **Verbatim datapoint** means the figure exactly as printed, plus a citation,
  and an abstention when unsupported.
- **The extraction schema is extensible** — FTE and sustainability goals are
  two shapes (a scalar and an open list); adding revenue or net profit is a
  new Pydantic model and a query.
- **You supply your own API keys** in `.env`. None are committed.
- **The PDFs are not committed** (~134 MB); `scripts/fetch_reports.sh` pulls
  them from the issuers' own sites.

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
