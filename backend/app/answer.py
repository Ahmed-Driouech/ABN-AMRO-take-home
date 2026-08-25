"""Question -> grounded answer, or an honest refusal.

Two model calls, each doing one job.

**Planning.** The question is resolved into a company, any years mentioned,
and a standalone rewording that carries the context of earlier turns ("and
for 2023?" means nothing on its own). The company is then validated against
the reports actually indexed, so a plan can never filter retrieval down to a
company we do not hold -- the model proposes, the ledger decides.

**Answering.** The model sees only the retrieved chunks and is required to
quote figures exactly as printed. Where it cannot answer from them it must
say so: "not found" is a first-class result, not a failure. For a bank, a
confident wrong figure is far worse than an admission of ignorance.

Citations are not written by the model. It cites the *numbered* chunks it
used and those numbers are mapped back to real pages here, so a page
reference cannot be invented.
"""

from __future__ import annotations

import functools

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config import settings
from app.store import ingested_documents, search


class QueryPlan(BaseModel):
    """How a question should be routed to the index."""

    company: str | None = Field(
        description="Company the question is about, exactly as named in the list of "
        "available companies, or null if it names none of them."
    )
    years: list[int] = Field(description="Reporting years mentioned, empty if none.")
    standalone_question: str = Field(
        description="The question rewritten to stand alone, resolving any reference "
        "to the earlier conversation."
    )


class Answer(BaseModel):
    """A grounded answer, or an explicit refusal."""

    found: bool = Field(description="True only if the context supports an answer.")
    answer: str = Field(
        description="The answer, quoting figures exactly as they appear. If found is "
        "false, a one-line explanation of what is missing."
    )
    sources: list[int] = Field(
        description="Numbers of the context excerpts the answer relies on. Empty if "
        "found is false."
    )


class Citation(BaseModel):
    company: str
    page: int
    page_end: int
    section: str
    content_type: str


class AnswerResult(BaseModel):
    question: str
    found: bool
    answer: str
    citations: list[Citation]
    company: str | None = None


PLAN_PROMPT = """\
Route this question about company annual reports.

Available companies (use one of these names exactly, or null):
{companies}

Rules:
- Pick a company only if the question clearly refers to one of the above,
  including by an obvious alias or description ("the Dutch bank" is ABN AMRO).
- If it names a company that is not listed, return null.
- Resolve follow-ups against the conversation so the rewritten question stands
  on its own.

{history}Question: {question}
"""

ANSWER_PROMPT = """\
Answer the question using only the excerpts below, which come from company
annual reports.

Rules:
- Use only these excerpts. Never use outside knowledge, and never calculate a
  figure that is not printed.
- Quote figures exactly as they appear, including units, currency and any
  thousands separators or decimals.
- State the unit and the reporting year when the excerpt gives them.
- State the basis of a figure whenever the excerpts distinguish one: year-end
  against average for the year, headcount against full-time equivalents,
  group against a single segment.
- A report often prints more than one figure for the same concept on
  different bases. Give the one the question asks for, and say that the other
  exists rather than choosing between them silently. Picking one without
  saying so reads as the definitive figure when it is one of several.
- If the excerpts do not contain the answer, set found to false and say what
  is missing. Do not guess, and do not offer a related figure as if it were
  the answer.
- In sources, list the numbers of the excerpts you actually used.

Excerpts:
{context}

Question: {question}
"""


@functools.lru_cache(maxsize=1)
def _model() -> ChatOpenAI:
    # Low temperature: the same question should give the same answer, and
    # faithfulness matters more than fluency.
    return ChatOpenAI(model=settings.chat_model, temperature=0, api_key=settings.openai_api_key)


def available_companies() -> list[str]:
    return sorted({d["company"] for d in ingested_documents().values() if d.get("company")})


def plan_query(question: str, history: list[tuple[str, str]] | None = None) -> QueryPlan:
    """Resolve a question into a retrieval plan, constrained by what is indexed."""
    companies = available_companies()
    transcript = ""
    if history:
        turns = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history[-3:])
        transcript = f"Conversation so far:\n{turns}\n\n"

    plan: QueryPlan = _model().with_structured_output(QueryPlan).invoke(
        PLAN_PROMPT.format(
            companies="\n".join(f"- {c}" for c in companies) or "(none indexed)",
            history=transcript,
            question=question,
        )
    )
    # The model proposes; the ledger decides. A company we do not hold would
    # filter retrieval down to nothing and produce a silent empty answer.
    if plan.company not in companies:
        plan.company = None
    return plan


def _format_context(chunks: list[Document]) -> str:
    """Number each excerpt and label it, so the model can cite by number and
    can see which company, year and page it is reading."""
    blocks = []
    for i, doc in enumerate(chunks, start=1):
        meta = doc.metadata
        pages = (
            f"p. {meta.get('page')}"
            if meta.get("page") == meta.get("page_end", meta.get("page"))
            else f"pp. {meta.get('page')}-{meta.get('page_end')}"
        )
        header = f"[{i}] {meta.get('company')} {meta.get('report_year')} annual report, {pages}"
        if section := meta.get("section"):
            header += f" | {section}"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def _resolve_citations(result: Answer, chunks: list[Document]) -> list[Citation]:
    """Map cited excerpt numbers back to the chunks retrieval returned.

    The model never writes a page number. Left to do so it can produce one
    that is entirely plausible and entirely invented, which is indistinguishable
    from a real citation to anyone reading the answer.
    """
    citations: list[Citation] = []
    for number in dict.fromkeys(result.sources):
        if 1 <= number <= len(chunks):
            meta = chunks[number - 1].metadata
            citations.append(
                Citation(
                    company=meta.get("company", ""),
                    page=meta.get("page") or 0,
                    page_end=meta.get("page_end") or meta.get("page") or 0,
                    section=meta.get("section", ""),
                    content_type=meta.get("content_type", ""),
                )
            )
    return citations


def answer_question(
    question: str, history: list[tuple[str, str]] | None = None, k: int = 10
) -> AnswerResult:
    """Retrieve, then answer strictly from what was retrieved."""
    plan = plan_query(question, history)
    chunks = search(plan.standalone_question, company=plan.company, k=k)

    if not chunks:
        return AnswerResult(
            question=plan.standalone_question,
            found=False,
            answer="No indexed report covers that question.",
            citations=[],
            company=plan.company,
        )

    result: Answer = _model().with_structured_output(Answer).invoke(
        ANSWER_PROMPT.format(context=_format_context(chunks), question=plan.standalone_question)
    )

    citations = _resolve_citations(result, chunks)

    return AnswerResult(
        question=plan.standalone_question,
        found=result.found and bool(citations),
        answer=result.answer,
        citations=citations,
        company=plan.company,
    )
