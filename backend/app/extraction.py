"""Pre-extraction: structured datapoints pulled at ingest time.

The brief requires each report to be pre-processed on ingestion for at least
the FTE count and the sustainability goals, and for that data to be visible in
the application. So this runs once per document and writes to SQLite -- it is
not a query-time lookup dressed up as one.

The approach is retrieve-then-extract rather than whole-document-to-LLM: a
targeted search pulls the handful of chunks likely to hold the datapoint, and
only those go to the model. A 500-page report will not fit in a context
window, and sending it would cost far more for a worse result.

Every datapoint records the page it came from, so a reader can check it. And
"not found" is a valid outcome: a missing figure is recorded as missing rather
than invented.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config import settings
from app.store import search

SCHEMA = """
CREATE TABLE IF NOT EXISTS datapoints (
    company     TEXT NOT NULL,
    year        INTEGER NOT NULL,
    kind        TEXT NOT NULL,
    label       TEXT,
    value       TEXT,
    unit        TEXT,
    target_year INTEGER,
    page        INTEGER,
    PRIMARY KEY (company, year, kind, label)
);
"""

# Wider than a chat query: recall matters more than precision here, because a
# datapoint missed at ingest is simply absent from the application.
EXTRACT_K = 12

FTE_QUERY = "total number of employees full-time equivalent FTE average headcount"
GOALS_QUERY = (
    "sustainability targets climate goals net-zero emissions reduction target by 2030 2050"
)


class FTE(BaseModel):
    found: bool = Field(description="False if the excerpts do not state a total employee count.")
    value: str = Field(
        description="The figure exactly as printed, e.g. '18,725'. Empty if not found."
    )
    unit: str = Field(description="What is counted, e.g. 'FTE' or 'headcount'. Empty if not found.")
    excerpt: int = Field(description="Number of the excerpt the figure came from, 0 if not found.")


class Goal(BaseModel):
    label: str = Field(description="Short name of the goal, e.g. 'Scope 1 and 2 emissions'.")
    value: str = Field(description="The target exactly as stated, e.g. 'reduce by 50%'.")
    target_year: int = Field(description="Year the target is set for, 0 if unstated.")
    excerpt: int = Field(description="Number of the excerpt this came from.")


class Goals(BaseModel):
    goals: list[Goal] = Field(
        description="Sustainability goals stated in the excerpts. May be empty."
    )


PROMPT = """\
Extract from these excerpts of {company}'s {year} annual report.

Rules:
- Use only the excerpts. Quote figures exactly as printed.
- If the excerpts do not contain it, say so rather than estimating.
- Cite the excerpt number you took each item from.

{task}

Excerpts:
{context}
"""


def connect() -> sqlite3.Connection:
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.sqlite_path)
    conn.executescript(SCHEMA)
    return conn


def _context(chunks) -> str:
    return "\n\n---\n\n".join(
        f"[{i}] p. {d.metadata.get('page')}\n{d.page_content}" for i, d in enumerate(chunks, 1)
    )


def _page_of(chunks, excerpt: int) -> int | None:
    if 1 <= excerpt <= len(chunks):
        return chunks[excerpt - 1].metadata.get("page")
    return None


def extract_for(company: str, year: int) -> list[dict]:
    """Pull FTE and sustainability goals for one company."""
    model = ChatOpenAI(model=settings.chat_model, temperature=0, api_key=settings.openai_api_key)
    rows: list[dict] = []

    fte_chunks = search(FTE_QUERY, company=company, k=EXTRACT_K)
    fte: FTE = model.with_structured_output(FTE).invoke(
        PROMPT.format(
            company=company,
            year=year,
            task="Find the total number of employees.",
            context=_context(fte_chunks),
        )
    )
    rows.append(
        {
            "company": company,
            "year": year,
            "kind": "fte",
            "label": "Total employees",
            "value": fte.value if fte.found else None,
            "unit": fte.unit if fte.found else None,
            "target_year": None,
            "page": _page_of(fte_chunks, fte.excerpt) if fte.found else None,
        }
    )

    goal_chunks = search(GOALS_QUERY, company=company, k=EXTRACT_K)
    goals: Goals = model.with_structured_output(Goals).invoke(
        PROMPT.format(
            company=company,
            year=year,
            task="Find the stated sustainability and climate goals.",
            context=_context(goal_chunks),
        )
    )
    for goal in goals.goals:
        rows.append(
            {
                "company": company,
                "year": year,
                "kind": "sustainability_goal",
                "label": goal.label,
                "value": goal.value,
                "unit": None,
                "target_year": goal.target_year or None,
                "page": _page_of(goal_chunks, goal.excerpt),
            }
        )
    return rows


def save(rows: list[dict]) -> None:
    with closing(connect()) as conn, conn:
        conn.executemany(
            "INSERT OR REPLACE INTO datapoints"
            " (company, year, kind, label, value, unit, target_year, page)"
            " VALUES (:company, :year, :kind, :label, :value, :unit, :target_year, :page)",
            rows,
        )


def load_datapoints(company: str | None = None) -> list[dict]:
    with closing(connect()) as conn:
        conn.row_factory = sqlite3.Row
        sql = "SELECT * FROM datapoints"
        params: tuple = ()
        if company:
            sql += " WHERE company = ?"
            params = (company,)
        sql += " ORDER BY company, kind DESC, label"
        return [dict(r) for r in conn.execute(sql, params)]


if __name__ == "__main__":  # pragma: no cover
    from app.config import BUNDLED_REPORTS

    for _, company, year in BUNDLED_REPORTS:
        extracted = extract_for(company, year)
        save(extracted)
        print(f"{company:<10} {len(extracted)} datapoints", flush=True)
    print(json.dumps(load_datapoints()[:3], indent=1))
