"""HTTP API.

Thin on purpose: every decision lives in the modules behind it, so the
frontend holds no intelligence and the same behaviour is reachable from the
tests and the eval scripts.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.answer import AnswerResult, answer_question, available_companies
from app.config import settings
from app.extraction import extract_for, load_datapoints, save
from app.store import ingest_report, ingested_documents

app = FastAPI(title="Annual Report RAG", version="1.0")

# Only needed when the Vue dev server runs separately; the built frontend is
# served from this same origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    history: list[tuple[str, str]] = []


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "companies": available_companies()}


@app.get("/api/documents")
def documents() -> list[dict]:
    indexed = sorted(ingested_documents().items(), key=lambda kv: kv[1]["company"] or "")
    return [{"doc_id": doc_id, **info} for doc_id, info in indexed]


@app.post("/api/ask", response_model=AnswerResult)
def ask(request: AskRequest) -> AnswerResult:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    return answer_question(request.question, history=[tuple(h) for h in request.history])


@app.get("/api/datapoints")
def datapoints(company: str | None = None) -> list[dict]:
    """The pre-extracted FTE counts and sustainability goals, from SQLite."""
    return load_datapoints(company)


@app.post("/api/documents")
async def upload(file: UploadFile, company: str, year: int) -> dict:
    """Index a new report.

    Company and year are supplied by the caller rather than guessed from the
    page, so a figure can never be attributed to the wrong issuer.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="a PDF is required")
    destination = settings.reports_dir / file.filename
    destination.write_bytes(await file.read())

    result = ingest_report(str(destination), company, year)
    if result["status"] == "indexed":
        save(extract_for(company, year))
    return result


# The built frontend is served from the API so the whole application is one
# process and one port.
FRONTEND = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if FRONTEND.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND / "assets"), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(FRONTEND / "index.html")
