"""Runtime configuration. Everything secret comes from the environment."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # --- Unstructured (PDF parsing) -------------------------------------
    unstructured_api_key: str = ""
    unstructured_api_url: str = "https://api.unstructuredapp.io/general/v0/general"
    unstructured_strategy: str = "hi_res"
    # Client-side page splitting: the only practical way to get a 350-page
    # annual report through the API in reasonable wall-clock time.
    unstructured_concurrency: int = 15

    # --- OpenAI (embeddings + generation) -------------------------------
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    table_rewrite_workers: int = 8

    # Chunking. Large enough to hold a complete argument, small enough that a
    # retrieved chunk is still specific.
    chunk_size: int = 800
    chunk_overlap: int = 100
    # Tables are coherent units and splitting one strands rows from the header
    # that gives them meaning, so they get a larger budget than prose.
    table_max_tokens: int = 2000
    # Bounds a runaway generation: one table came back with a markdown
    # separator row a million characters long.
    table_rewrite_max_tokens: int = 4000

    # --- Local paths ----------------------------------------------------
    data_dir: Path = ROOT / "data"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def parse_cache_dir(self) -> Path:
        """Raw Unstructured responses, keyed by PDF content hash.

        Parsing is the slowest and the only metered step in ingestion, so its
        output is cached on disk. Re-ingesting is then free, and a demo still
        works if the API is unreachable.
        """
        return self.data_dir / "parse_cache"

    @property
    def table_cache_dir(self) -> Path:
        """LLM-rendered markdown per table, keyed by document hash: the rewrite
        is paid once and the result is a fixed artefact from then on."""
        return self.data_dir / "table_cache"

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"


# Reports pre-loaded so the application is not empty on first run. Company and
# year are asserted here and never inferred from a page: a figure attributed to
# the wrong issuer is the one error a bank cannot tolerate.
BUNDLED_REPORTS = [
    ("shell-2024.pdf", "Shell", 2024),
    ("abnamro-2024.pdf", "ABN AMRO", 2024),
    ("asml-2024.pdf", "ASML", 2024),
    ("heineken-2024.pdf", "Heineken", 2024),
    ("cmcom-2024.pdf", "CM.com", 2024),
]

settings = Settings()
