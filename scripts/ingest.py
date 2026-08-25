#!/usr/bin/env python
"""Index every report so the application has data on first run.

Ingestion is idempotent: identity is the PDF's content hash, so re-running
this is free and a report already indexed is skipped.

    python scripts/ingest.py
    python scripts/ingest.py --force     # re-index everything
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]

from app.config import BUNDLED_REPORTS, settings  # noqa: E402
from app.store import ingest_report, ingested_documents  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-index even if present")
    args = ap.parse_args()

    print(f"{'report':<20}{'status':>10}{'chunks':>9}{'time':>8}")
    for filename, company, year in BUNDLED_REPORTS:
        path = settings.reports_dir / filename
        if not path.exists():
            print(f"{filename:<20}{'missing':>10}", file=sys.stderr)
            continue
        started = time.time()
        result = ingest_report(str(path), company, year, force=args.force)
        print(
            f"{filename:<20}{result['status']:>10}{result['chunks']:>9}"
            f"{time.time() - started:>7.0f}s",
            flush=True,
        )

    total = ingested_documents()
    chunks = sum(d["chunks"] for d in total.values())
    print(f"\n{len(total)} documents indexed | {chunks} chunks in {settings.chroma_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
