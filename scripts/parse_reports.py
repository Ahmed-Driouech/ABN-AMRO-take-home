#!/usr/bin/env python
"""Parse every report and audit the figures that came out.

Two cached steps: Unstructured partitioning (metered, keyed by content hash)
and the per-table markdown rewrite. Re-runs are free.

The ``unmatched`` column is the one that matters. It counts figures in the
rendered tables that do not appear anywhere on their source page -- the
measure of whether the rewrite transcribed faithfully or invented something.
It is a measurement only; nothing is edited on the basis of it.

    python scripts/parse_reports.py
    python scripts/parse_reports.py --refresh
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]

from app.config import BUNDLED_REPORTS, settings  # noqa: E402
from app.parsing import load_report  # noqa: E402

from eval.parser_audit import audit_numbers, page_figures  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="re-parse even if cached")
    ap.add_argument("--no-audit", action="store_true", help="skip the figure audit")
    args = ap.parse_args()

    print(
        f"{'report':<20}{'chunks':>8}{'tables':>8}{'figures':>9}"
        f"{'unmatched':>11}{'ok':>8}{'time':>7}"
    )
    totals = [0, 0, 0]

    for filename, company, year in BUNDLED_REPORTS:
        path = settings.reports_dir / filename
        if not path.exists():
            print(f"{filename:<20}  missing", file=sys.stderr)
            continue

        started = time.time()
        docs = load_report(path, company=company, year=year, refresh=args.refresh)
        tables = [d for d in docs if d.metadata["content_type"] == "table"]

        figures = unmatched = 0
        if not args.no_audit:
            figures, unmatched = audit_numbers(docs, page_figures(path))

        totals[0] += len(tables)
        totals[1] += figures
        totals[2] += unmatched
        rate = f"{100 * (1 - unmatched / figures):.2f}%" if figures else "-"
        print(
            f"{filename:<20}{len(docs):>8}{len(tables):>8}{figures or '-':>9}"
            f"{unmatched if figures else '-':>11}{rate:>8}{time.time() - started:>6.0f}s",
            flush=True,
        )

    tables, figures, unmatched = totals
    if not figures:
        print(f"\n{tables} tables | audit skipped")
        return 0
    rate = 100 * (1 - unmatched / figures)
    print(
        f"\n{tables} tables | {figures} figures | {unmatched} unmatched "
        f"| {rate:.2f}% accounted for"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
