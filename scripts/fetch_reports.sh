#!/usr/bin/env bash
# Download the five annual reports from the issuers' own sites.
# They are ~134 MB in total and are deliberately not committed.
set -euo pipefail
cd "$(dirname "$0")/../data/reports"

fetch() {
  [ -f "$1" ] && { echo "have $1"; return; }
  echo "fetching $1"
  curl -sSL -A "Mozilla/5.0" -o "$1" "$2"
}

fetch shell-2024.pdf    "https://www.shell.com/investors/results-and-reporting/annual-report-archive/_jcr_content/root/main/section_812377294/tabs/tab_copy/text.multi.stream/1752580693041/6c20b8111738b9a590ba145f0d1c4fa0e530dae0/shell-annual-report-2024.pdf"
fetch abnamro-2024.pdf  "https://www.banktrack.org/download/integrated_annual_report_2024/250311_abn_amro___integrated_annual_report_2024.pdf"
fetch asml-2024.pdf     "https://ourbrand.asml.com/m/3035813cf1b8ea4f/original/2024-Annual-Report-based-on-IFRS-FINAL.pdf"
fetch heineken-2024.pdf "https://www.theheinekencompany.com/sites/heineken-corp/files/2025-02/heineken_n_v_annual_report_2024_final_20feb2025.pdf"
fetch cmcom-2024.pdf    "https://www.cm.com/cdn/web/en/file/investor-relations/annual-report-2024.pdf"

echo "done:"; ls -la
