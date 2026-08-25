.PHONY: install parse ingest extract serve dev test eval lint

install:                ## install backend and frontend dependencies
	uv venv --python 3.12 .venv && uv pip install -e ".[dev]"
	cd frontend && npm install

parse:                  ## parse the reports (metered; cached by content hash)
	.venv/bin/python scripts/parse_reports.py

ingest:                 ## chunk, embed and index the reports
	.venv/bin/python scripts/ingest.py

extract:                ## pre-extract FTE and sustainability goals into SQLite
	PYTHONPATH=backend .venv/bin/python backend/app/extraction.py

serve:                  ## run the API, serving the built frontend
	cd frontend && npm run build
	PYTHONPATH=backend .venv/bin/python -m uvicorn app.api:app --port 8000

dev:                    ## API plus Vue dev server with hot reload
	PYTHONPATH=backend .venv/bin/python -m uvicorn app.api:app --reload --port 8000 & \
	cd frontend && npm run dev

test:                   ## unit tests
	.venv/bin/python -m pytest backend/tests -q

eval:                   ## retrieval and answer quality against the labelled set
	.venv/bin/python eval/validate_questions.py
	.venv/bin/python eval/retrieval_eval.py
	.venv/bin/python eval/answer_eval.py

lint:
	.venv/bin/ruff check backend eval scripts
