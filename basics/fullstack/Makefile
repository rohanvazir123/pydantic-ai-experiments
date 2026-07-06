.PHONY: install test lint type check up down run-api run-worker

install:            ## sync deps into .venv (Python 3.13)
	uv sync

test:               ## run the full test suite
	uv run pytest

lint:               ## ruff lint
	uv run ruff check app tests

type:               ## mypy type-check
	uv run mypy app

check: lint type test  ## the full quality gate

up:                 ## start Postgres + Temporal dev server
	docker compose up -d

down:               ## stop services
	docker compose down

run-worker:         ## run the Temporal worker (needs `up`)
	uv run python -m app.main worker

run-api:            ## run the FastAPI app, dev/single-process (needs `up` + worker)
	uv run python -m app.main api

run-api-prod:       ## run the API under gunicorn + uvicorn workers (production)
	uv run gunicorn app.api.asgi:app -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8000
