.PHONY: help dev dev-obs databaseschemas seed dev-reset purge-corpus test test-unit test-integration lint fix ruff typecheck mypy check \
        pull-models chaos-kill-redis chaos-kill-ollama chaos-kill-postgres generate-eval test-generated-eval deepeval deepeval-large

# ── Development ───────────────────────────────────────────────────────────────

dev:
	docker compose up --build

dev-obs:
	docker compose -f docker-compose.yml -f docker-compose.observability.yml up --build

# ── Database ──────────────────────────────────────────────────────────────────

databaseschemas:
	@uv run python -c "\
import asyncio, asyncpg, os, glob; \
from dotenv import load_dotenv; load_dotenv(); \
async def run(): \
    conn = await asyncpg.connect(os.environ['DATABASE_URL']); \
    [print(f'  Applying {f}...') or await conn.execute(open(f).read()) for f in sorted(glob.glob('schema/*.sql'))]; \
    await conn.close(); print('All schemas applied.') \
asyncio.run(run())"

# ── Seed / bootstrap ─────────────────────────────────────────────────────────

seed:
	@echo "Applying database schemas..."
	@$(MAKE) databaseschemas
	@echo "Seeding default tenant, corpus, and sample documents..."
	@echo "(progress logged to /tmp/rag-seed.log — 'tail -f /tmp/rag-seed.log' in another terminal)"
	uv run python scripts/seed.py --force 2>&1 | tee /tmp/rag-seed.log
	@echo "Seed complete. Run 'make test-unit' to verify."

dev-reset:
	@echo "WARNING: This will drop all data and re-seed."
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	psql "$$DATABASE_URL" -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
	@$(MAKE) databaseschemas seed

purge-corpus:
	@echo "Purging default corpus content (documents, chunks, entity index, semantic cache)..."
	uv run python scripts/purge.py
	@echo "Running full re-ingestion of default document folder..."
	uv run python scripts/seed.py --force
	@echo "Done. Use 'make test-integration' to verify."

# ── Testing ───────────────────────────────────────────────────────────────────

test:
	uv run pytest tests/ -v --tb=short

test-unit:
	uv run pytest tests/unit/ tests/retrieval/test_retrieval_metrics.py::TestMetricFunctions -v

test-integration:
	uv run pytest tests/integration/ tests/retrieval/ tests/ingestion/ tests/agent/ tests/api/ -v --tb=short

generate-eval:
	uv run python scripts/generate_eval_questions.py

test-generated-eval:
	uv run pytest tests/retrieval/test_retrieval_metrics.py::TestGeneratedEvalDataset -v --tb=short

deepeval:
	uv run python scripts/run_deepeval.py

deepeval-large:
	uv run python scripts/run_deepeval.py --judge llama3.1:70b

# ── Code quality ──────────────────────────────────────────────────────────────

lint:
	uv run ruff check knowledge/ tests/

fix:
	uv run ruff check --fix knowledge/ tests/
	uv run ruff format knowledge/ tests/

ruff: fix

typecheck:
	uv run mypy knowledge/

mypy: typecheck

check: fix typecheck

# ── Ollama models ─────────────────────────────────────────────────────────────

pull-models:
	bash scripts/pull_models.sh

# ── Chaos testing ─────────────────────────────────────────────────────────────

chaos-kill-redis:
	docker compose stop redis && sleep 60 && docker compose start redis

chaos-kill-ollama:
	docker compose stop ollama && sleep 120 && docker compose start ollama

chaos-kill-postgres:
	docker compose stop postgres && sleep 30 && docker compose start postgres

# ── Latency monitoring ────────────────────────────────────────────────────────
# Watch live latency from docker compose logs.
# Both ingestion and retrieval stage latencies are emitted as structlog JSON.

watch-latency-retrieval:
	docker compose logs -f api | grep --line-buffered '"stage"' | \
	  python3 -c "import sys,json; [print(f'{e[\"stage\"]:<20} {e[\"duration_ms\"]}ms') for line in sys.stdin for e in [json.loads(line)] if 'duration_ms' in e]"

watch-latency-ingestion:
	docker compose logs -f ingest-worker | grep --line-buffered '"stage"' | \
	  python3 -c "import sys,json; [print(f'{e[\"stage\"]:<20} {e[\"duration_ms\"]}ms') for line in sys.stdin for e in [json.loads(line)] if 'duration_ms' in e]"

watch-errors:
	docker compose logs -f 2>&1 | grep --line-buffered '"level":"ERROR"'

load-baseline:
	locust -f tests/load/locustfile.py \
	  --headless --users 1 --spawn-rate 1 --run-time 5m \
	  --host http://localhost:8000 \
	  --csv tests/load/results/baseline-$$(date +%F)

load-peak:
	locust -f tests/load/locustfile.py \
	  --headless --users 5 --spawn-rate 1 --run-time 30m \
	  --host http://localhost:8000 \
	  --csv tests/load/results/peak-$$(date +%F) \
	  --exit-code-on-error 1

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo "Development:    make dev | make dev-obs"
	@echo "Database:       make databaseschemas | make seed"
	@echo "Testing:        make test | make test-unit | make test-integration"
	@echo "Code quality:   make ruff | make mypy | make check (ruff+mypy)"
	@echo "Models:         make pull-models"
	@echo "Seed:           make seed | make dev-reset | make purge-corpus"
	@echo "Latency:        make watch-latency-retrieval | watch-latency-ingestion | watch-errors"
	@echo "Load:           make load-baseline | make load-peak"
	@echo "Eval:           make generate-eval | make test-generated-eval | make deepeval | make deepeval-large"
	@echo "Chaos:          make chaos-kill-redis | chaos-kill-ollama | chaos-kill-postgres"
