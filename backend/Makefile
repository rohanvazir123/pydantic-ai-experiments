.PHONY: help dev dev-obs migrate test test-unit lint fix typecheck check \
        pull-models chaos-kill-redis chaos-kill-ollama chaos-kill-postgres

# ── Development ───────────────────────────────────────────────────────────────

dev:
	docker compose up --build

dev-obs:
	docker compose -f docker-compose.yml -f docker-compose.observability.yml up --build

# ── Database ──────────────────────────────────────────────────────────────────

migrate:
	@for f in migrations/*.sql; do \
		echo "Running $$f ..."; \
		psql "$$DATABASE_URL" -f "$$f"; \
	done
	@echo "All migrations complete."

# ── Testing ───────────────────────────────────────────────────────────────────

test:
	uv run pytest tests/ -v --tb=short

test-unit:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/integration/ tests/retrieval/ tests/ingestion/ tests/agent/ tests/api/ -v --tb=short

# ── Code quality ──────────────────────────────────────────────────────────────

lint:
	uv run ruff check knowledge/ tests/

fix:
	uv run ruff check --fix knowledge/ tests/
	uv run ruff format knowledge/ tests/

typecheck:
	uv run mypy knowledge/

check: fix typecheck

# ── Ollama models ─────────────────────────────────────────────────────────────

pull-models:
	ollama pull llama3.2:3b
	ollama pull nomic-embed-text:latest
	ollama pull qwen2.5:0.5b
	ollama pull llama3.1:70b

# ── Chaos testing ─────────────────────────────────────────────────────────────

chaos-kill-redis:
	docker compose stop redis && sleep 60 && docker compose start redis

chaos-kill-ollama:
	docker compose stop ollama && sleep 120 && docker compose start ollama

chaos-kill-postgres:
	docker compose stop postgres && sleep 30 && docker compose start postgres

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo "Development:    make dev | make dev-obs"
	@echo "Database:       make migrate"
	@echo "Testing:        make test | make test-unit | make test-integration"
	@echo "Code quality:   make fix | make typecheck | make check"
	@echo "Models:         make pull-models"
	@echo "Chaos:          make chaos-kill-redis | chaos-kill-ollama | chaos-kill-postgres"
