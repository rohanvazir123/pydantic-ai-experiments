# Root Makefile — delegates to rag/v2/
#
# All active development is in rag/v2/. Targets here are thin wrappers
# so you can run `make <target>` from the repo root without cd-ing first.
#
# Usage:
#   make install      — one-shot setup (runs rag/v2/INSTALL.sh)
#   make pull-models  — pull Ollama models
#   make start        — start API + frontend
#   make seed         — apply DB schemas + seed sample docs
#   make lint         — ruff check
#   make typecheck    — mypy
#   make test         — full unit test suite
#   make test-smoke   — smoke tests only (<2s, no services)

.DEFAULT_GOAL := help
.PHONY: install start pull-models seed lint typecheck test test-smoke \
        test-unit test-integration ruff check clean help \
        v1-lint v1-test

V2 := rag/v2

# ── Setup & Launch ────────────────────────────────────────────────────────────

install:
	cd $(V2) && bash INSTALL.sh

start:
	cd $(V2) && bash start.sh

pull-models:
	cd $(V2) && bash scripts/pull_models.sh

seed:
	cd $(V2) && make seed

# ── Code Quality ──────────────────────────────────────────────────────────────

lint:
	cd $(V2) && make lint

ruff: lint

typecheck:
	cd $(V2) && make typecheck

# Full pre-commit gate (same as CI)
check: lint typecheck test-unit

# ── Testing ───────────────────────────────────────────────────────────────────

test: test-unit

test-unit:
	cd $(V2) && make test-unit

test-smoke:
	cd $(V2) && uv run pytest tests/unit/test_smoke.py -v

test-integration:
	cd $(V2) && make test-integration

# ── Housekeeping ──────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true

# ── Legacy v1 (rag/ — not actively developed) ─────────────────────────────────

v1-lint:
	ruff check rag/

v1-test:
	pytest rag/tests/ -v

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  RAG v2 — root Makefile (delegates to rag/v2/)"
	@echo ""
	@echo "  Setup"
	@echo "    make install        One-shot install + launch (bash INSTALL.sh)"
	@echo "    make pull-models    Pull Ollama models (nomic + qwen2.5:0.5b + llama3.2:3b)"
	@echo "    make seed           Apply DB migrations + ingest sample docs"
	@echo "    make start          Start API (:8001) + frontend (:3000)"
	@echo ""
	@echo "  Quality  (run before every commit)"
	@echo "    make lint           ruff check knowledge/ tests/"
	@echo "    make typecheck      mypy knowledge/"
	@echo "    make check          lint + typecheck + test-unit"
	@echo ""
	@echo "  Tests"
	@echo "    make test           Unit tests (no services, ~15s)"
	@echo "    make test-smoke     Smoke tests only (<2s)"
	@echo "    make test-integration  Integration tests (needs postgres + redis)"
	@echo ""
	@echo "  All targets delegate to rag/v2/Makefile."
	@echo ""
