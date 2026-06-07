.PHONY: lint format check fix clean install dev test run validate ingest \
        v2-lint v2-format v2-fix v2-typecheck v2-check

# Ruff linting and formatting (targets rag/ directory)
lint:
	ruff check rag/

format:
	ruff format rag/

check: lint
	ruff format --check rag/

fix:
	ruff check --fix rag/
	ruff format rag/

# Combined ruff command (lint + format)
ruff: fix

# Installation
install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

# RAG commands
run:
	python -m rag.main

validate:
	python -m rag.main --validate

ingest:
	python -m rag.main --ingest

ingest-no-clean:
	python -m rag.main --ingest --no-clean

# Testing
test:
	pytest tests/ -v

# Clean
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

# ── RAG v2 (backend/knowledge/) ──────────────────────────────────────────────

v2-lint:
	ruff check backend/knowledge/

v2-format:
	ruff format backend/knowledge/

v2-fix:
	ruff check --fix backend/knowledge/
	ruff format backend/knowledge/

v2-typecheck:
	mypy backend/knowledge/

# Full quality gate — run this before committing v2 code
v2-check: v2-fix v2-typecheck

# ── Help ─────────────────────────────────────────────────────────────────────

# Help
help:
	@echo "Available targets:"
	@echo "  lint          - Run ruff linter"
	@echo "  format        - Run ruff formatter"
	@echo "  check         - Check linting and formatting (no changes)"
	@echo "  fix           - Fix linting issues and format code"
	@echo "  ruff          - Alias for fix (lint + format)"
	@echo "  install       - Install dependencies"
	@echo "  dev           - Install dev dependencies"
	@echo "  run           - Run RAG main (validate + ingest)"
	@echo "  validate      - Validate configuration only"
	@echo "  ingest        - Run document ingestion only"
	@echo "  ingest-no-clean - Ingest without cleaning existing data"
	@echo "  test          - Run tests"
	@echo "  clean         - Remove cache files"
	@echo ""
	@echo "RAG v2 (backend/knowledge/):"
	@echo "  v2-lint       - Ruff lint check"
	@echo "  v2-format     - Ruff format"
	@echo "  v2-fix        - Ruff fix + format"
	@echo "  v2-typecheck  - mypy strict type check"
	@echo "  v2-check      - Full quality gate (fix + typecheck)"
