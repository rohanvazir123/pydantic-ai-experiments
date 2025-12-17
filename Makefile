.PHONY: lint format check fix clean install dev test run validate ingest

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
