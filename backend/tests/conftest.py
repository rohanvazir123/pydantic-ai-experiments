"""Shared pytest fixtures for the knowledge test suite."""

import os
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio


# ---------------------------------------------------------------------------
# Environment: point tests at a test .env if DATABASE_URL is not set
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    test_env = {
        "DATABASE_URL": "postgresql://ragv2:test@localhost:5432/ragv2_test",
        "AGE_DATABASE_URL": "postgresql://age:test@localhost:5433/age_test",
        "REDIS_URL": "redis://localhost:6379/1",
        "LLM_PROVIDER": "ollama",
        "LLM_MODEL": "llama3.2:3b",
        "LLM_BASE_URL": "http://localhost:11434/v1",
        "LLM_API_KEY": "ollama",
        "EMBEDDING_PROVIDER": "ollama",
        "EMBEDDING_MODEL": "nomic-embed-text:latest",
        "EMBEDDING_BASE_URL": "http://localhost:11434/v1",
        "EMBEDDING_API_KEY": "ollama",
        "EMBEDDING_DIMENSION": "768",
        "JWT_ALGORITHM": "RS256",
        "JWT_PUBLIC_KEY_PATH": "tests/fixtures/test_public.pem",
        "ALERT_EMAIL": "test@example.com",
        "CORPUS_CONFIGS_JSON": "[]",
    }
    for key, value in test_env.items():
        os.environ.setdefault(key, value)
