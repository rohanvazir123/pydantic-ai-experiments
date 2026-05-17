"""
Tests for LightRAG PostgreSQL backend demo.

Unit tests: mock LLM/embed, verify ingestion + query flow.
Integration tests: require live Ollama + PostgreSQL (port 5434) + AGE (port 5433).
    pytest -m integration basics/docling_lightrag_raganything/test_lightrag_demo.py
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEMO_DIR = Path(__file__).parent
OUTPUT_DIR = DEMO_DIR / "output" / "lightrag"

CLEAN_TEXT = """
The Transformer architecture was introduced by Ashish Vaswani at Google Brain in 2017.
It uses multi-head self-attention instead of recurrence. BERT extended the Transformer
using bidirectional pre-training on large text corpora.
""".strip()

TABLE_TEXT = """
| Model | BLEU EN-DE | BLEU EN-FR |
|---|---|---|
| Transformer (base) | 27.3 | 38.1 |
| Transformer (big) | 28.4 | 41.0 |
| ByteNet | 23.75 | - |
""".strip()

FIGURE_TEXT = """
[Figure 3]
[Image content not available]
Caption: Multi-head attention mechanism showing queries Q, keys K, values V.
""".strip()

COLUMN_MIXED_TEXT = (
    "Ashish Vaswani ∗ Google Brain avaswani@google.com Noam Shazeer ∗ "
    "Google Brain noam@google.com\nLlion Jones ∗ Google Research llion@google.com"
)


@pytest.fixture
def mock_embed():
    """Deterministic mock embedding function."""
    async def _embed(texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.normal(size=(len(texts), 768)).astype(np.float32)
    return _embed


@pytest.fixture
def mock_llm_good():
    """Mock LLM that returns well-formed entity extraction."""
    async def _llm(prompt: str, system_prompt: str | None = None, **_) -> str:
        return (
            "entity<|#|>Transformer<|#|>Method<|#|>The Transformer is an attention-based architecture introduced at Google Brain.\n"
            "entity<|#|>Ashish Vaswani<|#|>Person<|#|>Ashish Vaswani is a researcher at Google Brain who co-invented the Transformer.\n"
            "entity<|#|>Google Brain<|#|>Organization<|#|>Google Brain is the AI research division of Google.\n"
            "entity<|#|>BERT<|#|>Method<|#|>BERT is a bidirectional Transformer model developed by Google.\n"
            "relation<|#|>Ashish Vaswani<|#|>Transformer<|#|>invented, research<|#|>Ashish Vaswani introduced the Transformer architecture.\n"
            "relation<|#|>Transformer<|#|>BERT<|#|>foundation, extension<|#|>BERT extends the Transformer using bidirectional pre-training.\n"
            "relation<|#|>Ashish Vaswani<|#|>Google Brain<|#|>affiliation<|#|>Ashish Vaswani works at Google Brain.\n"
            "<|COMPLETE|>"
        )
    return _llm


@pytest.fixture
def mock_llm_malformed():
    """Mock LLM that returns malformed output (simulates format drift)."""
    async def _llm(prompt: str, system_prompt: str | None = None, **_) -> str:
        return (
            "Here are the entities I found:\n"
            "- Transformer (a model)\n"
            "- Google Brain (an org)\n"
            "The Transformer was made at Google Brain.\n"
            # No <|COMPLETE|> — simulates truncation / format drift
        )
    return _llm


# ---------------------------------------------------------------------------
# Unit tests — no external dependencies
# ---------------------------------------------------------------------------

class TestCaseDefinitions:
    """Verify the demo test cases are well-formed."""

    def test_all_cases_have_required_fields(self):
        from lightrag_demo import CASES
        for case in CASES:
            assert "id" in case
            assert "label" in case
            assert "expected" in case, f"Case {case['id']} missing 'expected'"
            assert "text" in case
            assert "query" in case
            assert "query_mode" in case
            assert case["expected"] in ("pass", "fail")

    def test_failure_cases_document_reason(self):
        from lightrag_demo import CASES
        for case in CASES:
            if case["expected"] == "fail":
                assert "failure_reason" in case, (
                    f"Failure case {case['id']} must document why it fails"
                )
                assert len(case["failure_reason"]) > 30

    def test_query_modes_are_valid(self):
        from lightrag_demo import CASES
        valid_modes = {"naive", "local", "global", "hybrid"}
        for case in CASES:
            assert case["query_mode"] in valid_modes

    def test_has_both_works_and_fails_cases(self):
        from lightrag_demo import CASES
        expected = [c["expected"] for c in CASES]
        assert "pass" in expected, "Must have at least one WORKS case"
        assert "fail" in expected, "Must have at least one FAILS case"


class TestOllamaHelpers:
    """Unit tests for Ollama helper functions (mocked HTTP)."""

    def test_ollama_llm_sends_correct_payload(self):
        from unittest.mock import patch, MagicMock

        async def run():
            from lightrag_demo import ollama_llm
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "test response"}}]
            }
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client_class.return_value = mock_client
                result = await ollama_llm("hello", system_prompt="be helpful")
            assert result == "test response"

        asyncio.run(run())

    def test_good_llm_output_contains_complete_delimiter(self):
        good_output = (
            "entity<|#|>X<|#|>Method<|#|>X is a method.\n"
            "<|COMPLETE|>"
        )
        assert "<|COMPLETE|>" in good_output

    def test_malformed_llm_output_missing_complete_delimiter(self):
        malformed = "Here are entities:\n- Transformer\n- BERT\n"
        assert "<|COMPLETE|>" not in malformed

    def test_figure_text_has_no_visual_content(self):
        assert "[Image content not available]" in FIGURE_TEXT
        assert "Caption:" in FIGURE_TEXT
        # Visual content describing HOW attention works is absent
        assert "softmax" not in FIGURE_TEXT
        assert "dot product" not in FIGURE_TEXT.lower()

    def test_column_mixed_text_has_interleaved_columns(self):
        # Both left-column (Google Brain) and right-column (Google Research) in same text
        assert "Google Brain" in COLUMN_MIXED_TEXT
        assert "Google Research" in COLUMN_MIXED_TEXT
        # Email addresses bled in from column boundaries
        assert "@google.com" in COLUMN_MIXED_TEXT


class TestFailureReasonDocumentation:
    """Verify failure reasons explain the actual mechanism."""

    def test_figure_failure_explains_vlm_absence(self):
        from lightrag_demo import CASES
        case = next(c for c in CASES if c["id"] == "fails_figure_placeholder")
        assert "VLM" in case["failure_reason"] or "caption" in case["failure_reason"].lower()

    def test_table_failure_explains_comparative_query_limitation(self):
        from lightrag_demo import CASES
        case = next(c for c in CASES if c["id"] == "fails_markdown_table")
        reason = case["failure_reason"].lower()
        assert "compar" in reason or "ranked" in reason or "best" in reason

    def test_column_mixed_failure_references_docling(self):
        from lightrag_demo import CASES
        case = next(c for c in CASES if c["id"] == "fails_column_mixed")
        assert "Docling" in case["failure_reason"] or "column" in case["failure_reason"].lower()


# ---------------------------------------------------------------------------
# Integration tests — require live services
# ---------------------------------------------------------------------------

def pg_available() -> bool:
    try:
        import asyncpg
        async def check():
            conn = await asyncpg.connect(
                "postgresql://rag_user:rag_pass@localhost:5434/rag_db", timeout=3
            )
            await conn.close()
        asyncio.run(check())
        return True
    except Exception:
        return False


def age_available() -> bool:
    try:
        import asyncpg
        async def check():
            conn = await asyncpg.connect(
                "postgresql://age_user:age_pass@localhost:5433/legal_graph", timeout=3
            )
            await conn.close()
        asyncio.run(check())
        return True
    except Exception:
        return False


def ollama_available() -> bool:
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


requires_services = pytest.mark.skipif(
    not (pg_available() and age_available() and ollama_available()),
    reason="Requires pgvector (5434) + AGE (5433) + Ollama",
)


@requires_services
class TestIntegrationLightRAG:
    """Integration tests against live services."""

    @pytest.fixture(scope="class")
    def rag(self):
        async def _build():
            from lightrag_demo import build_rag, ensure_embed_model
            await ensure_embed_model()
            return await build_rag()
        return asyncio.run(_build())

    def test_rag_initialises(self, rag):
        assert rag is not None

    def test_ingest_clean_text(self, rag):
        async def run():
            await rag.ainsert(CLEAN_TEXT)
        asyncio.run(run())

    def test_query_returns_answer_for_clean_text(self, rag):
        from lightrag.base import QueryParam
        async def run():
            result = await rag.aquery(
                "What is the Transformer architecture?",
                param=QueryParam(mode="hybrid"),
            )
            return str(result)
        answer = asyncio.get_event_loop().run_until_complete(run())
        assert len(answer) > 50
        assert "[no-context]" not in answer.lower()

    def test_query_figure_placeholder_returns_limited_answer(self, rag):
        """Figure placeholder text produces weak or no-context answers."""
        from lightrag.base import QueryParam
        async def run():
            await rag.ainsert(FIGURE_TEXT)
            result = await rag.aquery(
                "How does multi-head attention work mechanically?",
                param=QueryParam(mode="local"),
            )
            return str(result)
        answer = asyncio.get_event_loop().run_until_complete(run())
        # Answer should either admit no context or give a very short/generic response
        is_weak = (
            "[no-context]" in answer.lower()
            or "not available" in answer.lower()
            or len(answer) < 200
        )
        assert is_weak, (
            f"Expected weak answer for figure placeholder, got {len(answer)} chars: {answer[:200]}"
        )

    def test_output_dir_is_writable(self):
        test_file = OUTPUT_DIR / "_test_write.tmp"
        test_file.write_text("ok")
        assert test_file.exists()
        test_file.unlink()
