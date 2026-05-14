"""Tests for the MCP server (rag.mcp.server).

All external dependencies (DB, LLM, ingestion pipeline) are mocked so that
these tests run without any live services.

Because FastMCP's @mcp.tool() returns the original function unchanged, the
tool functions can be imported and called directly as plain async functions.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag.mcp.server import health, ingest, retrieve, search


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_result(text: str) -> MagicMock:
    """Minimal mock that looks like a Pydantic AI RunResult."""
    result = MagicMock()
    result.output = text
    return result


def _make_pipeline(results: list) -> MagicMock:
    """Mock ingestion pipeline."""
    pipeline = MagicMock()
    pipeline.initialize = AsyncMock()
    pipeline.ingest_documents = AsyncMock(return_value=results)
    pipeline.close = AsyncMock()
    return pipeline


def _make_ingest_result(
    title: str = "Doc",
    chunks: int = 5,
    errors: list[str] | None = None,
) -> MagicMock:
    r = MagicMock()
    r.title = title
    r.chunks_created = chunks
    r.errors = errors or []
    return r


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------


class TestSearch:
    """Tests for the search tool."""

    @pytest.mark.asyncio
    async def test_returns_answer_string(self):
        """Happy path: returns str(result.output) from traced_agent_run."""
        mock_result = _make_agent_result("NeuralFlow AI builds RAG systems.")

        with patch("rag.mcp.server.traced_agent_run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            answer = await search("What does NeuralFlow do?")

        assert answer == "NeuralFlow AI builds RAG systems."

    @pytest.mark.asyncio
    async def test_forwards_user_id_and_session_id(self):
        """user_id and session_id are passed through to traced_agent_run."""
        with patch("rag.mcp.server.traced_agent_run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = _make_agent_result("42 engineers.")
            await search("How many engineers?", user_id="alice", session_id="sess-1")

        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["query"] == "How many engineers?"
        assert call_kwargs["user_id"] == "alice"
        assert call_kwargs["session_id"] == "sess-1"

    @pytest.mark.asyncio
    async def test_default_optional_args_are_none(self):
        """user_id and session_id default to None."""
        with patch("rag.mcp.server.traced_agent_run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = _make_agent_result("ok")
            await search("query")

        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["user_id"] is None
        assert call_kwargs["session_id"] is None

    @pytest.mark.asyncio
    async def test_propagates_exception(self):
        """If traced_agent_run raises, the exception propagates to the MCP layer."""
        with patch("rag.mcp.server.traced_agent_run", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = RuntimeError("LLM timeout")
            with pytest.raises(RuntimeError, match="LLM timeout"):
                await search("crash")

    @pytest.mark.asyncio
    async def test_output_coerced_to_string(self):
        """Non-string output is coerced via str()."""
        mock_result = MagicMock()
        mock_result.output = 12345  # integer output

        with patch("rag.mcp.server.traced_agent_run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            answer = await search("numeric?")

        assert answer == "12345"


# ---------------------------------------------------------------------------
# retrieve()
# ---------------------------------------------------------------------------


class TestRetrieve:
    """Tests for the retrieve tool."""

    def _mock_rag_state(self, context: str) -> MagicMock:
        """Build a RAGState mock whose retriever returns *context*."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve_as_context = AsyncMock(return_value=context)

        mock_state = MagicMock()
        mock_state.get_retriever = AsyncMock(return_value=mock_retriever)
        mock_state.close = AsyncMock()
        return mock_state

    @pytest.mark.asyncio
    async def test_returns_context_string(self):
        """Happy path: returns formatted context from the retriever."""
        mock_state = self._mock_rag_state("Source 1: NeuralFlow builds AI tools.")

        with patch("rag.mcp.server.RAGState", return_value=mock_state):
            result = await retrieve("What does NeuralFlow do?")

        assert result == "Source 1: NeuralFlow builds AI tools."

    @pytest.mark.asyncio
    async def test_empty_context_returns_no_results(self):
        """Empty string from retriever → 'No results found.'"""
        mock_state = self._mock_rag_state("")

        with patch("rag.mcp.server.RAGState", return_value=mock_state):
            result = await retrieve("unknown query")

        assert result == "No results found."

    @pytest.mark.asyncio
    async def test_none_context_returns_no_results(self):
        """None from retriever → 'No results found.'"""
        mock_state = self._mock_rag_state(None)

        with patch("rag.mcp.server.RAGState", return_value=mock_state):
            result = await retrieve("unknown query")

        assert result == "No results found."

    @pytest.mark.asyncio
    async def test_forwards_match_count_and_search_type(self):
        """match_count and search_type are passed to retrieve_as_context."""
        mock_state = self._mock_rag_state("chunks")

        with patch("rag.mcp.server.RAGState", return_value=mock_state):
            await retrieve("PTO policy", match_count=3, search_type="text")

        mock_state.get_retriever.return_value.retrieve_as_context.assert_awaited_once_with(
            query="PTO policy",
            match_count=3,
            search_type="text",
        )

    @pytest.mark.asyncio
    async def test_close_called_on_success(self):
        """state.close() is always called after a successful retrieval."""
        mock_state = self._mock_rag_state("data")

        with patch("rag.mcp.server.RAGState", return_value=mock_state):
            await retrieve("query")

        mock_state.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_called_on_error(self):
        """state.close() is called even when retrieve_as_context raises."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve_as_context = AsyncMock(side_effect=RuntimeError("db error"))

        mock_state = MagicMock()
        mock_state.get_retriever = AsyncMock(return_value=mock_retriever)
        mock_state.close = AsyncMock()

        with patch("rag.mcp.server.RAGState", return_value=mock_state):
            with pytest.raises(RuntimeError, match="db error"):
                await retrieve("query")

        mock_state.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# ingest()
# ---------------------------------------------------------------------------


class TestIngest:
    """Tests for the ingest tool."""

    @pytest.mark.asyncio
    async def test_returns_summary_string(self):
        """Happy path: returns document count and chunk totals."""
        results = [
            _make_ingest_result("Doc A", chunks=3),
            _make_ingest_result("Doc B", chunks=7),
        ]
        pipeline = _make_pipeline(results)

        with patch("rag.mcp.server.create_pipeline", return_value=pipeline):
            output = await ingest()

        assert "2 document(s)" in output
        assert "10 chunk(s)" in output
        assert "Doc A" in output
        assert "Doc B" in output

    @pytest.mark.asyncio
    async def test_forwards_parameters_to_create_pipeline(self):
        """All ingest parameters are passed to create_pipeline."""
        pipeline = _make_pipeline([_make_ingest_result()])

        with patch("rag.mcp.server.create_pipeline", return_value=pipeline) as mock_create:
            await ingest(
                documents_folder="custom/path",
                clean=False,
                chunk_size=500,
                max_tokens=128,
            )

        mock_create.assert_called_once_with(
            documents_folder="custom/path",
            clean=False,
            chunk_size=500,
            max_tokens=128,
        )

    @pytest.mark.asyncio
    async def test_errors_included_in_output(self):
        """Ingestion errors for a document appear in the output string."""
        results = [_make_ingest_result("Bad Doc", chunks=0, errors=["parse failed"])]
        pipeline = _make_pipeline(results)

        with patch("rag.mcp.server.create_pipeline", return_value=pipeline):
            output = await ingest()

        assert "error" in output.lower()
        assert "parse failed" in output

    @pytest.mark.asyncio
    async def test_pipeline_close_called_on_success(self):
        """pipeline.close() is called after successful ingestion."""
        pipeline = _make_pipeline([_make_ingest_result()])

        with patch("rag.mcp.server.create_pipeline", return_value=pipeline):
            await ingest()

        pipeline.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_pipeline_close_called_on_error(self):
        """pipeline.close() is called even when ingest_documents raises."""
        pipeline = _make_pipeline([])
        pipeline.ingest_documents.side_effect = RuntimeError("disk full")

        with patch("rag.mcp.server.create_pipeline", return_value=pipeline):
            with pytest.raises(RuntimeError, match="disk full"):
                await ingest()

        pipeline.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# health()
# ---------------------------------------------------------------------------


class TestHealth:
    """Tests for the health tool."""

    def _mock_store(self, fail: bool = False) -> MagicMock:
        store = MagicMock()
        if fail:
            store.initialize = AsyncMock(side_effect=ConnectionRefusedError("no db"))
        else:
            store.initialize = AsyncMock()
        store.close = AsyncMock()
        return store

    def _mock_http(self, status_code: int = 200, fail: bool = False) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.status_code = status_code

        mock_client = AsyncMock()
        if fail:
            mock_client.get = AsyncMock(side_effect=ConnectionRefusedError("no api"))
        else:
            mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        return mock_client

    @pytest.mark.asyncio
    async def test_all_healthy_returns_ok(self):
        """All components up → 'status: ok' with all ticks."""
        store = self._mock_store()
        http = self._mock_http(200)

        with patch("rag.mcp.server.PostgresHybridStore", return_value=store):
            with patch("httpx.AsyncClient", return_value=http):
                result = await health()

        assert "status: ok" in result
        assert result.count("✓") == 3

    @pytest.mark.asyncio
    async def test_db_down_returns_degraded(self):
        """DB failure → 'status: degraded', db shows ✗."""
        store = self._mock_store(fail=True)
        http = self._mock_http(200)

        with patch("rag.mcp.server.PostgresHybridStore", return_value=store):
            with patch("httpx.AsyncClient", return_value=http):
                result = await health()

        assert "degraded" in result
        assert "db: ✗" in result

    @pytest.mark.asyncio
    async def test_all_down_returns_unhealthy(self):
        """All components down → 'status: unhealthy', all show ✗."""
        store = self._mock_store(fail=True)
        http = self._mock_http(fail=True)

        with patch("rag.mcp.server.PostgresHybridStore", return_value=store):
            with patch("httpx.AsyncClient", return_value=http):
                result = await health()

        assert "unhealthy" in result
        assert result.count("✗") == 3

    @pytest.mark.asyncio
    async def test_api_500_counts_as_failure(self):
        """HTTP 500 from the API is treated as a failed check."""
        store = self._mock_store()
        http = self._mock_http(status_code=500)

        with patch("rag.mcp.server.PostgresHybridStore", return_value=store):
            with patch("httpx.AsyncClient", return_value=http):
                result = await health()

        assert "status: ok" not in result
        assert "embedding_api: ✗" in result
        assert "llm_api: ✗" in result

    @pytest.mark.asyncio
    async def test_output_lists_all_components(self):
        """Output always mentions db, embedding_api, and llm_api."""
        store = self._mock_store()
        http = self._mock_http()

        with patch("rag.mcp.server.PostgresHybridStore", return_value=store):
            with patch("httpx.AsyncClient", return_value=http):
                result = await health()

        assert "db:" in result
        assert "embedding_api:" in result
        assert "llm_api:" in result
