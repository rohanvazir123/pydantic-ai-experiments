"""Tests for the FastAPI REST API layer (rag.api.app).

All external dependencies (DB, LLM, ingestion pipeline) are mocked so that
these tests run without any live services.
"""

import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from rag.api.app import app

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient]:
    """Async test client backed by the FastAPI ASGI app (no real network)."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


def _make_agent_result(text: str) -> MagicMock:
    """Return a minimal mock that looks like a Pydantic AI RunResult."""
    result = MagicMock()
    result.output = text
    return result


# ---------------------------------------------------------------------------
# POST /v1/chat
# ---------------------------------------------------------------------------


class TestChatEndpoint:
    """Tests for POST /v1/chat."""

    @pytest.mark.asyncio
    async def test_chat_returns_answer(self, client: AsyncClient):
        """Happy path: returns answer from traced_agent_run."""
        mock_result = _make_agent_result(
            "NeuralFlow AI builds intelligent document systems."
        )

        with patch("rag.api.app.traced_agent_run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            response = await client.post(
                "/v1/chat", json={"query": "What does NeuralFlow do?"}
            )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "NeuralFlow AI builds intelligent document systems."
        assert body["session_id"] is None

    @pytest.mark.asyncio
    async def test_chat_passes_optional_fields(self, client: AsyncClient):
        """user_id, session_id, and message_history are forwarded correctly."""
        mock_result = _make_agent_result("42 engineers.")

        with patch("rag.api.app.traced_agent_run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            response = await client.post(
                "/v1/chat",
                json={
                    "query": "How many engineers?",
                    "user_id": "alice",
                    "session_id": "sess-1",
                    "message_history": [{"role": "user", "content": "Hi"}],
                },
            )

        assert response.status_code == 200
        assert response.json()["session_id"] == "sess-1"

        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["user_id"] == "alice"
        assert call_kwargs["session_id"] == "sess-1"
        assert call_kwargs["message_history"] == [{"role": "user", "content": "Hi"}]

    @pytest.mark.asyncio
    async def test_chat_missing_query_returns_422(self, client: AsyncClient):
        """Request body without 'query' must be rejected with 422."""
        response = await client.post("/v1/chat", json={"user_id": "alice"})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_agent_error_returns_500(self, client: AsyncClient):
        """If traced_agent_run raises, the endpoint returns HTTP 500."""
        with patch("rag.api.app.traced_agent_run", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = RuntimeError("LLM timeout")
            response = await client.post("/v1/chat", json={"query": "crash please"})

        assert response.status_code == 500
        assert "LLM timeout" in response.json()["detail"]


# ---------------------------------------------------------------------------
# POST /v1/chat/stream
# ---------------------------------------------------------------------------


class TestChatStreamEndpoint:
    """Tests for POST /v1/chat/stream."""

    @pytest.mark.asyncio
    async def test_stream_returns_sse_events(self, client: AsyncClient):
        """Happy path: events arrive as SSE delta lines followed by done."""

        async def _fake_stream_text(delta: bool = False):
            for token in ["Hello", " world", "!"]:
                yield token

        mock_streamed = MagicMock()
        mock_streamed.stream_text = _fake_stream_text

        @asynccontextmanager
        async def _fake_run_stream(*args: Any, **kwargs: Any):
            yield mock_streamed

        with patch("rag.api.app.RAGState") as mock_state_cls:
            mock_state = AsyncMock()
            mock_state_cls.return_value = mock_state

            with patch("rag.api.app.agent") as mock_agent:
                mock_agent.run_stream = _fake_run_stream

                response = await client.post(
                    "/v1/chat/stream", json={"query": "stream test"}
                )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        lines = [ln for ln in response.text.splitlines() if ln.startswith("data: ")]
        payloads = [json.loads(ln[len("data: ") :]) for ln in lines]

        delta_texts = [p["delta"] for p in payloads if "delta" in p]
        assert delta_texts == ["Hello", " world", "!"]
        assert payloads[-1].get("done") is True

    @pytest.mark.asyncio
    async def test_stream_error_yields_error_event(self, client: AsyncClient):
        """If the generator raises, an error SSE event is emitted."""

        @asynccontextmanager
        async def _bad_run_stream(*args: Any, **kwargs: Any):
            raise RuntimeError("stream boom")
            yield  # make it a generator

        with patch("rag.api.app.RAGState") as mock_state_cls:
            mock_state = AsyncMock()
            mock_state_cls.return_value = mock_state

            with patch("rag.api.app.agent") as mock_agent:
                mock_agent.run_stream = _bad_run_stream

                response = await client.post(
                    "/v1/chat/stream", json={"query": "explode"}
                )

        assert response.status_code == 200  # headers already sent
        lines = [ln for ln in response.text.splitlines() if ln.startswith("data: ")]
        assert any("error" in json.loads(ln[len("data: ") :]) for ln in lines)


# ---------------------------------------------------------------------------
# POST /v1/ingest
# ---------------------------------------------------------------------------


class TestIngestEndpoint:
    """Tests for POST /v1/ingest."""

    def _mock_pipeline(self, results: list[Any]) -> MagicMock:
        """Build a mock pipeline whose ingest_documents returns *results*."""
        pipeline = MagicMock()
        pipeline.initialize = AsyncMock()
        pipeline.ingest_documents = AsyncMock(return_value=results)
        pipeline.close = AsyncMock()
        return pipeline

    def _make_result(
        self,
        doc_id: str = "doc-1",
        title: str = "Test Doc",
        chunks: int = 5,
        ms: float = 100.0,
        errors: list[str] | None = None,
    ) -> MagicMock:
        r = MagicMock()
        r.document_id = doc_id
        r.title = title
        r.chunks_created = chunks
        r.processing_time_ms = ms
        r.errors = errors or []
        return r

    @pytest.mark.asyncio
    async def test_ingest_returns_summary(self, client: AsyncClient):
        """Happy path: returns document count and total chunks."""
        results = [
            self._make_result("id-1", "Doc A", 3),
            self._make_result("id-2", "Doc B", 7),
        ]
        mock_pipeline = self._mock_pipeline(results)

        with patch("rag.api.app.create_pipeline", return_value=mock_pipeline):
            response = await client.post(
                "/v1/ingest",
                json={"documents_folder": "rag/documents", "clean": False},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["total_documents"] == 2
        assert body["total_chunks"] == 10
        assert body["results"][0]["title"] == "Doc A"
        assert body["results"][1]["chunks_created"] == 7

    @pytest.mark.asyncio
    async def test_ingest_passes_config_to_pipeline(self, client: AsyncClient):
        """chunk_size and max_tokens are forwarded to create_pipeline."""
        mock_pipeline = self._mock_pipeline([self._make_result()])

        with patch(
            "rag.api.app.create_pipeline", return_value=mock_pipeline
        ) as mock_create:
            await client.post(
                "/v1/ingest",
                json={
                    "documents_folder": "custom/path",
                    "clean": True,
                    "chunk_size": 500,
                    "max_tokens": 128,
                },
            )

        mock_create.assert_called_once_with(
            documents_folder="custom/path",
            clean=True,
            chunk_size=500,
            max_tokens=128,
        )

    @pytest.mark.asyncio
    async def test_ingest_chunk_size_below_minimum_returns_422(
        self, client: AsyncClient
    ):
        """chunk_size < 100 violates the ge=100 constraint → 422."""
        response = await client.post("/v1/ingest", json={"chunk_size": 10})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_ingest_pipeline_error_returns_500(self, client: AsyncClient):
        """If the pipeline raises, the endpoint returns HTTP 500."""
        mock_pipeline = self._mock_pipeline([])
        mock_pipeline.ingest_documents.side_effect = RuntimeError("disk full")

        with patch("rag.api.app.create_pipeline", return_value=mock_pipeline):
            response = await client.post("/v1/ingest", json={})

        assert response.status_code == 500
        assert "disk full" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_ingest_pipeline_close_called_on_error(self, client: AsyncClient):
        """close() must be called even when ingest_documents raises."""
        mock_pipeline = self._mock_pipeline([])
        mock_pipeline.ingest_documents.side_effect = RuntimeError("boom")

        with patch("rag.api.app.create_pipeline", return_value=mock_pipeline):
            await client.post("/v1/ingest", json={})

        mock_pipeline.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for GET /health."""

    @pytest.mark.asyncio
    async def test_health_all_ok(self, client: AsyncClient):
        """Returns 200 and status='ok' when all three checks pass."""
        mock_store = MagicMock()
        mock_store.initialize = AsyncMock()
        mock_store.close = AsyncMock()

        mock_http_response = MagicMock()
        mock_http_response.status_code = 200

        with patch("rag.api.app.PostgresHybridStore", return_value=mock_store):
            with patch("httpx.AsyncClient") as mock_http_cls:
                mock_http = AsyncMock()
                mock_http.get = AsyncMock(return_value=mock_http_response)
                mock_http.__aenter__ = AsyncMock(return_value=mock_http)
                mock_http.__aexit__ = AsyncMock(return_value=False)
                mock_http_cls.return_value = mock_http

                response = await client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["db"] is True
        assert body["embedding_api"] is True
        assert body["llm_api"] is True

    @pytest.mark.asyncio
    async def test_health_db_down_returns_503(self, client: AsyncClient):
        """Returns 503 with status='degraded'/'unhealthy' when DB is unreachable."""
        mock_store = MagicMock()
        mock_store.initialize = AsyncMock(side_effect=ConnectionRefusedError("no db"))
        mock_store.close = AsyncMock()

        mock_http_response = MagicMock()
        mock_http_response.status_code = 200

        with patch("rag.api.app.PostgresHybridStore", return_value=mock_store):
            with patch("httpx.AsyncClient") as mock_http_cls:
                mock_http = AsyncMock()
                mock_http.get = AsyncMock(return_value=mock_http_response)
                mock_http.__aenter__ = AsyncMock(return_value=mock_http)
                mock_http.__aexit__ = AsyncMock(return_value=False)
                mock_http_cls.return_value = mock_http

                response = await client.get("/health")

        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail["db"] is False
        assert detail["status"] in ("degraded", "unhealthy")

    @pytest.mark.asyncio
    async def test_health_all_down_returns_unhealthy(self, client: AsyncClient):
        """Returns status='unhealthy' when all components fail."""
        mock_store = MagicMock()
        mock_store.initialize = AsyncMock(side_effect=ConnectionRefusedError())
        mock_store.close = AsyncMock()

        with patch("rag.api.app.PostgresHybridStore", return_value=mock_store):
            with patch("httpx.AsyncClient") as mock_http_cls:
                mock_http = AsyncMock()
                mock_http.get = AsyncMock(side_effect=ConnectionRefusedError())
                mock_http.__aenter__ = AsyncMock(return_value=mock_http)
                mock_http.__aexit__ = AsyncMock(return_value=False)
                mock_http_cls.return_value = mock_http

                response = await client.get("/health")

        assert response.status_code == 503
        assert response.json()["detail"]["status"] == "unhealthy"
