"""Smoke tests for the knowledge package.

Fast pass — no external services. Verifies the key import chains compile
and core data structures behave correctly. Run after any structural change
before doing a full test suite.
"""

import os
from unittest import mock

import pytest

# Required env for Settings validation
_REQUIRED_ENV = {
    "DATABASE_URL": "postgresql://ragv2:test@localhost:5432/ragv2_test",
    "AGE_DATABASE_URL": "postgresql://age:test@localhost:5433/age_test",
}


# ---------------------------------------------------------------------------
# Package imports
# ---------------------------------------------------------------------------

class TestPackageImports:
    def test_config_imports(self) -> None:
        from knowledge.config.settings import Settings, load_settings  # noqa: F401

    def test_ingestion_models_import(self) -> None:
        from knowledge.ingestion.models import (  # noqa: F401
            ChunkData,
            ChunkingConfig,
            IngestionConfig,
            SearchResult,
        )

    def test_retrieval_imports(self) -> None:
        from knowledge.retrieval.fusion import (  # noqa: F401
            fuse_to_search_results,
            rrf_fuse,
        )
        from knowledge.retrieval.retriever import Retriever  # noqa: F401

    def test_store_imports(self) -> None:
        from knowledge.store.cache import RedisCache  # noqa: F401
        from knowledge.store.vector import PostgresHybridStore  # noqa: F401

    def test_hooks_imports(self) -> None:
        from knowledge.hooks.context import HookContext  # noqa: F401
        from knowledge.hooks.registry import HookPoint, HookRegistry  # noqa: F401

    def test_bus_imports(self) -> None:
        from knowledge.bus.backoff import exponential_backoff  # noqa: F401
        from knowledge.bus.circuit_breaker import CircuitBreaker  # noqa: F401
        from knowledge.bus.schemas import IngestJob, SearchRequest  # noqa: F401

    def test_validation_imports(self) -> None:
        from knowledge.validation.pipeline import ValidationPipeline  # noqa: F401

    def test_observability_imports(self) -> None:
        from knowledge.observability.metrics import (  # noqa: F401
            observe_request,
            observe_retrieval,
        )

    def test_memory_imports(self) -> None:
        from knowledge.memory.working_memory import (  # noqa: F401
            AssembledContext,
            assemble,
        )

    def test_evaluation_imports(self) -> None:
        from knowledge.evaluation.schemas import EvalResult  # noqa: F401


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class TestSettings:
    def _make(self, **overrides: str):  # type: ignore[no-untyped-def]
        from knowledge.config.settings import Settings

        env = {**_REQUIRED_ENV, **{k.upper(): v for k, v in overrides.items()}}
        with mock.patch.dict(os.environ, env, clear=True):
            return Settings(_env_file=None)  # type: ignore[call-arg]

    def test_loads_with_required_fields(self) -> None:
        s = self._make()
        assert "localhost" in s.database_url

    def test_default_llm_model(self) -> None:
        s = self._make()
        assert s.llm_model  # not empty

    def test_default_embedding_dimension(self) -> None:
        s = self._make()
        assert s.embedding_dimension > 0

    def test_masked_hides_api_key(self) -> None:
        s = self._make(LLM_API_KEY="sk-secret")
        assert s.masked()["llm_api_key"] == "***"

    def test_missing_database_url_raises(self) -> None:
        with mock.patch.dict(os.environ, {"AGE_DATABASE_URL": "postgresql://x"}, clear=True):
            from knowledge.config.settings import Settings
            with pytest.raises(Exception):
                Settings(_env_file=None)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Ingestion models
# ---------------------------------------------------------------------------

class TestIngestionModels:
    def test_chunk_data_defaults(self) -> None:
        from knowledge.ingestion.models import ChunkData

        c = ChunkData(content="hello world")
        assert c.chunk_index == 0
        assert c.corpus_id == ""
        assert c.metadata == {}

    def test_chunking_config_defaults(self) -> None:
        from knowledge.ingestion.models import ChunkingConfig

        cfg = ChunkingConfig()
        assert cfg.chunk_size > 0
        assert cfg.chunk_overlap >= 0
        assert cfg.chunk_overlap < cfg.chunk_size

    def test_search_result_score_fields(self) -> None:
        from knowledge.ingestion.models import SearchResult

        r = SearchResult(
            chunk_id="00000000-0000-0000-0000-000000000001",
            document_id="00000000-0000-0000-0000-000000000002",
            content="test",
            document_title="Doc",
            document_source="s3://bucket/doc.pdf",
            raw_score=0.85,
            raw_score_type="cosine_similarity",
        )
        assert r.raw_score == pytest.approx(0.85)
        assert r.confidence is None


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

class TestRRFFusion:
    def test_rrf_fuse_empty(self) -> None:
        from knowledge.retrieval.fusion import rrf_fuse

        result = rrf_fuse([])
        assert result == []

    def test_rrf_fuse_single_list(self) -> None:
        from knowledge.retrieval.fusion import rrf_fuse

        items = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        result = rrf_fuse([items])
        assert [r["id"] for r in result] == ["a", "b", "c"]

    def test_rrf_fuse_deduplicates(self) -> None:
        from knowledge.retrieval.fusion import rrf_fuse

        list1 = [{"id": "a"}, {"id": "b"}]
        list2 = [{"id": "b"}, {"id": "c"}]
        result = rrf_fuse([list1, list2])
        ids = [r["id"] for r in result]
        assert len(ids) == len(set(ids))

    def test_rrf_fuse_agreement_boosts_rank(self) -> None:
        from knowledge.retrieval.fusion import rrf_fuse

        # "a" appears first in both lists → should rank above "b" or "c"
        list1 = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        list2 = [{"id": "a"}, {"id": "d"}, {"id": "e"}]
        result = rrf_fuse([list1, list2])
        assert result[0]["id"] == "a"


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

class TestHooks:
    def test_registry_registers_and_fires(self) -> None:
        import asyncio

        from knowledge.hooks.context import HookContext
        from knowledge.hooks.registry import HookPoint, HookRegistry

        reg = HookRegistry()
        fired: list[str] = []

        @reg.hook(HookPoint.POST_RETRIEVE)
        async def my_hook(ctx: HookContext) -> HookContext:
            fired.append("ran")
            return ctx

        ctx = HookContext(query="test")
        asyncio.run(reg.fire(HookPoint.POST_RETRIEVE, ctx))
        assert fired == ["ran"]

    def test_unregistered_point_noop(self) -> None:
        import asyncio

        from knowledge.hooks.context import HookContext
        from knowledge.hooks.registry import HookPoint, HookRegistry

        reg = HookRegistry()
        ctx = HookContext(query="q")
        result = asyncio.run(reg.fire(HookPoint.PRE_INGEST, ctx))
        assert result is ctx


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_circuit_breaker_imports(self) -> None:
        from knowledge.bus.circuit_breaker import CircuitBreaker  # noqa: F401

    def test_backoff_imports(self) -> None:
        from knowledge.bus.backoff import exponential_backoff  # noqa: F401


# ---------------------------------------------------------------------------
# API app factory (no services — just shape)
# ---------------------------------------------------------------------------

class TestAPIAppFactory:
    def test_create_app_returns_fastapi(self) -> None:
        from fastapi import FastAPI

        from knowledge.api.app import create_app
        from knowledge.config.settings import Settings

        env = {**_REQUIRED_ENV}
        with mock.patch.dict(os.environ, env, clear=True):
            s = Settings(_env_file=None)  # type: ignore[call-arg]
        app = create_app(settings=s)
        assert isinstance(app, FastAPI)

    def test_health_route_registered(self) -> None:
        from knowledge.api.app import create_app
        from knowledge.config.settings import Settings

        env = {**_REQUIRED_ENV}
        with mock.patch.dict(os.environ, env, clear=True):
            s = Settings(_env_file=None)  # type: ignore[call-arg]
        app = create_app(settings=s)
        paths = [r.path for r in app.routes]  # type: ignore[attr-defined]
        assert any("health" in p for p in paths)
