"""Central Pydantic-settings configuration for the knowledge module.

All tunables live here. Every module imports Settings (or the cached
load_settings() singleton) rather than reading os.environ directly.
"""

import json
from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Self


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class CorpusConfig:
    """Minimal in-memory representation of a corpus.

    Parsed from CORPUS_CONFIGS_JSON at startup. Not a Pydantic model itself —
    settings stores it as a plain list parsed in a validator.
    """

    __slots__ = (
        "id",
        "display_name",
        "source_folders",
        "allowed_roles",
        "metadata_tags",
        "enable_graph_extraction",
        "graph_ontology_path",
        "graph_extraction_provider",
        "graph_extraction_model",
        "graph_extraction_contract",
        "graph_processing_mode",
        "graph_extraction_backend",
        "allowed_topics",
        "data_region",
    )

    def __init__(self, data: dict) -> None:  # type: ignore[type-arg]
        self.id: str = data["id"]
        self.display_name: str = data.get("display_name", self.id)
        self.source_folders: list[Path] = [
            Path(p) for p in data.get("source_folders", [])
        ]
        self.allowed_roles: list[str] = data.get("allowed_roles", ["reader"])
        self.metadata_tags: dict[str, str] = data.get("metadata_tags", {})
        self.enable_graph_extraction: bool = data.get("enable_graph_extraction", False)
        self.graph_ontology_path: str | None = data.get("graph_ontology_path")
        self.graph_extraction_provider: str = data.get("graph_extraction_provider", "ollama")
        self.graph_extraction_model: str = data.get("graph_extraction_model", "llama3.2:3b")
        self.graph_extraction_contract: str = data.get("graph_extraction_contract", "staged")
        self.graph_processing_mode: str = data.get("graph_processing_mode", "many-to-one")
        self.graph_extraction_backend: str = data.get("graph_extraction_backend", "llm")
        self.allowed_topics: list[str] = data.get("allowed_topics", [])
        self.data_region: str = data.get("data_region", "us")


# ---------------------------------------------------------------------------
# Main settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """All runtime configuration.

    Loaded once from environment variables (or .env file) at startup.
    Use load_settings() for the cached singleton; instantiate directly in tests.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── PostgreSQL (main DB — pgvector) ──────────────────────────────────────
    database_url: str = Field(..., description="asyncpg DSN for the main PostgreSQL DB")

    # ── Apache AGE (graph DB, separate container) ─────────────────────────────
    age_database_url: str = Field(..., description="asyncpg DSN for the AGE PostgreSQL DB")
    age_graph_prefix: str = Field("kg", description="Prefix for AGE graph names: {prefix}_{tenant}_{corpus}")

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = Field("redis://localhost:6379", description="Redis connection URL")
    redis_max_connections: int = Field(20, ge=1)

    # ── LLM (default: Ollama local) ───────────────────────────────────────────
    llm_provider: str = Field("ollama")
    llm_model: str = Field("llama3.2:3b")
    llm_base_url: str = Field("http://localhost:11434/v1")
    llm_api_key: str = Field("ollama")
    llm_num_ctx: int = Field(8192, ge=512)

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_provider: str = Field("ollama")
    embedding_model: str = Field("nomic-embed-text:latest")
    embedding_base_url: str = Field("http://localhost:11434/v1")
    embedding_api_key: str = Field("ollama")
    embedding_dimension: int = Field(768, ge=64)

    # ── Model tiers ───────────────────────────────────────────────────────────
    model_tier_nano: str = Field("qwen2.5:0.5b")
    model_tier_small: str = Field("llama3.2:3b")
    model_tier_large: str = Field("llama3.1:70b")
    model_routing_enabled: bool = Field(True)
    model_routing_timeout_s: float = Field(3.0, gt=0)
    cloud_models_enabled: bool = Field(False)

    # ── VLM (optional, for docling PDF picture description) ───────────────────
    vlm_enabled: bool = Field(False)
    vlm_model: str = Field("qwen2.5vl:7b")
    vlm_base_url: str = Field("http://localhost:11434/v1")
    vlm_timeout: float = Field(120.0, gt=0)
    vlm_concurrency: int = Field(1, ge=1)

    # ── JWT auth ──────────────────────────────────────────────────────────────
    jwt_algorithm: str = Field("RS256")
    jwt_public_key_path: str = Field("infra/keys/public.pem")
    jwks_cache_ttl_s: int = Field(3600, ge=60)

    # ── JWE payload encryption ────────────────────────────────────────────────
    jwe_algorithm: str = Field("ECDH-ES+A256KW")
    jwe_content_encryption: str = Field("A256GCM")
    jwe_keys_dir: str = Field("infra/keys/jwe")

    # ── Semantic cache (L3 pgvector) ──────────────────────────────────────────
    semantic_cache_enabled: bool = Field(True)
    semantic_cache_threshold: float = Field(0.95, ge=0.0, le=1.0)
    semantic_cache_ttl_minutes: int = Field(60, ge=1)
    semantic_cache_max_rows: int = Field(10_000, ge=100)

    # ── Worker settings ───────────────────────────────────────────────────────
    ingest_worker_concurrency: int = Field(2, ge=1)
    retrieval_worker_concurrency: int = Field(2, ge=1)
    max_retries: int = Field(3, ge=1)
    job_timeout_s: float = Field(300.0, gt=0)

    # ── Ingestion / chunking ──────────────────────────────────────────────────
    chunk_max_tokens: int = Field(512, ge=64)
    graph_extraction_timeout_s: float = Field(120.0, gt=0)
    embedding_timeout_s: float = Field(30.0, gt=0)
    embedding_retry_attempts: int = Field(3, ge=1)
    embedding_retry_backoff_s: float = Field(1.0, gt=0)

    # ── DB query timeouts ─────────────────────────────────────────────────────
    db_query_timeout_s: float = Field(30.0, gt=0)
    db_health_timeout_s: float = Field(5.0, gt=0)

    # ── LLM call timeout ─────────────────────────────────────────────────────
    llm_timeout_s: float = Field(60.0, gt=0)

    # ── API rate limiting ─────────────────────────────────────────────────────
    api_rate_limit_rpm: int = Field(60, ge=1)
    api_rate_limit_burst: int = Field(10, ge=1)
    max_query_chars: int = Field(4096, ge=1)
    max_prompt_tokens: int = Field(8192, ge=512)
    max_output_tokens: int = Field(1024, ge=64)

    # ── Confidence thresholds ─────────────────────────────────────────────────
    min_confidence_score: float = Field(0.10, ge=0.0, le=1.0)
    confidence_warn_threshold: float = Field(0.40, ge=0.0, le=1.0)
    retrieval_confidence_threshold: float = Field(1.5, ge=0.0)
    judge_confidence_threshold: float = Field(0.60, ge=0.0, le=1.0)
    judge_k: int = Field(5, ge=1)

    # ── Scheduler ─────────────────────────────────────────────────────────────
    scheduler_enabled: bool = Field(True)
    scheduler_max_concurrent_jobs: int = Field(5, ge=1)

    # ── Cost circuit breaker ──────────────────────────────────────────────────
    system_daily_cost_limit_usd: float = Field(0.0, ge=0.0)

    # ── Alerts ───────────────────────────────────────────────────────────────
    alert_email: str = Field("rohan.vazirani@gmail.com")
    smtp_host: str = Field("smtp.gmail.com")
    smtp_port: int = Field(587, ge=1)
    smtp_user: str = Field("")
    smtp_password: str = Field("")
    smtp_from: str = Field("alerts@rag-system.local")

    # ── Observability ─────────────────────────────────────────────────────────
    langfuse_enabled: bool = Field(False)
    langfuse_public_key: str = Field("")
    langfuse_secret_key: str = Field("")
    langfuse_host: str = Field("http://localhost:3001")

    # ── User memory (Mem0) ────────────────────────────────────────────────────
    mem0_enabled: bool = Field(False)

    # ── Corpus configs (parsed JSON array) ────────────────────────────────────
    corpus_configs_json: str = Field("[]", description="JSON array of CorpusConfig dicts")

    # Parsed at validation time — not directly settable from env
    _corpus_configs: list[CorpusConfig] = []

    @model_validator(mode="after")
    def _parse_corpus_configs(self) -> Self:
        raw = json.loads(self.corpus_configs_json)
        self._corpus_configs = [CorpusConfig(item) for item in raw]
        return self

    @property
    def corpus_configs(self) -> list[CorpusConfig]:
        return self._corpus_configs

    # ── Derived helpers ───────────────────────────────────────────────────────

    def age_graph_name(self, tenant_id: str, corpus_id: str) -> str:
        """Return the AGE graph name for a corpus: {prefix}_{tenant}_{corpus}."""
        safe_tenant = tenant_id.replace("-", "_").replace(":", "_")
        safe_corpus = corpus_id.replace("-", "_").replace(":", "_")
        return f"{self.age_graph_prefix}_{safe_tenant}_{safe_corpus}"

    def masked(self) -> dict[str, str]:
        """Return a loggable dict with credentials replaced by ***.

        Masks the field regardless of whether it is empty — knowing that a
        credential is unset is itself information we don't want in logs.
        """
        data = self.model_dump()
        sensitive = {
            "smtp_password", "llm_api_key", "embedding_api_key",
            "langfuse_secret_key", "jwe_keys_dir",
        }
        return {
            k: "***" if k in sensitive else str(v)
            for k, v in data.items()
        }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """Return a cached Settings instance.

    Call once at app startup. Tests should instantiate Settings directly
    (bypassing the cache) to avoid cross-test pollution.
    """
    return Settings()
