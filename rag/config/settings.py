# Copyright 2024 The Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Settings configuration for RAG Agent.

Module: rag.config.settings
===========================

This module provides configuration management using Pydantic Settings with
environment variable support. All configuration is loaded from .env file.

Classes
-------
Settings(BaseSettings)
    Main configuration class with all application settings.

Functions
---------
load_settings() -> Settings
    Load and return Settings instance with error handling.

mask_credential(value: str) -> str
    Mask sensitive strings for safe logging (shows first/last 4 chars).

Module Attributes
-----------------
settings: Settings
    Singleton Settings instance for convenience import.

Usage
-----
    from rag.config.settings import settings, load_settings, mask_credential

    s = load_settings()
    print(mask_credential(s.database_url))
"""

import re

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # PostgreSQL Configuration (pgvector)
    database_url: str = Field(
        default="", description="PostgreSQL connection string"
    )

    postgres_table_documents: str = Field(
        default="documents", description="PostgreSQL table for source documents"
    )

    postgres_table_chunks: str = Field(
        default="chunks", description="PostgreSQL table for document chunks with embeddings"
    )

    # Connection pool sizing
    db_pool_min_size: int = Field(
        default=1, ge=1, description="Minimum PostgreSQL connection pool size"
    )
    db_pool_max_size: int = Field(
        default=10, ge=1, description="Maximum PostgreSQL connection pool size"
    )

    @field_validator("postgres_table_documents", "postgres_table_chunks", mode="before")
    @classmethod
    def validate_table_name(cls, v: str) -> str:
        """Prevent SQL injection via settings-supplied table names."""
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", v):
            raise ValueError(
                f"Invalid table name '{v}': only letters, digits, and underscores allowed"
            )
        return v

    # LLM Configuration (OpenAI-compatible)
    llm_provider: str = Field(
        default="ollama",
        description="LLM provider (openai, anthropic, gemini, ollama, etc.)",
    )

    llm_api_key: str = Field(
        default="ollama",
        description="API key for the LLM provider (use 'ollama' for local)",
    )

    llm_model: str = Field(
        default="llama3.1:8b", description="Model to use for search and summarization"
    )

    llm_base_url: str | None = Field(
        default="http://localhost:11434/v1",
        description="Base URL for the LLM API (for OpenAI-compatible providers)",
    )

    # Embedding Configuration
    embedding_provider: str = Field(
        default="ollama", description="Embedding provider (openai, ollama)"
    )

    embedding_api_key: str = Field(
        default="ollama",
        description="API key for embedding provider (use 'ollama' for local)",
    )

    embedding_model: str = Field(
        default="nomic-embed-text:latest", description="Embedding model to use"
    )

    embedding_base_url: str | None = Field(
        default="http://localhost:11434/v1", description="Base URL for embedding API"
    )

    embedding_dimension: int = Field(
        default=768, description="Embedding vector dimension (768 for nomic-embed-text)"
    )

    # Search Configuration
    default_match_count: int = Field(
        default=10, description="Default number of search results to return"
    )

    max_match_count: int = Field(
        default=50, description="Maximum number of search results allowed"
    )

    default_text_weight: float = Field(
        default=0.3, description="Default text weight for hybrid search (0-1)"
    )

    min_relevance_score: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum similarity score for retrieved chunks (0-1). "
            "Chunks below this threshold are dropped before being passed to the agent. "
            "Set to 0.0 to disable. Tune upward (0.5-0.6) for higher-precision corpora."
        ),
    )

    # HyDE (Hypothetical Document Embeddings) Configuration
    hyde_enabled: bool = Field(
        default=False,
        description="Enable HyDE: embed a hypothetical answer instead of the raw query for better recall",
    )

    # Reranker Configuration
    reranker_enabled: bool = Field(
        default=False,
        description="Enable reranking of initial search results for better precision",
    )

    reranker_type: str = Field(
        default="llm",
        description="Reranker type: 'llm' (no extra deps) or 'cross_encoder' (requires sentence-transformers)",
    )

    reranker_model: str = Field(
        default="",
        description="Model for reranker. For llm: defaults to llm_model. For cross_encoder: defaults to BAAI/bge-reranker-base",
    )

    reranker_overfetch_factor: int = Field(
        default=3,
        ge=1,
        description="Fetch this multiple of match_count from DB before reranking",
    )

    # Langfuse Observability Configuration
    langfuse_enabled: bool = Field(
        default=False, description="Enable Langfuse tracing and observability"
    )

    langfuse_public_key: str | None = Field(
        default=None, description="Langfuse public key"
    )

    langfuse_secret_key: str | None = Field(
        default=None, description="Langfuse secret key"
    )

    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse host URL (use https://cloud.langfuse.com for cloud)",
    )

    # Mem0 Memory Layer Configuration
    mem0_enabled: bool = Field(
        default=False, description="Enable Mem0 memory layer for user personalization"
    )

    mem0_collection_name: str = Field(
        default="mem0_memories", description="Collection name for Mem0 memories"
    )

    # Knowledge Graph backend
    # "age"      — Apache AGE Cypher graph (requires AGE container, see docker-compose.yml)
    # "postgres" — entity/relationship SQL tables (legacy, no Cypher support)
    kg_backend: str = Field(
        default="age",
        description="Knowledge graph backend: 'age' (Apache AGE Cypher, default) or 'postgres' (SQL tables, legacy)",
    )

    # Apache AGE connection (used when kg_backend = "age")
    age_database_url: str | None = Field(
        default=None,
        description="Connection string for the Apache AGE PostgreSQL instance "
                    "(e.g. postgresql://age_user:age_pass@localhost:5433/legal_graph). "
                    "Defaults to AGE_DATABASE_URL env var.",
    )

    age_graph_name: str = Field(
        default="legal_graph",
        description="Name of the AGE graph to use (created automatically on initialize).",
    )

    # Vision Model Configuration (for multimodal processing)
    vision_model_provider: str = Field(
        default="ollama",
        description="Vision model provider (ollama, openai, anthropic)",
    )

    vision_model: str = Field(
        default="llava:latest",
        description="Vision model name (llava:latest for Ollama, gpt-4o for OpenAI)",
    )

    vision_model_base_url: str | None = Field(
        default="http://localhost:11434/v1",
        description="Base URL for vision model API",
    )

    vision_model_api_key: str = Field(
        default="ollama",
        description="API key for vision model (use 'ollama' for local)",
    )

    vision_max_tokens: int = Field(
        default=1024,
        description="Maximum tokens for vision model response",
    )

    # Multimodal Processing Configuration
    multimodal_processing_enabled: bool = Field(
        default=True,
        description="Enable multimodal content processing (images, tables, etc.)",
    )

    # Context window for local LLMs (Ollama only — ignored for cloud providers).
    # Default 131072 = 128K tokens. Controls num_ctx passed via extra_body.
    llm_num_ctx: int = Field(
        default=131072,
        description="Context window size (num_ctx) for Ollama LLMs. "
                    "Passed via extra_body; ignored for non-Ollama providers.",
    )

    # KG extraction LLM — dedicated model for one-time entity extraction.
    # Defaults to the main LLM settings when not set.
    # Set KG_LLM_MODEL=llama3.1:8b in .env for best local extraction quality.
    kg_llm_model: str = Field(
        default="",
        description="Model for KG extraction (defaults to llm_model when empty)",
    )
    kg_llm_api_key: str = Field(
        default="",
        description="API key for KG extraction LLM (defaults to llm_api_key when empty)",
    )
    kg_llm_base_url: str | None = Field(
        default=None,
        description="Base URL for KG extraction LLM (defaults to llm_base_url when None)",
    )
    kg_llm_num_ctx: int = Field(
        default=4096,
        description=(
            "Context window (num_ctx) for the KG extraction LLM (Ollama only). "
            "Extraction prompts are ~2000-3000 tokens max (1500-char chunk + system prompt "
            "+ entity/relationship JSON for passes 2-5). 4096 is sufficient. "
            "8GB VRAM budget: Mistral 7B 4-bit ~4.5 GB model + ~512 MB KV at 4096 ctx = ~5 GB total. "
            "Raise to 8192 only if validation pass truncates on long contracts."
        ),
    )

    image_description_detail: str = Field(
        default="high",
        description="Detail level for image descriptions (low, high, auto)",
    )

    # Extraction pipeline (Bronze/Silver/Gold)
    kg_extraction_chunk_size: int = Field(
        default=1500,
        description="Characters per chunk for KG extraction (~1-3 clauses). "
                    "Smaller = more focused extraction, more LLM calls.",
    )
    kg_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for Silver layer promotion (0-1).",
    )
    kg_ontology_version: str = Field(
        default="1.0",
        description="Ontology version tag stored in Bronze artifacts.",
    )


def load_settings() -> Settings:
    """Load settings with proper error handling."""
    try:
        return Settings()
    except Exception as e:
        error_msg = f"Failed to load settings: {e}"
        if "database_url" in str(e).lower():
            error_msg += "\nMake sure to set DATABASE_URL in your .env file"
        raise ValueError(error_msg) from e


def mask_credential(value: str) -> str:
    """Mask credentials for safe display."""
    if not value or len(value) < 8:
        return "***"
    return value[:4] + "..." + value[-4:]


# Singleton instance for convenience
settings = load_settings()


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    _logger = logging.getLogger(__name__)

    # Standalone test for settings module
    _logger.info("=" * 60)
    _logger.info("RAG Settings Module Test")
    _logger.info("=" * 60)

    # Load settings
    s = load_settings()
    _logger.info("[Settings Loaded Successfully]")

    # Display configuration (with masked credentials)
    _logger.info("--- PostgreSQL Configuration ---")
    _logger.info(f"  Database URL: {mask_credential(s.database_url)}")
    _logger.info(f"  Documents Table: {s.postgres_table_documents}")
    _logger.info(f"  Chunks Table: {s.postgres_table_chunks}")

    _logger.info("--- LLM Configuration ---")
    _logger.info(f"  Provider: {s.llm_provider}")
    _logger.info(f"  Model: {s.llm_model}")
    _logger.info(f"  Base URL: {s.llm_base_url}")
    _logger.info(f"  API Key: {mask_credential(s.llm_api_key)}")

    _logger.info("--- Embedding Configuration ---")
    _logger.info(f"  Provider: {s.embedding_provider}")
    _logger.info(f"  Model: {s.embedding_model}")
    _logger.info(f"  Base URL: {s.embedding_base_url}")
    _logger.info(f"  Dimension: {s.embedding_dimension}")

    _logger.info("--- Search Configuration ---")
    _logger.info(f"  Default Match Count: {s.default_match_count}")
    _logger.info(f"  Max Match Count: {s.max_match_count}")
    _logger.info(f"  Default Text Weight: {s.default_text_weight}")

    _logger.info("--- Observability ---")
    _logger.info(f"  Langfuse Enabled: {s.langfuse_enabled}")
    if s.langfuse_enabled:
        _logger.info(f"  Langfuse Host: {s.langfuse_host}")

    _logger.info("=" * 60)
    _logger.info("Settings test completed successfully!")
    _logger.info("=" * 60)
