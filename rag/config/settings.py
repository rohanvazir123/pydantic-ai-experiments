"""
Settings configuration for MongoDB RAG Agent.

Module: rag.config.settings
===========================

This module provides configuration management using Pydantic Settings with
environment variable support. All configuration is loaded from .env file.

Classes
-------
Settings(BaseSettings)
    Main configuration class with all application settings.

    Attributes:
        mongodb_uri: str              - MongoDB Atlas connection string
        mongodb_database: str         - Database name (default: "rag_db")
        mongodb_collection_documents: str - Documents collection (default: "documents")
        mongodb_collection_chunks: str    - Chunks collection (default: "chunks")
        mongodb_vector_index: str     - Vector index name (default: "vector_index")
        mongodb_text_index: str       - Text index name (default: "text_index")
        llm_provider: str             - LLM provider (default: "ollama")
        llm_api_key: str              - LLM API key
        llm_model: str                - LLM model name (default: "llama3.1:8b")
        llm_base_url: str | None      - LLM API base URL
        embedding_provider: str       - Embedding provider (default: "ollama")
        embedding_api_key: str        - Embedding API key
        embedding_model: str          - Embedding model (default: "nomic-embed-text:latest")
        embedding_base_url: str | None - Embedding API base URL
        embedding_dimension: int      - Vector dimension (default: 768)
        default_match_count: int      - Default search results (default: 10)
        langfuse_enabled: bool        - Enable Langfuse tracing (default: False)

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

    # Use singleton
    print(settings.mongodb_database)

    # Or load fresh instance
    s = load_settings()
    print(mask_credential(s.mongodb_uri))
"""

from dotenv import load_dotenv
from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = ConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # MongoDB Configuration
    mongodb_uri: str = Field(..., description="MongoDB Atlas connection string")

    mongodb_database: str = Field(default="rag_db", description="MongoDB database name")

    mongodb_collection_documents: str = Field(
        default="documents", description="Collection for source documents"
    )

    mongodb_collection_chunks: str = Field(
        default="chunks", description="Collection for document chunks with embeddings"
    )

    mongodb_vector_index: str = Field(
        default="vector_index",
        description="Vector search index name (must be created in Atlas UI)",
    )

    mongodb_text_index: str = Field(
        default="text_index",
        description="Full-text search index name (must be created in Atlas UI)",
    )

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


def load_settings() -> Settings:
    """Load settings with proper error handling."""
    try:
        return Settings()
    except Exception as e:
        error_msg = f"Failed to load settings: {e}"
        if "mongodb_uri" in str(e).lower():
            error_msg += "\nMake sure to set MONGODB_URI in your .env file"
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
    _logger.info("--- MongoDB Configuration ---")
    _logger.info(f"  URI: {mask_credential(s.mongodb_uri)}")
    _logger.info(f"  Database: {s.mongodb_database}")
    _logger.info(f"  Documents Collection: {s.mongodb_collection_documents}")
    _logger.info(f"  Chunks Collection: {s.mongodb_collection_chunks}")
    _logger.info(f"  Vector Index: {s.mongodb_vector_index}")
    _logger.info(f"  Text Index: {s.mongodb_text_index}")

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
