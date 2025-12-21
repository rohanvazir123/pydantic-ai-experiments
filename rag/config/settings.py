"""Settings configuration for MongoDB RAG Agent."""

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
