from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Local LLM
    llm_model: str = Field(
        default="llama3",
        description="Local LLM model name (Ollama / vLLM / llama.cpp)",
    )
    llm_base_url: str = Field(
        default="http://localhost:11434",
        description="Local LLM server URL",
    )

    # Embeddings
    embedding_model: str = Field(
        default="nomic-embed-text",
        description="Local embedding model",
    )

    # Mongo
    mongo_uri: str = Field(default="mongodb://localhost:27017")
    mongo_db: str = Field(default="rag")
    mongo_collection: str = Field(default="chunks")

    # Retrieval
    top_k: int = Field(default=5)


settings = Settings()
