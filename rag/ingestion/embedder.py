"""Embedding generation for document chunks."""

import logging
import time
from collections.abc import Callable
from datetime import datetime

import openai
from async_lru import alru_cache

from rag.config.settings import load_settings
from rag.ingestion.models import ChunkData

logger = logging.getLogger(__name__)


# Module-level client (lazy initialized)
_client: openai.AsyncOpenAI | None = None
_settings = None


def _get_client() -> openai.AsyncOpenAI:
    """Get or create the shared OpenAI client."""
    global _client, _settings
    if _client is None:
        _settings = load_settings()
        _client = openai.AsyncOpenAI(
            api_key=_settings.embedding_api_key,
            base_url=_settings.embedding_base_url,
        )
    return _client


@alru_cache(maxsize=1000)
async def _cached_embed(text: str, model: str) -> tuple[float, ...]:
    """
    Cached embedding generation.

    Uses @alru_cache for async LRU caching. Returns tuple (hashable) instead of list.
    """
    client = _get_client()
    start_time = time.time()
    response = await client.embeddings.create(model=model, input=text)
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(f"[CACHE MISS] Generated embedding in {elapsed_ms:.0f}ms")
    # Return as tuple (hashable, required by alru_cache)
    return tuple(response.data[0].embedding)


class EmbeddingGenerator:
    """Generates embeddings for document chunks using OpenAI-compatible API."""

    def __init__(self, model: str | None = None, batch_size: int = 100):
        """
        Initialize embedding generator.

        Args:
            model: Embedding model to use (defaults to settings)
            batch_size: Number of texts to process in parallel
        """
        self.settings = load_settings()
        self.model = model or self.settings.embedding_model
        self.batch_size = batch_size

        # Initialize OpenAI client for embeddings
        self.client = openai.AsyncOpenAI(
            api_key=self.settings.embedding_api_key,
            base_url=self.settings.embedding_base_url,
        )

        # Model-specific configurations
        self.model_configs = {
            "text-embedding-3-small": {"dimensions": 1536, "max_tokens": 8191},
            "text-embedding-3-large": {"dimensions": 3072, "max_tokens": 8191},
            "text-embedding-ada-002": {"dimensions": 1536, "max_tokens": 8191},
        }

        self.config = self.model_configs.get(
            self.model, {"dimensions": 1536, "max_tokens": 8191}
        )

    async def generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for a single text (no caching).

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Truncate text if too long (rough estimation: 4 chars per token)
        if len(text) > self.config["max_tokens"] * 4:
            text = text[: self.config["max_tokens"] * 4]

        response = await self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        # Truncate texts if too long
        processed_texts = []
        for text in texts:
            if len(text) > self.config["max_tokens"] * 4:
                text = text[: self.config["max_tokens"] * 4]
            processed_texts.append(text)

        response = await self.client.embeddings.create(
            model=self.model, input=processed_texts
        )

        return [data.embedding for data in response.data]

    async def embed_chunks(
        self, chunks: list[ChunkData], progress_callback: Callable | None = None
    ) -> list[ChunkData]:
        """
        Generate embeddings for document chunks.

        Args:
            chunks: List of document chunks
            progress_callback: Optional callback for progress updates

        Returns:
            Chunks with embeddings added
        """
        if not chunks:
            return chunks

        logger.info(f"Generating embeddings for {len(chunks)} chunks")

        # Process chunks in batches
        embedded_chunks = []
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i : i + self.batch_size]
            batch_texts = [chunk.content for chunk in batch_chunks]

            # Generate embeddings for this batch
            embeddings = await self.generate_embeddings_batch(batch_texts)

            # Add embeddings to chunks
            for chunk, embedding in zip(batch_chunks, embeddings):
                embedded_chunk = ChunkData(
                    content=chunk.content,
                    index=chunk.index,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    metadata={
                        **chunk.metadata,
                        "embedding_model": self.model,
                        "embedding_generated_at": datetime.now().isoformat(),
                    },
                    token_count=chunk.token_count,
                )
                embedded_chunk.embedding = embedding
                embedded_chunks.append(embedded_chunk)

            # Progress update
            current_batch = (i // self.batch_size) + 1
            if progress_callback:
                progress_callback(current_batch, total_batches)

            logger.info(f"Processed batch {current_batch}/{total_batches}")

        logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
        return embedded_chunks

    async def embed_query(self, query: str, use_cache: bool = True) -> list[float]:
        """
        Generate embedding for a search query with caching.

        Args:
            query: Search query
            use_cache: Whether to use embedding cache (default: True)

        Returns:
            Query embedding
        """
        # Truncate if needed
        if len(query) > self.config["max_tokens"] * 4:
            query = query[: self.config["max_tokens"] * 4]

        if use_cache:
            # Use cached function (returns tuple, convert to list)
            result = await _cached_embed(query, self.model)
            return list(result)
        else:
            # Bypass cache
            return await self.generate_embedding(query)

    @staticmethod
    def get_cache_stats() -> dict:
        """Get embedding cache statistics."""
        info = _cached_embed.cache_info()
        total = info.hits + info.misses
        hit_rate = (info.hits / total * 100) if total > 0 else 0
        return {
            "size": info.currsize,
            "max_size": info.maxsize,
            "hits": info.hits,
            "misses": info.misses,
            "hit_rate": f"{hit_rate:.1f}%",
        }

    @staticmethod
    def clear_cache() -> None:
        """Clear the embedding cache."""
        _cached_embed.cache_clear()
        logger.info("Embedding cache cleared")

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for this model."""
        return self.config["dimensions"]


def create_embedder(model: str | None = None, **kwargs) -> EmbeddingGenerator:
    """
    Create embedding generator.

    Args:
        model: Embedding model to use
        **kwargs: Additional arguments for EmbeddingGenerator

    Returns:
        EmbeddingGenerator instance
    """
    return EmbeddingGenerator(model=model, **kwargs)
