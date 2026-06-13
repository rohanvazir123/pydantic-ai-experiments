"""Async embedder with L1 in-process cache, timeout, and exponential backoff.

L1 cache: bounded dict per worker process (max 1000 entries, FIFO eviction).
          Avoids round-trips to Ollama for repeated texts during batch ingestion.

Retries: on RateLimitError, APIConnectionError, APITimeoutError (transient).
         Non-retriable errors (AuthenticationError, bad request) bubble up immediately.
"""

import asyncio
import logging
from typing import Any, cast

from knowledge.bus.backoff import exponential_backoff
from knowledge.config.settings import Settings, load_settings
from knowledge.ingestion.models import ChunkData

logger = logging.getLogger(__name__)

_L1_MAX = 1_000   # entries per worker process

# Retriable openai exception class names (checked by string to avoid hard import)
_RETRIABLE_NAMES = {"RateLimitError", "APIConnectionError", "APITimeoutError"}


def _is_retriable(exc: Exception) -> bool:
    return type(exc).__name__ in _RETRIABLE_NAMES


class Embedder:
    """Async OpenAI-compatible embedder.

    Instantiate once per worker; the L1 cache is process-local.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or load_settings()
        self._client: Any = None           # openai.AsyncOpenAI, set lazily
        self._cache: dict[str, list[float]] = {}

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                base_url=self._settings.embedding_base_url,
                api_key=self._settings.embedding_api_key,
            )
        return self._client

    # ── L1 cache helpers ──────────────────────────────────────────────────────

    def _cache_get(self, text: str) -> list[float] | None:
        return self._cache.get(text)

    def _cache_set(self, text: str, vector: list[float]) -> None:
        if len(self._cache) >= _L1_MAX:
            # FIFO eviction — remove insertion-order oldest
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[text] = vector

    # ── Core embed call ───────────────────────────────────────────────────────

    async def _call_api(self, text: str) -> list[float]:
        """Call the embedding API with timeout and exponential backoff."""
        client = self._get_client()
        attempt = 0
        last_exc: Exception | None = None

        while attempt < self._settings.embedding_retry_attempts:
            attempt += 1
            try:
                response = await asyncio.wait_for(
                    client.embeddings.create(
                        input=text,
                        model=self._settings.embedding_model,
                    ),
                    timeout=self._settings.embedding_timeout_s,
                )
                return cast("list[float]", response.data[0].embedding)
            except TimeoutError as exc:
                last_exc = exc
                logger.warning("Embed timeout (attempt %d/%d)", attempt, self._settings.embedding_retry_attempts)
            except Exception as exc:
                if not _is_retriable(exc):
                    raise
                last_exc = exc
                logger.warning(
                    "Embed transient error (attempt %d/%d): %s",
                    attempt, self._settings.embedding_retry_attempts, exc,
                )

            if attempt < self._settings.embedding_retry_attempts:
                await asyncio.sleep(
                    exponential_backoff(
                        attempt,
                        base_s=self._settings.embedding_retry_backoff_s,
                    )
                )

        raise RuntimeError(
            f"Embedding failed after {self._settings.embedding_retry_attempts} attempts"
        ) from last_exc

    # ── Public API ────────────────────────────────────────────────────────────

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string. Hits L1 cache first."""
        cached = self._cache_get(text)
        if cached is not None:
            return cached
        vector = await self._call_api(text)
        self._cache_set(text, vector)
        return vector

    async def embed_query(self, query: str) -> list[float]:
        """Embed a search query (alias for embed — same model)."""
        return await self.embed(query)

    async def embed_batch(self, chunks: list[ChunkData]) -> list[ChunkData]:
        """Embed all chunks, attaching the embedding to chunk metadata.

        Returns the same list with metadata["embedding"] populated.
        Processes concurrently with a semaphore to avoid flooding Ollama.
        """
        sem = asyncio.Semaphore(8)

        async def _embed_one(chunk: ChunkData) -> ChunkData:
            async with sem:
                vector = await self.embed(chunk.content)
                chunk.metadata["embedding"] = vector
                return chunk

        return list(await asyncio.gather(*[_embed_one(c) for c in chunks]))

    def cache_size(self) -> int:
        return len(self._cache)
