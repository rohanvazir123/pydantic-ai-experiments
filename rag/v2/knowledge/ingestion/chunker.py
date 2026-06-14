"""Docling HybridChunker wrapper.

Architecture anchor: mirrors rag/ingestion/chunkers/docling.py exactly.
Key pattern:
  - With DoclingDocument: HybridChunker.chunk() → contextualize() per chunk
    (contextualize prepends heading hierarchy — do not skip this call)
  - Without DoclingDocument: sliding window fallback
  - Chunk token count computed via the same HuggingFace tokenizer

Sync — runs in the same asyncio.to_thread block as DoclingProcessor.process()
so no additional to_thread wrapper is needed here.
"""

import logging
from typing import Any

from knowledge.ingestion.models import ChunkData, ChunkingConfig

logger = logging.getLogger(__name__)

TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class DoclingHybridChunker:
    """Docling HybridChunker wrapper for intelligent document splitting."""

    def __init__(
        self,
        config: ChunkingConfig,
        corpus_id: str = "",
        tenant_id: str = "",
        metadata_tags: dict[str, str] | None = None,
    ) -> None:
        self.config = config
        self.corpus_id = corpus_id
        self.tenant_id = tenant_id
        self.metadata_tags = metadata_tags or {}
        self._chunker: Any = None
        self._tokenizer: Any = None

    def _init_chunker(self) -> None:
        """Lazy-init so import errors surface only when chunking is actually used."""
        if self._chunker is not None:
            return
        from docling.chunking import HybridChunker
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
        self._chunker = HybridChunker(
            tokenizer=tokenizer,  # type: ignore[arg-type]
            max_tokens=self.config.max_tokens,  # type: ignore[call-arg]
            merge_peers=True,
        )
        self._tokenizer = tokenizer
        logger.info("HybridChunker initialised (max_tokens=%d)", self.config.max_tokens)

    def _count_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            return len(text.split())   # rough fallback
        return len(self._tokenizer.encode(text))

    def chunk_document(
        self,
        content: str,
        title: str,
        source: str,
        metadata: dict[str, Any] | None = None,
        docling_doc: Any | None = None,
    ) -> list[ChunkData]:
        """Chunk a document. Must be called from inside asyncio.to_thread."""
        if not content.strip():
            return []

        base_metadata: dict[str, Any] = {
            "title":   title,
            "source":  source,
            **(metadata or {}),
            **self.metadata_tags,
        }

        if docling_doc is None:
            logger.warning("No DoclingDocument for '%s' — using sliding window fallback", source)
            return self._simple_fallback_chunk(content, base_metadata)

        try:
            self._init_chunker()
            chunk_iter = self._chunker.chunk(dl_doc=docling_doc)
            chunks = list(chunk_iter)

            result: list[ChunkData] = []
            pos = 0
            for i, chunk in enumerate(chunks):
                # contextualize() prepends heading hierarchy — do not skip
                ctx_text: str = self._chunker.contextualize(chunk=chunk)
                tok_count = self._count_tokens(ctx_text)
                chunk_meta = {
                    **base_metadata,
                    "total_chunks": len(chunks),
                    "token_count":  tok_count,
                    "has_context":  True,
                    "chunk_method": "hybrid",
                }
                end = pos + len(ctx_text)
                result.append(ChunkData(
                    content=ctx_text.strip(),
                    metadata=chunk_meta,
                    chunk_index=i,
                    token_count=tok_count,
                    start_char=pos,
                    end_char=end,
                    corpus_id=self.corpus_id,
                    tenant_id=self.tenant_id,
                ))
                pos = end

            logger.info("Created %d chunks via HybridChunker for '%s'", len(result), source)
            return result

        except Exception as exc:
            logger.error("HybridChunker failed for '%s': %s — falling back", source, exc)
            return self._simple_fallback_chunk(content, base_metadata)

    def _simple_fallback_chunk(
        self, content: str, base_metadata: dict[str, Any]
    ) -> list[ChunkData]:
        """Sliding window fallback when HybridChunker is unavailable."""
        chunks: list[ChunkData] = []
        chunk_size    = self.config.chunk_size
        overlap       = self.config.chunk_overlap
        min_size      = self.config.min_chunk_size
        start         = 0
        chunk_index   = 0

        while start < len(content):
            end = start + chunk_size
            if end >= len(content):
                chunk_text = content[start:]
            else:
                # Try to end on a sentence boundary
                boundary = end
                for i in range(end, max(start + min_size, end - 200), -1):
                    if i < len(content) and content[i] in ".!?\n":
                        boundary = i + 1
                        break
                chunk_text = content[start:boundary]
                end = boundary

            text = chunk_text.strip()
            if text:
                tok = self._count_tokens(text)
                chunks.append(ChunkData(
                    content=text,
                    metadata={
                        **base_metadata,
                        "chunk_method": "simple_fallback",
                        "total_chunks": -1,    # updated below
                        "token_count":  tok,
                        "has_context":  False,
                    },
                    chunk_index=chunk_index,
                    token_count=tok,
                    start_char=start,
                    end_char=start + len(text),
                    corpus_id=self.corpus_id,
                    tenant_id=self.tenant_id,
                ))
                chunk_index += 1

            start = end - overlap

        for c in chunks:
            c.metadata["total_chunks"] = len(chunks)

        logger.info("Created %d chunks via fallback for '%s'", len(chunks), base_metadata.get("source", "?"))
        return chunks
