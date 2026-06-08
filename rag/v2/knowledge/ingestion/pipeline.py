"""Document ingestion pipeline orchestrator.

Ties together: DoclingProcessor → DoclingHybridChunker → Embedder →
asyncio.gather(vector_store.upsert_chunks, age_store.import_docling_graph).

Incremental mode (job.mode == "incremental"):
  1. Compute SHA-256 of file content.
  2. Check RedisCache fingerprint → skip if hit.
  3. On miss: check DB content_hash → skip if unchanged.
  4. If changed/new: delete old → ingest fresh → set fingerprint.

All file I/O, Docling conversion, and chunking are sync operations that run
inside asyncio.to_thread(). Embedding and DB writes are fully async.
"""

import asyncio
import glob
import hashlib
import logging
import os
import yaml
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from knowledge.bus.schemas import IngestJob
from knowledge.config.settings import Settings, load_settings
from knowledge.ingestion.chunker import DoclingHybridChunker
from knowledge.ingestion.docling_processor import (
    DoclingProcessor,
    _AUDIO_FORMATS,
    _PDF_FORMATS,
    _STRUCTURED_FORMATS,
)
from knowledge.ingestion.embedder import Embedder
from knowledge.ingestion.graph_extractor import extract_graph
from knowledge.ingestion.models import ChunkData, ChunkingConfig, IngestResult, IngestionResult

logger = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _extract_title(content: str, file_path: Path) -> str:
    for line in content.split("\n")[:10]:
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return file_path.stem


def _extract_metadata(content: str, path: Path, content_hash: str) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "file_path":     str(path),
        "file_size":     len(content),
        "ingestion_date": datetime.now(UTC).isoformat(),
        "content_hash":  content_hash,
        "line_count":    len(content.split("\n")),
        "word_count":    len(content.split()),
    }
    # Parse YAML frontmatter if present
    if content.startswith("---"):
        try:
            end = content.find("\n---\n", 4)
            if end != -1:
                fm = yaml.safe_load(content[4:end])
                if isinstance(fm, dict):
                    meta.update(fm)
        except Exception:
            pass
    return meta


def _find_document_files(source_path: Path) -> list[Path]:
    """Recursively find all supported document files under source_path."""
    all_exts = _PDF_FORMATS | _STRUCTURED_FORMATS | _AUDIO_FORMATS | {".txt"}
    files: list[Path] = []
    if source_path.is_file():
        if source_path.suffix.lower() in all_exts:
            files.append(source_path)
    else:
        for ext in all_exts:
            files.extend(source_path.rglob(f"*{ext}"))
    return sorted(files)


class DocumentIngestionPipeline:
    """Orchestrates ingestion of one document per run() call.

    Instantiate once per worker; all stores are shared across documents.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        vector_store: Any | None = None,
        age_store: Any | None = None,
        entity_index: Any | None = None,
        cache: Any | None = None,
        publisher: Any | None = None,
        corpus_config: Any | None = None,
    ) -> None:
        self._settings    = settings or load_settings()
        self._vector_store = vector_store
        self._age_store    = age_store
        self._entity_index = entity_index
        self._cache        = cache
        self._publisher    = publisher
        self._corpus_config = corpus_config

        chunking_config = ChunkingConfig(max_tokens=self._settings.chunk_max_tokens)
        self._processor = DoclingProcessor(settings=self._settings)
        self._embedder   = Embedder(settings=self._settings)

    def _make_chunker(self, corpus_id: str, tenant_id: str) -> DoclingHybridChunker:
        tags = {}
        if self._corpus_config:
            tags = self._corpus_config.metadata_tags
        return DoclingHybridChunker(
            config=ChunkingConfig(max_tokens=self._settings.chunk_max_tokens),
            corpus_id=corpus_id,
            tenant_id=tenant_id,
            metadata_tags=tags,
        )

    async def run(self, job: IngestJob) -> IngestResult:
        """Ingest one document (or all documents under a source folder).

        For folder-level jobs, calls _ingest_single_file for each file found.
        """
        t0 = asyncio.get_event_loop().time()
        source_path = Path(job.source_path or ".")
        files = await asyncio.to_thread(_find_document_files, source_path)

        if not files:
            logger.warning("No supported files found in '%s'", source_path)
            return IngestResult(job_id=job.job_id, errors=["No files found"])

        total_chunks = 0
        total_entities = 0
        errors: list[str] = []

        for file_path in files:
            try:
                result = await self._ingest_single_file(
                    file_path, job.job_id, job.corpus_id, job.tenant_id,
                    mode=job.mode,
                )
                total_chunks   += result.chunks_created
                total_entities += 0   # graph entities tracked separately
            except Exception as exc:
                logger.exception("Failed to ingest '%s': %s", file_path, exc)
                errors.append(f"{file_path.name}: {exc}")

        duration = asyncio.get_event_loop().time() - t0
        return IngestResult(
            job_id=job.job_id,
            chunks_ingested=total_chunks,
            duration_s=duration,
            errors=errors,
        )

    async def _ingest_single_file(
        self,
        file_path: Path,
        job_id: str,
        corpus_id: str,
        tenant_id: str,
        mode: str = "incremental",
    ) -> IngestionResult:
        t0 = asyncio.get_event_loop().time()

        # ── Incremental fingerprint check ────────────────────────────────────
        content_hash = await asyncio.to_thread(_sha256_file, file_path)

        if mode == "incremental":
            if self._cache and await self._cache.get_fingerprint(content_hash):
                logger.info("[SKIP] '%s' (fingerprint cache hit)", file_path.name)
                return IngestionResult(
                    document_id="", title=file_path.stem,
                    chunks_created=0,
                    processing_time_ms=(asyncio.get_event_loop().time() - t0) * 1000,
                )

            if self._vector_store:
                db_hash = await self._vector_store.get_document_hash(
                    str(file_path), corpus_id, tenant_id
                )
                if db_hash == content_hash:
                    logger.info("[SKIP] '%s' (DB hash match)", file_path.name)
                    if self._cache:
                        await self._cache.set_fingerprint(content_hash)
                    return IngestionResult(
                        document_id="", title=file_path.stem,
                        chunks_created=0,
                        processing_time_ms=(asyncio.get_event_loop().time() - t0) * 1000,
                    )

                if db_hash is not None:
                    logger.info("[UPDATE] '%s' (content changed)", file_path.name)
                    if self._vector_store:
                        await self._vector_store.delete_document_and_chunks(
                            str(file_path), corpus_id, tenant_id
                        )

        # ── Docling conversion (async.to_thread inside processor) ────────────
        conversion = await self._processor.process(file_path)
        title       = await asyncio.to_thread(_extract_title, conversion.markdown, file_path)
        metadata    = await asyncio.to_thread(_extract_metadata, conversion.markdown, file_path, content_hash)

        # ── Save document record ─────────────────────────────────────────────
        document_id = ""
        if self._vector_store:
            document_id = await self._vector_store.save_document(
                title=title,
                source=str(file_path),
                corpus_id=corpus_id,
                tenant_id=tenant_id,
                content=conversion.markdown,
                metadata=metadata,
            )

        # ── Parallel: chunk+embed+upsert  AND  graph extraction ──────────────
        chunker = self._make_chunker(corpus_id, tenant_id)

        async def _chunker_task() -> int:
            raw_chunks: list[ChunkData] = await asyncio.to_thread(
                chunker.chunk_document,
                conversion.markdown,
                title,
                str(file_path),
                metadata,
                conversion.docling_doc,
            )
            if not raw_chunks:
                return 0
            embedded = await self._embedder.embed_batch(raw_chunks)
            # Convert ChunkData list to dicts for vector_store
            chunk_dicts = [
                {
                    "content":     c.content,
                    "embedding":   c.metadata.pop("embedding", None),
                    "chunk_index": c.chunk_index,
                    "token_count": c.token_count,
                    "metadata":    c.metadata,
                }
                for c in embedded
            ]
            if self._vector_store and document_id:
                await self._vector_store.upsert_chunks(
                    chunk_dicts, document_id, corpus_id, tenant_id
                )
            return len(embedded)

        async def _graph_task() -> int:
            if not (self._corpus_config and self._corpus_config.enable_graph_extraction):
                return 0
            context = await extract_graph(file_path, self._corpus_config, self._settings)
            if context is None:
                metadata["graph_extraction_failed"] = True
                return 0
            if self._age_store and document_id:
                node_count, _ = await self._age_store.import_docling_graph(
                    context, corpus_id, tenant_id, document_id
                )
                return node_count
            return 0

        chunks_created, _ = await asyncio.gather(_chunker_task(), _graph_task())

        # ── Set fingerprint cache ────────────────────────────────────────────
        if self._cache:
            await self._cache.set_fingerprint(content_hash)

        duration_ms = (asyncio.get_event_loop().time() - t0) * 1000
        logger.info(
            "Ingested '%s': %d chunks in %.0fms",
            file_path.name, chunks_created, duration_ms,
        )

        return IngestionResult(
            document_id=document_id,
            title=title,
            chunks_created=chunks_created,
            processing_time_ms=duration_ms,
        )
