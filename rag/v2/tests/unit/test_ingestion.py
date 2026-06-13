"""Unit tests for the ingestion pipeline components.

No external services, no Docling, no docling-graph required.
Docling-dependent classes are tested with mocked internals.
"""

import hashlib
from pathlib import Path
from unittest import mock

import pytest

from knowledge.ingestion.chunker import DoclingHybridChunker
from knowledge.ingestion.embedder import _L1_MAX, Embedder
from knowledge.ingestion.models import (
    ChunkData,
    ChunkingConfig,
    Citation,
    IngestResult,
    SearchResult,
)
from knowledge.ingestion.pipeline import (
    DocumentIngestionPipeline,
    _extract_metadata,
    _extract_title,
    _find_document_files,
    _sha256_file,
)

# ── Models ────────────────────────────────────────────────────────────────────

class TestModels:
    def test_chunk_data_defaults(self) -> None:
        c = ChunkData(content="hello")
        assert c.chunk_index == 0
        assert c.corpus_id == ""
        assert c.metadata == {}

    def test_search_result_confidence_none_by_default(self) -> None:
        import uuid
        sr = SearchResult(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            document_title="T",
            document_source="s",
            content="c",
            raw_score=0.9,
            raw_score_type="rrf",
        )
        assert sr.confidence is None

    def test_citation_relevance_score(self) -> None:
        import uuid
        cit = Citation(
            chunk_id=uuid.uuid4(),
            document_title="T",
            document_source="s",
            relevance_score=0.87,
            excerpt="hello world",
        )
        assert cit.relevance_score == pytest.approx(0.87)

    def test_ingest_result_defaults(self) -> None:
        r = IngestResult(job_id="j1")
        assert r.chunks_ingested == 0
        assert r.skipped is False
        assert r.errors == []


# ── Embedder ──────────────────────────────────────────────────────────────────

def _make_embedder() -> Embedder:
    with mock.patch.dict("os.environ", {
        "DATABASE_URL":     "postgresql://x:x@localhost/x",
        "AGE_DATABASE_URL": "postgresql://x:x@localhost/x",
    }, clear=True):
        from knowledge.config.settings import Settings
        s = Settings(_env_file=None)  # type: ignore[call-arg]
    return Embedder(settings=s)


class TestEmbedder:
    @pytest.mark.asyncio
    async def test_embed_calls_api_on_miss(self) -> None:
        emb = _make_embedder()
        fake_vector = [0.1, 0.2, 0.3]

        mock_response = mock.MagicMock()
        mock_response.data = [mock.MagicMock(embedding=fake_vector)]

        with mock.patch("openai.AsyncOpenAI") as mock_openai:
            instance = mock_openai.return_value
            instance.embeddings.create = mock.AsyncMock(return_value=mock_response)
            emb._client = instance

            result = await emb.embed("hello world")

        assert result == fake_vector

    @pytest.mark.asyncio
    async def test_l1_cache_hit_skips_api(self) -> None:
        emb = _make_embedder()
        emb._cache["cached text"] = [1.0, 2.0, 3.0]

        with mock.patch("openai.AsyncOpenAI") as mock_openai:
            instance = mock_openai.return_value
            instance.embeddings.create = mock.AsyncMock()
            emb._client = instance

            result = await emb.embed("cached text")
            instance.embeddings.create.assert_not_called()

        assert result == [1.0, 2.0, 3.0]

    @pytest.mark.asyncio
    async def test_cache_eviction_at_max(self) -> None:
        emb = _make_embedder()
        # Fill cache to max
        for i in range(_L1_MAX):
            emb._cache[f"key_{i}"] = [float(i)]
        assert len(emb._cache) == _L1_MAX

        # One more set should evict oldest
        emb._cache_set("new_key", [99.0])
        assert len(emb._cache) == _L1_MAX
        assert "key_0" not in emb._cache   # oldest evicted
        assert "new_key" in emb._cache

    @pytest.mark.asyncio
    async def test_embed_batch_populates_metadata(self) -> None:
        emb = _make_embedder()
        chunks = [
            ChunkData(content="chunk one"),
            ChunkData(content="chunk two"),
        ]
        emb._cache["chunk one"] = [1.0, 2.0]
        emb._cache["chunk two"] = [3.0, 4.0]

        result = await emb.embed_batch(chunks)
        assert result[0].metadata["embedding"] == [1.0, 2.0]
        assert result[1].metadata["embedding"] == [3.0, 4.0]

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self) -> None:
        emb = _make_embedder()

        class FakeRateLimitError(Exception):
            pass
        FakeRateLimitError.__name__ = "RateLimitError"

        call_count = 0

        async def _fake_create(**_: object) -> object:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise FakeRateLimitError("rate limited")
            resp = mock.MagicMock()
            resp.data = [mock.MagicMock(embedding=[5.0])]
            return resp

        emb._client = mock.MagicMock()
        emb._client.embeddings.create = _fake_create

        with mock.patch("knowledge.ingestion.embedder.exponential_backoff", return_value=0.0):
            result = await emb.embed("test retry")

        assert result == [5.0]
        assert call_count == 3


# ── Chunker ───────────────────────────────────────────────────────────────────

class TestDoclingHybridChunker:
    def _make_chunker(self, **kwargs: object) -> DoclingHybridChunker:
        config = ChunkingConfig(max_tokens=512, chunk_size=500)
        return DoclingHybridChunker(config=config, corpus_id="c1", tenant_id="t1", **kwargs)

    def test_fallback_on_no_docling_doc(self) -> None:
        chunker = self._make_chunker()
        chunks = chunker.chunk_document(
            content="Hello world. " * 50,
            title="Test",
            source="test.md",
        )
        assert len(chunks) > 0
        assert all(c.corpus_id == "c1" for c in chunks)
        assert all(c.metadata["chunk_method"] == "simple_fallback" for c in chunks)

    def test_empty_content_returns_empty(self) -> None:
        chunker = self._make_chunker()
        assert chunker.chunk_document("", "T", "s") == []

    def test_whitespace_only_returns_empty(self) -> None:
        chunker = self._make_chunker()
        assert chunker.chunk_document("   \n  ", "T", "s") == []

    def test_metadata_tags_injected(self) -> None:
        chunker = self._make_chunker(metadata_tags={"env": "prod"})
        chunks = chunker.chunk_document("Hello " * 100, "T", "s")
        assert all(c.metadata.get("env") == "prod" for c in chunks)

    def test_total_chunks_updated_in_fallback(self) -> None:
        chunker = self._make_chunker()
        chunks = chunker.chunk_document("word " * 500, "T", "s")
        total = len(chunks)
        assert all(c.metadata["total_chunks"] == total for c in chunks)

    def test_chunk_index_sequential(self) -> None:
        chunker = self._make_chunker()
        chunks = chunker.chunk_document("a " * 1000, "T", "s")
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_contextualize_called_with_docling_doc(self) -> None:
        chunker = self._make_chunker()
        chunker._init_chunker = mock.MagicMock()  # prevent real init

        mock_chunk = mock.MagicMock()
        mock_chunk_iter = [mock_chunk]

        mock_hc = mock.MagicMock()
        mock_hc.chunk.return_value = iter(mock_chunk_iter)
        mock_hc.contextualize.return_value = "Heading: Content here."
        chunker._chunker = mock_hc
        chunker._tokenizer = mock.MagicMock()
        chunker._tokenizer.encode.return_value = [1, 2, 3]

        mock_doc = mock.MagicMock()
        chunks = chunker.chunk_document("raw content", "T", "s", docling_doc=mock_doc)

        mock_hc.chunk.assert_called_once_with(dl_doc=mock_doc)
        mock_hc.contextualize.assert_called_once_with(chunk=mock_chunk)
        assert len(chunks) == 1
        assert chunks[0].content == "Heading: Content here."
        assert chunks[0].metadata["has_context"] is True


# ── Pipeline helpers ──────────────────────────────────────────────────────────

class TestPipelineHelpers:
    def test_extract_title_from_h1(self) -> None:
        content = "# My Document\n\nSome content here."
        assert _extract_title(content, Path("file.md")) == "My Document"

    def test_extract_title_falls_back_to_stem(self) -> None:
        content = "No heading here, just plain text."
        assert _extract_title(content, Path("my_report.pdf")) == "my_report"

    def test_extract_metadata_includes_hash(self, tmp_path: Path) -> None:
        p = tmp_path / "doc.md"
        p.write_text("content", encoding="utf-8")
        meta = _extract_metadata("content", p, "abc123")
        assert meta["content_hash"] == "abc123"
        assert meta["word_count"] == 1

    def test_extract_metadata_parses_frontmatter(self, tmp_path: Path) -> None:
        content = "---\nauthor: Alice\ntags: [a, b]\n---\nBody text."
        p = tmp_path / "doc.md"
        p.write_text(content, encoding="utf-8")
        meta = _extract_metadata(content, p, "hash")
        assert meta.get("author") == "Alice"

    def test_sha256_file(self, tmp_path: Path) -> None:
        p = tmp_path / "file.txt"
        p.write_bytes(b"hello")
        expected = hashlib.sha256(b"hello").hexdigest()
        assert _sha256_file(p) == expected

    def test_find_document_files_single_file(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.pdf"
        f.touch()
        result = _find_document_files(f)
        assert result == [f]

    def test_find_document_files_directory(self, tmp_path: Path) -> None:
        (tmp_path / "a.pdf").touch()
        (tmp_path / "b.md").touch()
        (tmp_path / "ignore.xyz").touch()
        result = _find_document_files(tmp_path)
        names = {p.name for p in result}
        assert "a.pdf" in names
        assert "b.md" in names
        assert "ignore.xyz" not in names


# ── Pipeline orchestrator ─────────────────────────────────────────────────────

class TestDocumentIngestionPipeline:
    def _make_pipeline(self) -> DocumentIngestionPipeline:
        with mock.patch.dict("os.environ", {
            "DATABASE_URL":     "postgresql://x:x@localhost/x",
            "AGE_DATABASE_URL": "postgresql://x:x@localhost/x",
        }, clear=True):
            from knowledge.config.settings import Settings
            s = Settings(_env_file=None)  # type: ignore[call-arg]
        return DocumentIngestionPipeline(settings=s)

    @pytest.mark.asyncio
    async def test_incremental_skip_on_fingerprint_cache_hit(self, tmp_path: Path) -> None:
        pipeline = self._make_pipeline()
        f = tmp_path / "doc.md"
        f.write_text("hello world", encoding="utf-8")
        _sha256_file(f)  # compute hash to warm any internal state

        mock_cache = mock.AsyncMock()
        mock_cache.get_fingerprint.return_value = True   # cache HIT
        pipeline._cache = mock_cache

        from knowledge.bus.schemas import IngestJob
        job = IngestJob(
            tenant_id="t1", corpus_id="c1",
            source_path=str(f), mode="incremental",
        )
        result = await pipeline.run(job)
        assert result.chunks_ingested == 0   # skipped

    @pytest.mark.asyncio
    async def test_no_files_returns_error(self, tmp_path: Path) -> None:
        pipeline = self._make_pipeline()
        from knowledge.bus.schemas import IngestJob
        job = IngestJob(
            tenant_id="t1", corpus_id="c1",
            source_path=str(tmp_path / "nonexistent"),
        )
        result = await pipeline.run(job)
        assert result.errors

    @pytest.mark.asyncio
    async def test_pipeline_calls_vector_store_upsert(self, tmp_path: Path) -> None:
        pipeline = self._make_pipeline()
        f = tmp_path / "doc.md"
        f.write_text("# Hello\n\nThis is test content. " * 10, encoding="utf-8")

        mock_vs = mock.AsyncMock()
        mock_vs.get_document_hash.return_value = None   # new document
        mock_vs.save_document.return_value = "doc-uuid-123"
        mock_vs.upsert_chunks.return_value = None
        pipeline._vector_store = mock_vs

        mock_cache = mock.AsyncMock()
        mock_cache.get_fingerprint.return_value = False
        mock_cache.set_fingerprint.return_value = None
        pipeline._cache = mock_cache

        # Mock embedder to return fake vectors without API call
        pipeline._embedder._cache["chunk_content"] = [0.1, 0.2]

        async def _fake_embed(text: str) -> list[float]:
            return [0.1] * 768

        pipeline._embedder.embed = _fake_embed  # type: ignore[method-assign]

        from knowledge.bus.schemas import IngestJob
        job = IngestJob(
            tenant_id="t1", corpus_id="c1",
            source_path=str(f), mode="incremental",
        )
        result = await pipeline.run(job)

        mock_vs.save_document.assert_called_once()
        mock_vs.upsert_chunks.assert_called()
        assert result.chunks_ingested > 0
