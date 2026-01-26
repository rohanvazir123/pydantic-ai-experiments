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

"""Tests for PDF Question Generator with PostgreSQL/pgvector storage.

These tests verify:
1. PDFQuestionStore - PostgreSQL storage for questions and chunks
2. Question/chunk storage with embeddings
3. Semantic, text, and hybrid search using pgvector
4. PDF document management

Requirements:
    - PostgreSQL/Neon with pgvector extension
    - DATABASE_URL in .env
    - Ollama running for embeddings
"""

import logging
import pytest
import pytest_asyncio

from rag.config.settings import load_settings
from rag.ingestion.processors.pdf_question_store import PDFQuestionStore

logger = logging.getLogger(__name__)


class TestPDFQuestionStoreBasic:
    """Test basic PDFQuestionStore initialization."""

    @pytest.fixture
    def settings(self):
        """Load settings fixture."""
        return load_settings()

    @pytest.fixture
    def store(self):
        """Create PDFQuestionStore fixture."""
        return PDFQuestionStore()

    def test_store_initialization(self, store):
        """Test that store initializes with correct settings."""
        assert store.settings is not None
        assert store.pool is None  # Not connected yet
        assert store._initialized is False

    def test_store_has_table_names(self, store):
        """Test that store has table names configured."""
        assert store.pdf_documents_table == "pdf_documents"
        assert store.pdf_questions_table == "pdf_questions"
        assert store.pdf_chunks_table == "pdf_chunks"


@pytest.mark.asyncio
class TestPDFQuestionStoreConnection:
    """Test PDFQuestionStore connection - requires PostgreSQL."""

    @pytest_asyncio.fixture
    async def connected_store(self):
        """Create and initialize a connected PDFQuestionStore."""
        store = PDFQuestionStore()
        await store.initialize()
        yield store
        await store.close()

    async def test_connection(self, connected_store):
        """Test that connection is established."""
        assert connected_store._initialized is True
        assert connected_store.pool is not None

    async def test_tables_created(self, connected_store):
        """Test that required tables are created."""
        async with connected_store.pool.acquire() as conn:
            # Check pdf_documents table
            docs_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'pdf_documents'
                )
            """)
            assert docs_exists, "pdf_documents table should exist"

            # Check pdf_questions table
            questions_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'pdf_questions'
                )
            """)
            assert questions_exists, "pdf_questions table should exist"

            # Check pdf_chunks table
            chunks_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'pdf_chunks'
                )
            """)
            assert chunks_exists, "pdf_chunks table should exist"

    async def test_get_statistics(self, connected_store):
        """Test getting statistics."""
        stats = await connected_store.get_statistics()

        assert "pdf_documents" in stats
        assert "questions" in stats
        assert "chunks" in stats
        assert isinstance(stats["pdf_documents"], int)
        assert isinstance(stats["questions"], int)
        assert isinstance(stats["chunks"], int)


@pytest.mark.asyncio
class TestPDFQuestionStoreCRUD:
    """Test PDFQuestionStore CRUD operations."""

    @pytest_asyncio.fixture
    async def connected_store(self):
        """Create and initialize a connected PDFQuestionStore."""
        store = PDFQuestionStore()
        await store.initialize()
        yield store
        # Cleanup test data
        await store.delete_pdf_document("test/crud_test.pdf")
        await store.close()

    async def test_save_pdf_result(self, connected_store):
        """Test saving a PDF result."""
        doc_id = await connected_store.save_pdf_result(
            pdf_path="test/crud_test.pdf",
            title="CRUD Test PDF",
            num_pages=10,
            num_text_chunks=25,
            num_tables=3,
            num_equations=5,
            num_images=2,
            content_summary="Test document for CRUD operations",
            questions=[
                {"question": "What is the main topic?", "difficulty": "easy"},
                {"question": "Explain the methodology.", "difficulty": "medium"},
            ],
            chunks=[
                {"chunk_id": "c1", "content": "This is the first chunk of content."},
                {"chunk_id": "c2", "content": "This is the second chunk with more details."},
            ],
            metadata={"test": True},
        )

        assert doc_id is not None
        assert len(doc_id) > 0
        logger.info(f"Saved PDF with ID: {doc_id}")

    async def test_get_pdf_document(self, connected_store):
        """Test getting a PDF document."""
        # First save a document
        await connected_store.save_pdf_result(
            pdf_path="test/crud_test.pdf",
            title="Get Test PDF",
            num_pages=5,
        )

        # Then retrieve it
        doc = await connected_store.get_pdf_document("test/crud_test.pdf")

        assert doc is not None
        assert doc["title"] == "Get Test PDF"
        assert doc["num_pages"] == 5
        assert doc["pdf_path"] == "test/crud_test.pdf"

    async def test_get_questions_for_pdf(self, connected_store):
        """Test getting questions for a PDF."""
        # Save document with questions
        await connected_store.save_pdf_result(
            pdf_path="test/crud_test.pdf",
            title="Questions Test PDF",
            questions=[
                {"question": "Question 1?", "difficulty": "easy"},
                {"question": "Question 2?", "difficulty": "hard"},
            ],
        )

        # Get questions
        questions = await connected_store.get_questions_for_pdf("test/crud_test.pdf")

        assert len(questions) == 2
        assert questions[0]["question"] == "Question 1?"
        assert questions[1]["question"] == "Question 2?"

    async def test_delete_pdf_document(self, connected_store):
        """Test deleting a PDF document."""
        # Save a document
        await connected_store.save_pdf_result(
            pdf_path="test/crud_test.pdf",
            title="Delete Test PDF",
        )

        # Verify it exists
        doc = await connected_store.get_pdf_document("test/crud_test.pdf")
        assert doc is not None

        # Delete it
        deleted = await connected_store.delete_pdf_document("test/crud_test.pdf")
        assert deleted is True

        # Verify it's gone
        doc = await connected_store.get_pdf_document("test/crud_test.pdf")
        assert doc is None

    async def test_replace_existing_document(self, connected_store):
        """Test that saving to same path replaces existing document."""
        # Save first version
        await connected_store.save_pdf_result(
            pdf_path="test/crud_test.pdf",
            title="Version 1",
            num_pages=5,
        )

        # Save second version
        await connected_store.save_pdf_result(
            pdf_path="test/crud_test.pdf",
            title="Version 2",
            num_pages=10,
        )

        # Should get version 2
        doc = await connected_store.get_pdf_document("test/crud_test.pdf")
        assert doc["title"] == "Version 2"
        assert doc["num_pages"] == 10


@pytest.mark.asyncio
class TestPDFQuestionStoreSearch:
    """Test PDFQuestionStore search operations with pgvector."""

    @pytest_asyncio.fixture
    async def store_with_data(self):
        """Create store with test data."""
        store = PDFQuestionStore()
        await store.initialize()

        # Add test data
        await store.save_pdf_result(
            pdf_path="test/search_test.pdf",
            title="Machine Learning Basics",
            num_pages=20,
            questions=[
                {"question": "What is supervised learning?", "difficulty": "easy"},
                {"question": "How do neural networks work?", "difficulty": "medium"},
                {"question": "Explain backpropagation algorithm.", "difficulty": "hard"},
            ],
            chunks=[
                {"chunk_id": "c1", "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data."},
                {"chunk_id": "c2", "content": "Neural networks are computational models inspired by the human brain structure."},
                {"chunk_id": "c3", "content": "Backpropagation is an algorithm used to train neural networks by computing gradients."},
            ],
        )

        yield store

        # Cleanup
        await store.delete_pdf_document("test/search_test.pdf")
        await store.close()

    async def test_semantic_search_questions(self, store_with_data):
        """Test semantic search for questions."""
        results = await store_with_data.search_questions(
            "neural network architecture",
            limit=3,
            search_type="semantic",
        )

        assert len(results) > 0
        assert all("question" in r for r in results)
        assert all("similarity" in r for r in results)
        logger.info(f"Semantic search returned {len(results)} results")
        for r in results:
            logger.info(f"  {r['question']} (sim: {r['similarity']:.4f})")

    async def test_text_search_questions(self, store_with_data):
        """Test text search for questions."""
        results = await store_with_data.search_questions(
            "neural networks",
            limit=3,
            search_type="text",
        )

        assert isinstance(results, list)
        if results:
            assert all("question" in r for r in results)
            logger.info(f"Text search returned {len(results)} results")

    async def test_hybrid_search_questions(self, store_with_data):
        """Test hybrid search for questions."""
        results = await store_with_data.search_questions(
            "machine learning algorithms",
            limit=3,
            search_type="hybrid",
        )

        assert len(results) > 0
        assert all("question" in r for r in results)
        assert all("similarity" in r for r in results)
        logger.info(f"Hybrid search returned {len(results)} results")

    async def test_semantic_search_chunks(self, store_with_data):
        """Test semantic search for chunks."""
        results = await store_with_data.search_chunks(
            "artificial intelligence",
            limit=3,
            search_type="semantic",
        )

        assert len(results) > 0
        assert all("content" in r for r in results)
        assert all("similarity" in r for r in results)
        logger.info(f"Chunk semantic search returned {len(results)} results")
        for r in results:
            logger.info(f"  {r['content'][:50]}... (sim: {r['similarity']:.4f})")

    async def test_hybrid_search_chunks(self, store_with_data):
        """Test hybrid search for chunks."""
        results = await store_with_data.search_chunks(
            "brain inspired computing",
            limit=3,
            search_type="hybrid",
        )

        assert len(results) > 0
        assert all("content" in r for r in results)
        logger.info(f"Chunk hybrid search returned {len(results)} results")

    async def test_search_returns_pdf_metadata(self, store_with_data):
        """Test that search results include PDF metadata."""
        results = await store_with_data.search_questions(
            "learning",
            limit=1,
        )

        assert len(results) > 0
        result = results[0]
        assert "pdf_title" in result
        assert "pdf_path" in result
        assert result["pdf_title"] == "Machine Learning Basics"


class TestPDFQuestionGeneratorModels:
    """Test PDF question generator data models."""

    def test_processing_result_dataclass(self):
        """Test ProcessingResult dataclass."""
        from rag.ingestion.processors.pdf_question_generator import ProcessingResult

        result = ProcessingResult(
            pdf_path="test.pdf",
            title="Test",
            num_pages=5,
        )

        assert result.pdf_path == "test.pdf"
        assert result.title == "Test"
        assert result.num_pages == 5
        assert result.questions == []
        assert result.error is None

    def test_chunk_context_dataclass(self):
        """Test ChunkContext dataclass."""
        from rag.ingestion.processors.pdf_question_generator import ChunkContext

        chunk = ChunkContext(
            chunk_id="c1",
            content="Test content",
            entity_name="TestEntity",
            entity_type="concept",
            page_idx=1,
            content_type="text",
        )

        assert chunk.chunk_id == "c1"
        assert chunk.content == "Test content"
        assert chunk.entity_name == "TestEntity"
        assert chunk.content_type == "text"

    def test_format_chunks_as_context(self):
        """Test format_chunks_as_context function."""
        from rag.ingestion.processors.pdf_question_generator import (
            ChunkContext,
            format_chunks_as_context,
        )

        chunks = [
            ChunkContext(chunk_id="c1", content="First chunk content", page_idx=0),
            ChunkContext(chunk_id="c2", content="Second chunk content", page_idx=1),
        ]

        context = format_chunks_as_context(chunks, max_chars=1000)

        assert "[chunk_id=c1]" in context
        assert "[chunk_id=c2]" in context
        assert "First chunk content" in context
        assert "Second chunk content" in context


@pytest.mark.asyncio
class TestPDFQuestionStoreIndexes:
    """Test that pgvector indexes are created correctly."""

    @pytest_asyncio.fixture
    async def connected_store(self):
        """Create and initialize a connected PDFQuestionStore."""
        store = PDFQuestionStore()
        await store.initialize()
        yield store
        await store.close()

    async def test_questions_vector_index_exists(self, connected_store):
        """Test that vector index exists on questions table."""
        async with connected_store.pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM pg_indexes
                    WHERE tablename = 'pdf_questions'
                    AND indexname = 'pdf_questions_embedding_idx'
                )
            """)
            assert result, "Vector index should exist on pdf_questions"

    async def test_chunks_vector_index_exists(self, connected_store):
        """Test that vector index exists on chunks table."""
        async with connected_store.pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM pg_indexes
                    WHERE tablename = 'pdf_chunks'
                    AND indexname = 'pdf_chunks_embedding_idx'
                )
            """)
            assert result, "Vector index should exist on pdf_chunks"

    async def test_questions_text_index_exists(self, connected_store):
        """Test that text search index exists on questions table."""
        async with connected_store.pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM pg_indexes
                    WHERE tablename = 'pdf_questions'
                    AND indexname = 'pdf_questions_tsv_idx'
                )
            """)
            assert result, "Text search index should exist on pdf_questions"

    async def test_chunks_text_index_exists(self, connected_store):
        """Test that text search index exists on chunks table."""
        async with connected_store.pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM pg_indexes
                    WHERE tablename = 'pdf_chunks'
                    AND indexname = 'pdf_chunks_tsv_idx'
                )
            """)
            assert result, "Text search index should exist on pdf_chunks"
