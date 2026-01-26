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
PostgreSQL storage for PDF Question Generator.

This module provides functions to store and retrieve PDF question generator
results using PostgreSQL with pgvector for semantic search.

Usage:
    from rag.ingestion.processors.pdf_question_store import PDFQuestionStore

    store = PDFQuestionStore()
    await store.initialize()

    # Store processing result
    await store.save_pdf_result(result)

    # Search for similar questions
    results = await store.search_questions("What is machine learning?", limit=5)

    await store.close()
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any

from rag.config.settings import load_settings
from rag.ingestion.embedder import EmbeddingGenerator
from rag.ingestion.models import ChunkData, SearchResult

logger = logging.getLogger(__name__)


class PDFQuestionStore:
    """PostgreSQL store for PDF question generator results with pgvector support."""

    def __init__(self):
        """Initialize the store."""
        self.settings = load_settings()
        self.pool = None
        self._initialized = False
        self.embedder = EmbeddingGenerator()

        # Table names
        self.pdf_documents_table = "pdf_documents"
        self.pdf_questions_table = "pdf_questions"
        self.pdf_chunks_table = "pdf_chunks"

    async def initialize(self) -> None:
        """Initialize PostgreSQL connection and create tables."""
        if self._initialized:
            return

        import asyncpg
        from pgvector.asyncpg import register_vector

        try:
            self.pool = await asyncpg.create_pool(
                self.settings.database_url,
                min_size=1,
                max_size=10,
                command_timeout=60,
            )

            async with self.pool.acquire() as conn:
                # Enable pgvector extension
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            async with self.pool.acquire() as conn:
                await register_vector(conn)

                # Create PDF documents table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.pdf_documents_table} (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        title TEXT NOT NULL,
                        pdf_path TEXT NOT NULL UNIQUE,
                        num_pages INTEGER DEFAULT 0,
                        num_text_chunks INTEGER DEFAULT 0,
                        num_tables INTEGER DEFAULT 0,
                        num_equations INTEGER DEFAULT 0,
                        num_images INTEGER DEFAULT 0,
                        content_summary TEXT,
                        metadata JSONB DEFAULT '{{}}',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)

                # Create PDF questions table with vector column
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.pdf_questions_table} (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        pdf_document_id UUID NOT NULL REFERENCES {self.pdf_documents_table}(id) ON DELETE CASCADE,
                        question TEXT NOT NULL,
                        question_embedding vector({self.settings.embedding_dimension}),
                        difficulty TEXT,
                        supported_by TEXT[],
                        metadata JSONB DEFAULT '{{}}',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        question_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', question)) STORED
                    )
                """)

                # Create PDF chunks table with vector column
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.pdf_chunks_table} (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        pdf_document_id UUID NOT NULL REFERENCES {self.pdf_documents_table}(id) ON DELETE CASCADE,
                        chunk_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        content_embedding vector({self.settings.embedding_dimension}),
                        content_type TEXT DEFAULT 'text',
                        page_idx INTEGER DEFAULT 0,
                        entity_name TEXT,
                        entity_type TEXT,
                        metadata JSONB DEFAULT '{{}}',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
                    )
                """)

                # Create indexes
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS pdf_questions_embedding_idx
                    ON {self.pdf_questions_table}
                    USING ivfflat (question_embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)

                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS pdf_chunks_embedding_idx
                    ON {self.pdf_chunks_table}
                    USING ivfflat (content_embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)

                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS pdf_questions_tsv_idx
                    ON {self.pdf_questions_table}
                    USING GIN(question_tsv)
                """)

                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS pdf_chunks_tsv_idx
                    ON {self.pdf_chunks_table}
                    USING GIN(content_tsv)
                """)

            logger.info("PDF Question Store initialized with PostgreSQL/pgvector")
            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize PDF Question Store: {e}")
            raise

    async def close(self) -> None:
        """Close PostgreSQL connection."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            self._initialized = False
            logger.info("PDF Question Store closed")

    async def save_pdf_result(
        self,
        pdf_path: str,
        title: str,
        num_pages: int = 0,
        num_text_chunks: int = 0,
        num_tables: int = 0,
        num_equations: int = 0,
        num_images: int = 0,
        content_summary: str = "",
        questions: list[dict] | None = None,
        chunks: list[dict] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Save PDF processing result to PostgreSQL.

        Args:
            pdf_path: Path to the PDF file
            title: Document title
            num_pages: Number of pages
            num_text_chunks: Number of text chunks
            num_tables: Number of tables
            num_equations: Number of equations
            num_images: Number of images
            content_summary: Summary of content
            questions: List of generated questions
            chunks: List of content chunks
            metadata: Additional metadata

        Returns:
            Document ID (UUID string)
        """
        await self.initialize()

        from pgvector.asyncpg import register_vector

        async with self.pool.acquire() as conn:
            await register_vector(conn)

            # Check if document already exists
            existing = await conn.fetchval(
                f"SELECT id FROM {self.pdf_documents_table} WHERE pdf_path = $1",
                pdf_path
            )

            if existing:
                # Delete existing document (cascades to questions and chunks)
                await conn.execute(
                    f"DELETE FROM {self.pdf_documents_table} WHERE id = $1",
                    existing
                )
                logger.info(f"Replaced existing PDF document: {pdf_path}")

            # Insert PDF document
            doc_id = await conn.fetchval(
                f"""
                INSERT INTO {self.pdf_documents_table}
                (title, pdf_path, num_pages, num_text_chunks, num_tables,
                 num_equations, num_images, content_summary, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id
                """,
                title,
                pdf_path,
                num_pages,
                num_text_chunks,
                num_tables,
                num_equations,
                num_images,
                content_summary,
                json.dumps(metadata or {}),
            )

            logger.info(f"Saved PDF document: {title} (ID: {doc_id})")

            # Insert questions with embeddings
            if questions:
                for q in questions:
                    question_text = q.get("question", str(q)) if isinstance(q, dict) else str(q)
                    difficulty = q.get("difficulty", "") if isinstance(q, dict) else ""
                    supported_by = q.get("supported_by", []) if isinstance(q, dict) else []

                    # Generate embedding for question
                    embedding = await self.embedder.embed_query(question_text)

                    await conn.execute(
                        f"""
                        INSERT INTO {self.pdf_questions_table}
                        (pdf_document_id, question, question_embedding, difficulty, supported_by, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        """,
                        doc_id,
                        question_text,
                        embedding,
                        difficulty,
                        supported_by,
                        json.dumps({}),
                    )

                logger.info(f"Saved {len(questions)} questions for document {doc_id}")

            # Insert chunks with embeddings
            if chunks:
                for chunk in chunks:
                    content = chunk.get("content", "")
                    if not content:
                        continue

                    # Generate embedding for chunk
                    embedding = await self.embedder.embed_query(content)

                    await conn.execute(
                        f"""
                        INSERT INTO {self.pdf_chunks_table}
                        (pdf_document_id, chunk_id, content, content_embedding,
                         content_type, page_idx, entity_name, entity_type, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        """,
                        doc_id,
                        chunk.get("chunk_id", f"c{chunks.index(chunk)+1}"),
                        content,
                        embedding,
                        chunk.get("content_type", "text"),
                        chunk.get("page_idx", 0),
                        chunk.get("entity_name", ""),
                        chunk.get("entity_type", ""),
                        json.dumps(chunk.get("metadata", {})),
                    )

                logger.info(f"Saved {len(chunks)} chunks for document {doc_id}")

            return str(doc_id)

    async def search_questions(
        self,
        query: str,
        limit: int = 10,
        search_type: str = "hybrid",
    ) -> list[dict]:
        """
        Search for similar questions using semantic or hybrid search.

        Args:
            query: Search query
            limit: Maximum number of results
            search_type: "semantic", "text", or "hybrid"

        Returns:
            List of matching questions with metadata
        """
        await self.initialize()

        from pgvector.asyncpg import register_vector

        async with self.pool.acquire() as conn:
            await register_vector(conn)

            # Set IVF probes for better recall
            await conn.execute("SET ivfflat.probes = 10")

            if search_type == "semantic":
                # Generate query embedding
                query_embedding = await self.embedder.embed_query(query)

                rows = await conn.fetch(
                    f"""
                    SELECT
                        q.id,
                        q.question,
                        q.difficulty,
                        q.supported_by,
                        d.title as pdf_title,
                        d.pdf_path,
                        1 - (q.question_embedding <=> $1::vector) as similarity
                    FROM {self.pdf_questions_table} q
                    JOIN {self.pdf_documents_table} d ON q.pdf_document_id = d.id
                    ORDER BY q.question_embedding <=> $1::vector
                    LIMIT $2
                    """,
                    query_embedding,
                    limit,
                )

            elif search_type == "text":
                rows = await conn.fetch(
                    f"""
                    SELECT
                        q.id,
                        q.question,
                        q.difficulty,
                        q.supported_by,
                        d.title as pdf_title,
                        d.pdf_path,
                        ts_rank(q.question_tsv, plainto_tsquery('english', $1)) as similarity
                    FROM {self.pdf_questions_table} q
                    JOIN {self.pdf_documents_table} d ON q.pdf_document_id = d.id
                    WHERE q.question_tsv @@ plainto_tsquery('english', $1)
                    ORDER BY similarity DESC
                    LIMIT $2
                    """,
                    query,
                    limit,
                )

            else:  # hybrid
                # Generate query embedding
                query_embedding = await self.embedder.embed_query(query)

                # Get semantic results
                semantic_rows = await conn.fetch(
                    f"""
                    SELECT
                        q.id,
                        q.question,
                        q.difficulty,
                        q.supported_by,
                        d.title as pdf_title,
                        d.pdf_path,
                        1 - (q.question_embedding <=> $1::vector) as similarity
                    FROM {self.pdf_questions_table} q
                    JOIN {self.pdf_documents_table} d ON q.pdf_document_id = d.id
                    ORDER BY q.question_embedding <=> $1::vector
                    LIMIT $2
                    """,
                    query_embedding,
                    limit * 2,
                )

                # Get text results
                text_rows = await conn.fetch(
                    f"""
                    SELECT
                        q.id,
                        q.question,
                        q.difficulty,
                        q.supported_by,
                        d.title as pdf_title,
                        d.pdf_path,
                        ts_rank(q.question_tsv, plainto_tsquery('english', $1)) as similarity
                    FROM {self.pdf_questions_table} q
                    JOIN {self.pdf_documents_table} d ON q.pdf_document_id = d.id
                    WHERE q.question_tsv @@ plainto_tsquery('english', $1)
                    ORDER BY similarity DESC
                    LIMIT $2
                    """,
                    query,
                    limit * 2,
                )

                # Merge with RRF
                rows = self._rrf_merge(semantic_rows, text_rows, limit)

            return [
                {
                    "id": str(row["id"]),
                    "question": row["question"],
                    "difficulty": row["difficulty"],
                    "supported_by": row["supported_by"],
                    "pdf_title": row["pdf_title"],
                    "pdf_path": row["pdf_path"],
                    "similarity": float(row["similarity"]),
                }
                for row in rows
            ]

    async def search_chunks(
        self,
        query: str,
        limit: int = 10,
        search_type: str = "hybrid",
    ) -> list[dict]:
        """
        Search for similar content chunks using semantic or hybrid search.

        Args:
            query: Search query
            limit: Maximum number of results
            search_type: "semantic", "text", or "hybrid"

        Returns:
            List of matching chunks with metadata
        """
        await self.initialize()

        from pgvector.asyncpg import register_vector

        async with self.pool.acquire() as conn:
            await register_vector(conn)

            # Set IVF probes for better recall
            await conn.execute("SET ivfflat.probes = 10")

            if search_type == "semantic":
                query_embedding = await self.embedder.embed_query(query)

                rows = await conn.fetch(
                    f"""
                    SELECT
                        c.id,
                        c.chunk_id,
                        c.content,
                        c.content_type,
                        c.page_idx,
                        c.entity_name,
                        d.title as pdf_title,
                        d.pdf_path,
                        1 - (c.content_embedding <=> $1::vector) as similarity
                    FROM {self.pdf_chunks_table} c
                    JOIN {self.pdf_documents_table} d ON c.pdf_document_id = d.id
                    ORDER BY c.content_embedding <=> $1::vector
                    LIMIT $2
                    """,
                    query_embedding,
                    limit,
                )

            elif search_type == "text":
                rows = await conn.fetch(
                    f"""
                    SELECT
                        c.id,
                        c.chunk_id,
                        c.content,
                        c.content_type,
                        c.page_idx,
                        c.entity_name,
                        d.title as pdf_title,
                        d.pdf_path,
                        ts_rank(c.content_tsv, plainto_tsquery('english', $1)) as similarity
                    FROM {self.pdf_chunks_table} c
                    JOIN {self.pdf_documents_table} d ON c.pdf_document_id = d.id
                    WHERE c.content_tsv @@ plainto_tsquery('english', $1)
                    ORDER BY similarity DESC
                    LIMIT $2
                    """,
                    query,
                    limit,
                )

            else:  # hybrid
                query_embedding = await self.embedder.embed_query(query)

                semantic_rows = await conn.fetch(
                    f"""
                    SELECT
                        c.id,
                        c.chunk_id,
                        c.content,
                        c.content_type,
                        c.page_idx,
                        c.entity_name,
                        d.title as pdf_title,
                        d.pdf_path,
                        1 - (c.content_embedding <=> $1::vector) as similarity
                    FROM {self.pdf_chunks_table} c
                    JOIN {self.pdf_documents_table} d ON c.pdf_document_id = d.id
                    ORDER BY c.content_embedding <=> $1::vector
                    LIMIT $2
                    """,
                    query_embedding,
                    limit * 2,
                )

                text_rows = await conn.fetch(
                    f"""
                    SELECT
                        c.id,
                        c.chunk_id,
                        c.content,
                        c.content_type,
                        c.page_idx,
                        c.entity_name,
                        d.title as pdf_title,
                        d.pdf_path,
                        ts_rank(c.content_tsv, plainto_tsquery('english', $1)) as similarity
                    FROM {self.pdf_chunks_table} c
                    JOIN {self.pdf_documents_table} d ON c.pdf_document_id = d.id
                    WHERE c.content_tsv @@ plainto_tsquery('english', $1)
                    ORDER BY similarity DESC
                    LIMIT $2
                    """,
                    query,
                    limit * 2,
                )

                rows = self._rrf_merge(semantic_rows, text_rows, limit)

            return [
                {
                    "id": str(row["id"]),
                    "chunk_id": row["chunk_id"],
                    "content": row["content"],
                    "content_type": row["content_type"],
                    "page_idx": row["page_idx"],
                    "entity_name": row["entity_name"],
                    "pdf_title": row["pdf_title"],
                    "pdf_path": row["pdf_path"],
                    "similarity": float(row["similarity"]),
                }
                for row in rows
            ]

    def _rrf_merge(self, list1: list, list2: list, limit: int, k: int = 60) -> list:
        """Merge two ranked lists using Reciprocal Rank Fusion."""
        scores = {}
        items = {}

        for rank, row in enumerate(list1):
            row_id = str(row["id"])
            scores[row_id] = scores.get(row_id, 0) + 1.0 / (k + rank)
            items[row_id] = row

        for rank, row in enumerate(list2):
            row_id = str(row["id"])
            scores[row_id] = scores.get(row_id, 0) + 1.0 / (k + rank)
            if row_id not in items:
                items[row_id] = row

        # Sort by RRF score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Return merged results with RRF score as similarity
        results = []
        for row_id in sorted_ids[:limit]:
            row = items[row_id]
            # Create a mutable copy with updated similarity
            result = dict(row)
            result["similarity"] = scores[row_id]
            results.append(result)

        return results

    async def get_pdf_document(self, pdf_path: str) -> dict | None:
        """Get PDF document by path."""
        await self.initialize()

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT id, title, pdf_path, num_pages, num_text_chunks,
                       num_tables, num_equations, num_images, content_summary,
                       metadata, created_at
                FROM {self.pdf_documents_table}
                WHERE pdf_path = $1
                """,
                pdf_path,
            )

            if row:
                return {
                    "id": str(row["id"]),
                    "title": row["title"],
                    "pdf_path": row["pdf_path"],
                    "num_pages": row["num_pages"],
                    "num_text_chunks": row["num_text_chunks"],
                    "num_tables": row["num_tables"],
                    "num_equations": row["num_equations"],
                    "num_images": row["num_images"],
                    "content_summary": row["content_summary"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "created_at": row["created_at"],
                }
            return None

    async def get_questions_for_pdf(self, pdf_path: str) -> list[dict]:
        """Get all questions for a PDF document."""
        await self.initialize()

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT q.id, q.question, q.difficulty, q.supported_by, q.created_at
                FROM {self.pdf_questions_table} q
                JOIN {self.pdf_documents_table} d ON q.pdf_document_id = d.id
                WHERE d.pdf_path = $1
                ORDER BY q.created_at
                """,
                pdf_path,
            )

            return [
                {
                    "id": str(row["id"]),
                    "question": row["question"],
                    "difficulty": row["difficulty"],
                    "supported_by": row["supported_by"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]

    async def delete_pdf_document(self, pdf_path: str) -> bool:
        """Delete a PDF document and all its questions/chunks."""
        await self.initialize()

        async with self.pool.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM {self.pdf_documents_table} WHERE pdf_path = $1",
                pdf_path,
            )
            deleted = result.split()[-1] != "0"
            if deleted:
                logger.info(f"Deleted PDF document: {pdf_path}")
            return deleted

    async def get_statistics(self) -> dict:
        """Get statistics about stored PDFs, questions, and chunks."""
        await self.initialize()

        async with self.pool.acquire() as conn:
            pdf_count = await conn.fetchval(
                f"SELECT COUNT(*) FROM {self.pdf_documents_table}"
            )
            question_count = await conn.fetchval(
                f"SELECT COUNT(*) FROM {self.pdf_questions_table}"
            )
            chunk_count = await conn.fetchval(
                f"SELECT COUNT(*) FROM {self.pdf_chunks_table}"
            )

            return {
                "pdf_documents": pdf_count,
                "questions": question_count,
                "chunks": chunk_count,
            }


if __name__ == "__main__":
    import asyncio
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    async def main():
        print("=" * 60)
        print("PDF Question Store Test")
        print("=" * 60)

        store = PDFQuestionStore()
        await store.initialize()

        # Get statistics
        stats = await store.get_statistics()
        print(f"\nStatistics:")
        print(f"  PDF Documents: {stats['pdf_documents']}")
        print(f"  Questions: {stats['questions']}")
        print(f"  Chunks: {stats['chunks']}")

        # Test save
        print("\nSaving test PDF result...")
        doc_id = await store.save_pdf_result(
            pdf_path="test/sample.pdf",
            title="Sample PDF",
            num_pages=5,
            num_text_chunks=10,
            questions=[
                {"question": "What is machine learning?", "difficulty": "easy"},
                {"question": "How do neural networks work?", "difficulty": "medium"},
            ],
            chunks=[
                {"chunk_id": "c1", "content": "Machine learning is a subset of AI."},
                {"chunk_id": "c2", "content": "Neural networks are inspired by the brain."},
            ],
        )
        print(f"  Saved document ID: {doc_id}")

        # Test search
        print("\nSearching for questions about 'neural networks'...")
        results = await store.search_questions("neural networks", limit=3)
        for r in results:
            print(f"  - {r['question']} (similarity: {r['similarity']:.4f})")

        # Cleanup
        await store.delete_pdf_document("test/sample.pdf")
        await store.close()

        print("\n" + "=" * 60)
        print("PDF Question Store test completed!")
        print("=" * 60)

    asyncio.run(main())
