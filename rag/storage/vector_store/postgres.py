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
PostgreSQL/Neon vector store implementation with pgvector and hybrid search.

Module: rag.storage.vector_store.postgres
=========================================

This module provides PostgreSQL integration for storing and searching
document chunks with embeddings using pgvector. Supports semantic (vector),
text (full-text), and hybrid (RRF fusion) search modes.

Classes
-------
PostgresHybridStore
    PostgreSQL store with hybrid vector + text search capabilities using pgvector.

    Methods:
        __init__()
            Initialize store (lazy connection).

        async initialize() -> None
            Establish PostgreSQL connection and create tables/indexes.

        async close() -> None
            Close PostgreSQL connection.

        async add(chunks: list[ChunkData], document_id: str) -> None
            Store document chunks with embeddings.

        async save_document(title, source, content, metadata) -> str
            Add a full document, returns document UUID.

        async semantic_search(query_embedding, match_count) -> list[SearchResult]
            Pure vector similarity search using pgvector.

        async text_search(query, match_count) -> list[SearchResult]
            Full-text search using PostgreSQL ts_vector.

        async hybrid_search(query, query_embedding, match_count) -> list[SearchResult]
            Combined search using Reciprocal Rank Fusion (RRF).

        async clean_collections() -> None
            Delete all chunks and documents.

        async get_document_by_source(source) -> dict | None
            Get document by source path.

        async get_document_hash(source) -> str | None
            Get content hash for a document.

        async delete_document_and_chunks(source) -> bool
            Delete a document and its chunks.

        async get_all_document_sources() -> list[str]
            Get all document source paths.

Usage
-----
    from rag.storage.vector_store.postgres import PostgresHybridStore

    # Create and initialize store
    store = PostgresHybridStore()
    await store.initialize()

    # Add document
    doc_id = await store.save_document(
        title="My Doc",
        source="doc.pdf",
        content="Full text...",
        metadata={}
    )

    # Add chunks
    await store.add(chunks, doc_id)

    # Search
    results = await store.hybrid_search(
        query="search text",
        query_embedding=[0.1, 0.2, ...],
        match_count=5
    )

    # Cleanup
    await store.close()
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any

import asyncpg
from pgvector.asyncpg import register_vector

from rag.config.settings import load_settings
from rag.ingestion.models import ChunkData, SearchResult

logger = logging.getLogger(__name__)


class PostgresHybridStore:
    """PostgreSQL implementation with hybrid vector + text search using pgvector."""

    def __init__(self):
        """Initialize PostgreSQL connection."""
        self.settings = load_settings()
        self.pool: asyncpg.Pool | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize PostgreSQL connection and create tables/indexes."""
        if self._initialized:
            return

        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                self.settings.database_url,
                min_size=1,
                max_size=10,
                command_timeout=60,
            )

            # Enable pgvector extension first, then register type
            async with self.pool.acquire() as conn:
                # Enable pgvector extension (must be done before register_vector)
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Register pgvector type (needs fresh connection after extension creation)
            async with self.pool.acquire() as conn:
                await register_vector(conn)

                # Create documents table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.settings.postgres_table_documents} (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        title TEXT NOT NULL,
                        source TEXT NOT NULL UNIQUE,
                        content TEXT,
                        metadata JSONB DEFAULT '{{}}',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)

                # Create chunks table with vector column
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.settings.postgres_table_chunks} (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        document_id UUID NOT NULL REFERENCES {self.settings.postgres_table_documents}(id) ON DELETE CASCADE,
                        content TEXT NOT NULL,
                        embedding vector({self.settings.embedding_dimension}),
                        chunk_index INTEGER NOT NULL,
                        metadata JSONB DEFAULT '{{}}',
                        token_count INTEGER,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
                    )
                """)

                # Create indexes
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS chunks_embedding_idx
                    ON {self.settings.postgres_table_chunks}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)

                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS chunks_document_id_idx
                    ON {self.settings.postgres_table_chunks}(document_id)
                """)

                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS chunks_content_tsv_idx
                    ON {self.settings.postgres_table_chunks}
                    USING GIN(content_tsv)
                """)

                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS documents_source_idx
                    ON {self.settings.postgres_table_documents}(source)
                """)

            logger.info(f"Connected to PostgreSQL and initialized tables")
            self._initialized = True

        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}")
            raise

    async def close(self) -> None:
        """Close PostgreSQL connection."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            self._initialized = False
            logger.info("PostgreSQL connection closed")

    async def add(self, chunks: list[ChunkData], document_id: str) -> None:
        """
        Store document chunks with embeddings.

        Args:
            chunks: List of chunks with embeddings
            document_id: Parent document ID (UUID string)
        """
        await self.initialize()

        async with self.pool.acquire() as conn:
            await register_vector(conn)

            for chunk in chunks:
                await conn.execute(
                    f"""
                    INSERT INTO {self.settings.postgres_table_chunks}
                    (document_id, content, embedding, chunk_index, metadata, token_count)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    uuid.UUID(document_id),
                    chunk.content,
                    chunk.embedding,
                    chunk.index,
                    json.dumps(chunk.metadata),
                    chunk.token_count,
                )

            logger.info(f"Inserted {len(chunks)} chunks for document {document_id}")

    async def semantic_search(
        self, query_embedding: list[float], match_count: int | None = None
    ) -> list[SearchResult]:
        """
        Perform pure semantic search using vector similarity.

        Args:
            query_embedding: Query embedding vector
            match_count: Number of results to return

        Returns:
            List of search results ordered by similarity
        """
        await self.initialize()

        if match_count is None:
            match_count = self.settings.default_match_count
        match_count = min(match_count, self.settings.max_match_count)

        try:
            async with self.pool.acquire() as conn:
                await register_vector(conn)

                # Set IVF probes for better recall (default is 1, we use 10)
                await conn.execute("SET ivfflat.probes = 10")

                rows = await conn.fetch(
                    f"""
                    SELECT
                        c.id as chunk_id,
                        c.document_id,
                        c.content,
                        1 - (c.embedding <=> $1::vector) as similarity,
                        c.metadata,
                        d.title as document_title,
                        d.source as document_source
                    FROM {self.settings.postgres_table_chunks} c
                    JOIN {self.settings.postgres_table_documents} d ON c.document_id = d.id
                    ORDER BY c.embedding <=> $1::vector
                    LIMIT $2
                    """,
                    query_embedding,
                    match_count,
                )

                return [
                    SearchResult(
                        chunk_id=str(row["chunk_id"]),
                        document_id=str(row["document_id"]),
                        content=row["content"],
                        similarity=float(row["similarity"]),
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                        document_title=row["document_title"],
                        document_source=row["document_source"],
                    )
                    for row in rows
                ]

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    async def text_search(
        self, query: str, match_count: int | None = None
    ) -> list[SearchResult]:
        """
        Perform full-text search using PostgreSQL ts_vector.

        Args:
            query: Search query text
            match_count: Number of results to return

        Returns:
            List of search results ordered by text relevance
        """
        await self.initialize()

        if match_count is None:
            match_count = self.settings.default_match_count
        match_count = min(match_count, self.settings.max_match_count)

        try:
            async with self.pool.acquire() as conn:
                # Convert query to tsquery format
                rows = await conn.fetch(
                    f"""
                    SELECT
                        c.id as chunk_id,
                        c.document_id,
                        c.content,
                        ts_rank(c.content_tsv, plainto_tsquery('english', $1)) as similarity,
                        c.metadata,
                        d.title as document_title,
                        d.source as document_source
                    FROM {self.settings.postgres_table_chunks} c
                    JOIN {self.settings.postgres_table_documents} d ON c.document_id = d.id
                    WHERE c.content_tsv @@ plainto_tsquery('english', $1)
                    ORDER BY ts_rank(c.content_tsv, plainto_tsquery('english', $1)) DESC
                    LIMIT $2
                    """,
                    query,
                    match_count * 2,  # Over-fetch for RRF
                )

                return [
                    SearchResult(
                        chunk_id=str(row["chunk_id"]),
                        document_id=str(row["document_id"]),
                        content=row["content"],
                        similarity=float(row["similarity"]),
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                        document_title=row["document_title"],
                        document_source=row["document_source"],
                    )
                    for row in rows
                ]

        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []

    async def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        match_count: int | None = None,
    ) -> list[SearchResult]:
        """
        Perform hybrid search combining semantic and keyword matching.

        Uses Reciprocal Rank Fusion (RRF) to merge results.

        Args:
            query: Search query text
            query_embedding: Query embedding vector
            match_count: Number of results to return

        Returns:
            List of search results sorted by combined RRF score
        """
        await self.initialize()

        if match_count is None:
            match_count = self.settings.default_match_count
        match_count = min(match_count, self.settings.max_match_count)

        # Over-fetch for better RRF results
        fetch_count = match_count * 2

        # Run both searches concurrently
        semantic_results, text_results = await asyncio.gather(
            self.semantic_search(query_embedding, fetch_count),
            self.text_search(query, fetch_count),
            return_exceptions=True,
        )

        # Handle errors gracefully
        if isinstance(semantic_results, Exception):
            logger.warning(f"Semantic search failed: {semantic_results}")
            semantic_results = []
        if isinstance(text_results, Exception):
            logger.warning(f"Text search failed: {text_results}")
            text_results = []

        if not semantic_results and not text_results:
            logger.error("Both searches failed")
            return []

        # Merge using RRF
        merged_results = self._reciprocal_rank_fusion(
            [semantic_results, text_results], k=60
        )

        return merged_results[:match_count]

    def _reciprocal_rank_fusion(
        self, search_results_list: list[list[SearchResult]], k: int = 60
    ) -> list[SearchResult]:
        """
        Merge multiple ranked lists using Reciprocal Rank Fusion.

        Args:
            search_results_list: List of ranked result lists
            k: RRF constant (default: 60)

        Returns:
            Unified list sorted by combined RRF score
        """
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, SearchResult] = {}

        for results in search_results_list:
            for rank, result in enumerate(results):
                chunk_id = result.chunk_id
                rrf_score = 1.0 / (k + rank)

                if chunk_id in rrf_scores:
                    rrf_scores[chunk_id] += rrf_score
                else:
                    rrf_scores[chunk_id] = rrf_score
                    chunk_map[chunk_id] = result

        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        merged_results = []
        for chunk_id, rrf_score in sorted_chunks:
            result = chunk_map[chunk_id]
            merged_result = SearchResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                content=result.content,
                similarity=rrf_score,
                metadata=result.metadata,
                document_title=result.document_title,
                document_source=result.document_source,
            )
            merged_results.append(merged_result)

        logger.info(
            f"RRF merged {len(search_results_list)} lists into {len(merged_results)} results"
        )
        return merged_results

    async def save_document(
        self, title: str, source: str, content: str, metadata: dict[str, Any]
    ) -> str:
        """
        Save a document to PostgreSQL.

        Args:
            title: Document title
            source: Document source path
            content: Document content
            metadata: Document metadata

        Returns:
            Document ID (UUID as string)
        """
        await self.initialize()

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                INSERT INTO {self.settings.postgres_table_documents}
                (title, source, content, metadata)
                VALUES ($1, $2, $3, $4)
                RETURNING id
                """,
                title,
                source,
                content,
                json.dumps(metadata),
            )

            doc_id = str(row["id"])
            logger.info(f"Saved document with ID: {doc_id}")
            return doc_id

    async def clean_collections(self) -> None:
        """Clean all data from tables."""
        await self.initialize()

        async with self.pool.acquire() as conn:
            # Delete chunks first (foreign key constraint)
            chunks_result = await conn.execute(
                f"DELETE FROM {self.settings.postgres_table_chunks}"
            )
            docs_result = await conn.execute(
                f"DELETE FROM {self.settings.postgres_table_documents}"
            )

            logger.info(f"Deleted all chunks and documents")

    async def get_document_by_source(self, source: str) -> dict[str, Any] | None:
        """
        Get a document by its source path.

        Args:
            source: Document source path (relative file path)

        Returns:
            Document dict if found, None otherwise
        """
        await self.initialize()

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT id, title, source, content, metadata, created_at
                FROM {self.settings.postgres_table_documents}
                WHERE source = $1
                """,
                source,
            )

            if row:
                return {
                    "id": str(row["id"]),
                    "title": row["title"],
                    "source": row["source"],
                    "content": row["content"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "created_at": row["created_at"],
                }
            return None

    async def get_document_hash(self, source: str) -> str | None:
        """
        Get the content hash for a document by source path.

        Args:
            source: Document source path

        Returns:
            Content hash if document exists, None otherwise
        """
        doc = await self.get_document_by_source(source)
        if doc and "metadata" in doc:
            return doc["metadata"].get("content_hash")
        return None

    async def delete_document_and_chunks(self, source: str) -> bool:
        """
        Delete a document and all its chunks by source path.

        Args:
            source: Document source path

        Returns:
            True if document was deleted, False if not found
        """
        await self.initialize()

        async with self.pool.acquire() as conn:
            # Delete document (chunks will be deleted via CASCADE)
            result = await conn.execute(
                f"""
                DELETE FROM {self.settings.postgres_table_documents}
                WHERE source = $1
                """,
                source,
            )

            deleted = result.split()[-1] != "0"
            if deleted:
                logger.info(f"Deleted document '{source}' and its chunks")
            return deleted

    async def get_all_document_sources(self) -> list[str]:
        """
        Get all document source paths currently in the database.

        Returns:
            List of source paths
        """
        await self.initialize()

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT source FROM {self.settings.postgres_table_documents}"
            )
            return [row["source"] for row in rows]

    async def get_chunk_count(self) -> int:
        """Get total number of chunks."""
        await self.initialize()

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT COUNT(*) as count FROM {self.settings.postgres_table_chunks}"
            )
            return row["count"]

    async def get_document_count(self) -> int:
        """Get total number of documents."""
        await self.initialize()

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT COUNT(*) as count FROM {self.settings.postgres_table_documents}"
            )
            return row["count"]


if __name__ == "__main__":
    import logging
    import time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    _logger = logging.getLogger(__name__)

    async def main():
        _logger.info("=" * 60)
        _logger.info("RAG PostgreSQL Store Module Test")
        _logger.info("=" * 60)

        # Create and initialize store
        store = PostgresHybridStore()
        _logger.info("[Initializing PostgreSQL connection...]")
        await store.initialize()
        _logger.info("  Connected successfully!")

        # Get counts
        _logger.info("--- Database Stats ---")
        doc_count = await store.get_document_count()
        chunk_count = await store.get_chunk_count()
        _logger.info(f"  Documents: {doc_count}")
        _logger.info(f"  Chunks: {chunk_count}")

        # Get document sources
        _logger.info("--- Document Sources ---")
        sources = await store.get_all_document_sources()
        for source in sources[:5]:
            _logger.info(f"  - {source}")
        if len(sources) > 5:
            _logger.info(f"  ... and {len(sources) - 5} more")

        # Test search if we have data
        if chunk_count > 0:
            _logger.info("--- Search Test ---")
            from rag.ingestion.embedder import EmbeddingGenerator

            embedder = EmbeddingGenerator()
            test_query = "What does the company do?"
            _logger.info(f"  Query: '{test_query}'")

            # Generate embedding
            query_embedding = await embedder.embed_query(test_query)

            # Semantic search
            start = time.time()
            semantic_results = await store.semantic_search(query_embedding, 3)
            semantic_time = (time.time() - start) * 1000
            _logger.info(f"  Semantic Search ({semantic_time:.0f}ms):")
            for i, r in enumerate(semantic_results):
                _logger.info(f"    [{i+1}] {r.document_title} (score: {r.similarity:.3f})")

            # Text search
            start = time.time()
            text_results = await store.text_search(test_query, 3)
            text_time = (time.time() - start) * 1000
            _logger.info(f"  Text Search ({text_time:.0f}ms):")
            for i, r in enumerate(text_results):
                _logger.info(f"    [{i+1}] {r.document_title} (score: {r.similarity:.3f})")

            # Hybrid search
            start = time.time()
            hybrid_results = await store.hybrid_search(test_query, query_embedding, 3)
            hybrid_time = (time.time() - start) * 1000
            _logger.info(f"  Hybrid Search ({hybrid_time:.0f}ms):")
            for i, r in enumerate(hybrid_results):
                _logger.info(f"    [{i+1}] {r.document_title} (score: {r.similarity:.4f})")
        else:
            _logger.info("[Skipping search test - no data]")

        # Close connection
        await store.close()

        _logger.info("=" * 60)
        _logger.info("PostgreSQL store test completed successfully!")
        _logger.info("=" * 60)

    asyncio.run(main())
