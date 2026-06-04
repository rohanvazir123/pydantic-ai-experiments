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
PostgreSQL vector store implementation with pgvector and hybrid search.

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
from typing import Any

import asyncpg
from pgvector.asyncpg import register_vector

from rag.config.settings import load_settings
from rag.ingestion.models import ChunkData, MetadataFilter, SearchResult

logger = logging.getLogger(__name__)


class PostgresHybridStore:
    """PostgreSQL implementation with hybrid vector + text search using pgvector."""

    # Reindex when chunk count exceeds this multiple of the count at index build time
    _IVFFLAT_REINDEX_FACTOR = 3

    def __init__(self):
        """Initialize PostgreSQL connection."""
        self.settings = load_settings()
        self.pool: asyncpg.Pool | None = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._ivfflat_index_build_count: int = 0  # chunk count when IVFFlat was last built

    async def initialize(self) -> None:
        """Initialize PostgreSQL connection and create tables/indexes."""
        async with self._init_lock:
            if self._initialized:
                return

            await self._do_initialize()

    async def _do_initialize(self) -> None:
        """Internal initialization (called under _init_lock)."""
        try:
            # Enable pgvector extension before creating the pool so the
            # register_vector init callback succeeds on the first connection.
            temp_conn = await asyncpg.connect(self.settings.database_url)
            try:
                await temp_conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                await temp_conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
                # Uncomment to enable BM25 search via ParadeDB (requires pg_search extension).
                # Install: https://docs.paradedb.com/documentation/getting-started/self-hosted
                # try:
                #     await temp_conn.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
                # except Exception:
                #     pass
            finally:
                await temp_conn.close()

            # Create connection pool; register_vector runs once per new connection
            self.pool = await asyncpg.create_pool(
                self.settings.database_url,
                min_size=self.settings.db_pool_min_size,
                max_size=self.settings.db_pool_max_size,
                command_timeout=60,
                init=register_vector,
            )

            async with self.pool.acquire() as conn:
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
                    CREATE INDEX IF NOT EXISTS chunks_content_trgm_idx
                    ON {self.settings.postgres_table_chunks}
                    USING GIN(content gin_trgm_ops)
                """)

                try:
                    await conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS chunks_bm25_idx
                        ON {self.settings.postgres_table_chunks}
                        USING bm25 (id, content)
                        WITH (key_field='id')
                    """)
                except Exception:
                    # bm25 access method requires pg_search (ParadeDB).
                    # Falls back to tsvector full-text search when not available.
                    pass

                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS documents_source_idx
                    ON {self.settings.postgres_table_documents}(source)
                """)

            # Record chunk count at index build time for IVFFlat reindex trigger
            async with self.pool.acquire() as count_conn:
                row = await count_conn.fetchrow(
                    f"SELECT COUNT(*) as n FROM {self.settings.postgres_table_chunks}"
                )
                self._ivfflat_index_build_count = row["n"]

            logger.info("Connected to PostgreSQL and initialized tables")
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
            await conn.executemany(
                f"""
                INSERT INTO {self.settings.postgres_table_chunks}
                (document_id, content, embedding, chunk_index, metadata, token_count)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                [
                    (
                        uuid.UUID(document_id),
                        chunk.content,
                        chunk.embedding,
                        chunk.index,
                        json.dumps(chunk.metadata),
                        chunk.token_count,
                    )
                    for chunk in chunks
                ],
            )

            logger.info(f"Inserted {len(chunks)} chunks for document {document_id}")

            # Check if IVFFlat index needs rebuilding due to data growth.
            # IVFFlat centroids are fixed at build time; recall degrades when the
            # chunk count grows beyond ~3x the count at index creation.
            row = await conn.fetchrow(
                f"SELECT COUNT(*) as n FROM {self.settings.postgres_table_chunks}"
            )
            current_count = row["n"]
            threshold = self._ivfflat_index_build_count * self._IVFFLAT_REINDEX_FACTOR
            if self._ivfflat_index_build_count > 0 and current_count >= threshold:
                logger.info(
                    f"IVFFlat reindex triggered: {current_count} chunks "
                    f"(threshold {threshold}, built at {self._ivfflat_index_build_count})"
                )
                await conn.execute(
                    "REINDEX INDEX CONCURRENTLY chunks_embedding_idx"
                )
                self._ivfflat_index_build_count = current_count
                logger.info("IVFFlat index rebuilt successfully")

    def _build_filter_clause(
        self,
        metadata_filter: MetadataFilter,
        param_offset: int,
    ) -> tuple[list[str], list[Any]]:
        """Build SQL WHERE clause fragments and bound params for a MetadataFilter.

        JSONB keys are passed as parameters (not inlined) to prevent injection.

        Args:
            metadata_filter: Filter specification.
            param_offset: First positional parameter index to use (1-based, e.g. 3).

        Returns:
            (clauses, params) — clauses is a list of SQL fragments to AND together;
            params is the list of values to append to the query's argument list.
        """
        clauses: list[str] = []
        params: list[Any] = []
        idx = param_offset

        for key, value in metadata_filter.metadata_eq.items():
            clauses.append(f"c.metadata->>${idx} = ${idx + 1}")
            params.extend([key, str(value)])
            idx += 2

        for key, values in metadata_filter.metadata_in.items():
            clauses.append(f"c.metadata->>${idx} = ANY(${idx + 1}::text[])")
            params.extend([key, [str(v) for v in values]])
            idx += 2

        # Range filters: ISO 8601 date strings ("YYYY-MM-DD") compare correctly
        # via text/lexicographic order, so c.metadata->>'date' >= $N works.
        for key, value in metadata_filter.metadata_gte.items():
            clauses.append(f"c.metadata->>${idx} >= ${idx + 1}")
            params.extend([key, value])
            idx += 2

        for key, value in metadata_filter.metadata_lte.items():
            clauses.append(f"c.metadata->>${idx} <= ${idx + 1}")
            params.extend([key, value])
            idx += 2

        if metadata_filter.document_source:
            clauses.append(f"d.source = ${idx}")
            params.append(metadata_filter.document_source)
            idx += 1

        if metadata_filter.document_sources:
            clauses.append(f"d.source = ANY(${idx}::text[])")
            params.append(metadata_filter.document_sources)
            idx += 1

        if metadata_filter.document_title:
            clauses.append(f"d.title = ${idx}")
            params.append(metadata_filter.document_title)

        return clauses, params

    async def semantic_search(
        self,
        query_embedding: list[float],
        match_count: int | None = None,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        """
        Perform pure semantic search using vector similarity.

        Args:
            query_embedding: Query embedding vector
            match_count: Number of results to return
            metadata_filter: Optional filter on chunk/document metadata

        Returns:
            List of search results ordered by similarity
        """
        await self.initialize()

        if match_count is None:
            match_count = self.settings.default_match_count
        match_count = min(match_count, self.settings.max_match_count)

        try:
            async with self.pool.acquire() as conn:
                # Set IVF probes for better recall (default is 1, we use 10)
                await conn.execute("SET ivfflat.probes = 10")

                # Build optional WHERE clause from metadata filter.
                # $1 = query_embedding, $2 = match_count; filter params start at $3.
                filter_clauses, filter_params = (
                    self._build_filter_clause(metadata_filter, 3)
                    if metadata_filter and not metadata_filter.is_empty
                    else ([], [])
                )
                where_sql = ("WHERE " + " AND ".join(filter_clauses)) if filter_clauses else ""

                # <=> is pgvector cosine distance (0=identical, 2=opposite).
                # 1 - distance converts it to similarity (1=identical, -1=opposite).
                # IVFFlat index with probes=10 trades recall for speed vs exact scan.
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
                    {where_sql}
                    ORDER BY c.embedding <=> $1::vector
                    LIMIT $2
                    """,
                    query_embedding,
                    match_count,
                    *filter_params,
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
        self,
        query: str,
        match_count: int | None = None,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        """
        Full-text search using PostgreSQL tsvector + ts_rank.

        plainto_tsquery stems and ANDs the query terms using the English dictionary
        (e.g. "machine learning" → 'machin' & 'learn'). Only documents containing
        ALL stemmed terms pass the WHERE filter.

        ts_rank scoring — what it does and does NOT do:
          - Counts how often each query term appears in the tsvector (term frequency).
          - Weights positions by lexeme type: title > body (if set at index time).
          - No IDF: a common word like "the" scores the same as rare "indemnification".
          - No length normalisation: we pass no normalization flag (default 0), so
            longer documents are NOT penalised. A 2000-word chunk with 5 hits scores
            the same as a 100-word chunk with 5 hits.
          - No corpus-level statistics of any kind — scores are computed per-document.

        Implication: ts_rank degrades as corpus grows because high-frequency terms
        in many documents all receive the same score. BM25 (not used here) fixes this
        via IDF. The semantic + fuzzy legs in hybrid_search compensate partially.

        Args:
            query: Search query text
            match_count: Number of results to return
            metadata_filter: Optional filter on chunk/document metadata

        Returns:
            List of search results ordered by ts_rank score
        """
        await self.initialize()

        if match_count is None:
            match_count = self.settings.default_match_count
        match_count = min(match_count, self.settings.max_match_count)

        try:
            async with self.pool.acquire() as conn:
                # $1 = query (used twice), $2 = match_count*2; filter params start at $3.
                filter_clauses, filter_params = (
                    self._build_filter_clause(metadata_filter, 3)
                    if metadata_filter and not metadata_filter.is_empty
                    else ([], [])
                )
                extra_where = (" AND " + " AND ".join(filter_clauses)) if filter_clauses else ""

                # ts_rank with no normalization flag (default 0) = raw term frequency only.
                # No IDF, no length penalty. See docstring for full implications.
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
                    WHERE c.content_tsv @@ plainto_tsquery('english', $1){extra_where}
                    ORDER BY ts_rank(c.content_tsv, plainto_tsquery('english', $1)) DESC
                    LIMIT $2
                    """,
                    query,
                    match_count * 2,  # Over-fetch for RRF
                    *filter_params,
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

    async def fuzzy_search(
        self,
        query: str,
        match_count: int | None = None,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        """
        Fuzzy search via pg_trgm's word_similarity function.

        Splits text into overlapping 3-character trigrams and scores the best
        matching word-boundary alignment between query and content (0–1 float).
        Threshold 0.2 filters noise; backed by a GIN trigram index for fast lookup.

        Catches typos ("NeuralFow" → "NeuralFlow"), abbreviations, and partial words
        that plainto_tsquery misses because it requires exact stem matches.

        Args:
            query: Search query text
            match_count: Number of results to return
            metadata_filter: Optional filter on chunk/document metadata

        Returns:
            List of search results ordered by trigram similarity score
        """
        await self.initialize()

        if match_count is None:
            match_count = self.settings.default_match_count
        match_count = min(match_count, self.settings.max_match_count)

        try:
            async with self.pool.acquire() as conn:
                # $1 = query (used twice), $2 = match_count*2; filter params start at $3.
                filter_clauses, filter_params = (
                    self._build_filter_clause(metadata_filter, 3)
                    if metadata_filter and not metadata_filter.is_empty
                    else ([], [])
                )
                extra_where = (" AND " + " AND ".join(filter_clauses)) if filter_clauses else ""

                # word_similarity scores the best trigram match between any word in
                # the query and any word in the content (0–1). Threshold 0.2 filters
                # noise; backed by a GIN pg_trgm index for fast lookup.
                rows = await conn.fetch(
                    f"""
                    SELECT
                        c.id as chunk_id,
                        c.document_id,
                        c.content,
                        word_similarity($1, c.content) as similarity,
                        c.metadata,
                        d.title as document_title,
                        d.source as document_source
                    FROM {self.settings.postgres_table_chunks} c
                    JOIN {self.settings.postgres_table_documents} d ON c.document_id = d.id
                    WHERE word_similarity($1, c.content) > 0.2{extra_where}
                    ORDER BY word_similarity($1, c.content) DESC
                    LIMIT $2
                    """,
                    query,
                    match_count * 2,  # Over-fetch for RRF
                    *filter_params,
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
            logger.error(f"Fuzzy search failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Uncomment to enable BM25 search via ParadeDB (pg_search extension).
    # Also uncomment the corresponding lines in hybrid_search() below and
    # the CREATE EXTENSION block in _do_initialize().
    # Install: https://docs.paradedb.com/documentation/getting-started/self-hosted
    # ------------------------------------------------------------------
    # async def bm25_search(
    #     self, query: str, match_count: int | None = None
    # ) -> list[SearchResult]:
    #     """
    #     BM25 search via ParadeDB pg_search extension.
    #     @@@ is ParadeDB's match operator. paradedb.score() computes BM25
    #     (term frequency + IDF + length normalisation) — better than ts_rank
    #     at scale. Requires pg_search to be installed and the extension enabled.
    #     """
    #     await self.initialize()
    #     if match_count is None:
    #         match_count = self.settings.default_match_count
    #     match_count = min(match_count, self.settings.max_match_count)
    #     try:
    #         async with self.pool.acquire() as conn:
    #             rows = await conn.fetch(
    #                 f"""
    #                 SELECT
    #                     c.id as chunk_id,
    #                     c.document_id,
    #                     c.content,
    #                     paradedb.score(c.id) as similarity,
    #                     c.metadata,
    #                     d.title as document_title,
    #                     d.source as document_source
    #                 FROM {self.settings.postgres_table_chunks} c
    #                 JOIN {self.settings.postgres_table_documents} d ON c.document_id = d.id
    #                 WHERE c.id @@@ paradedb.match('content', $1)
    #                 ORDER BY paradedb.score(c.id) DESC
    #                 LIMIT $2
    #                 """,
    #                 query,
    #                 match_count * 2,
    #             )
    #             return [
    #                 SearchResult(
    #                     chunk_id=str(row["chunk_id"]),
    #                     document_id=str(row["document_id"]),
    #                     content=row["content"],
    #                     similarity=float(row["similarity"]),
    #                     metadata=json.loads(row["metadata"]) if row["metadata"] else {},
    #                     document_title=row["document_title"],
    #                     document_source=row["document_source"],
    #                 )
    #                 for row in rows
    #             ]
    #     except Exception as e:
    #         logger.error(f"BM25 search failed: {e}")
    #         return []

    async def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        match_count: int | None = None,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        """
        Runs three searches concurrently and merges via Reciprocal Rank Fusion (RRF, k=60):

          1. semantic_search  — pgvector cosine distance (<=>); catches synonyms/paraphrasing
          2. text_search      — tsvector ts_rank via plainto_tsquery; exact stemmed terms
          3. fuzzy_search     — pg_trgm word_similarity; handles typos and partial matches

        Each leg over-fetches (match_count × 2) for better RRF coverage. Any leg that raises
        is caught by return_exceptions=True and treated as an empty list. RRF score =
        Σ 1/(60 + rank) across legs, deduped by chunk_id.

        Args:
            query: Search query text
            query_embedding: Query embedding vector
            match_count: Number of results to return
            metadata_filter: Optional filter applied uniformly to all search legs

        Returns:
            List of search results sorted by combined RRF score
        """
        await self.initialize()

        if match_count is None:
            match_count = self.settings.default_match_count
        match_count = min(match_count, self.settings.max_match_count)

        # Over-fetch for better RRF results
        fetch_count = match_count * 2

        # Run all three searches concurrently.
        # To add BM25 as a 4th leg: uncomment bm25_search() above, then replace
        # the three lines below with the four-leg version in the comment that follows.
        semantic_results, text_results, fuzzy_results = await asyncio.gather(
            self.semantic_search(query_embedding, fetch_count, metadata_filter),
            self.text_search(query, fetch_count, metadata_filter),
            self.fuzzy_search(query, fetch_count, metadata_filter),
            return_exceptions=True,
        )
        # Four-leg version (ParadeDB):
        # semantic_results, text_results, fuzzy_results, bm25_results = await asyncio.gather(
        #     self.semantic_search(query_embedding, fetch_count),
        #     self.text_search(query, fetch_count),
        #     self.fuzzy_search(query, fetch_count),
        #     self.bm25_search(query, fetch_count),
        #     return_exceptions=True,
        # )

        # Handle errors gracefully
        if isinstance(semantic_results, Exception):
            logger.warning(f"Semantic search failed: {semantic_results}")
            semantic_results = []
        if isinstance(text_results, Exception):
            logger.warning(f"Text search failed: {text_results}")
            text_results = []
        if isinstance(fuzzy_results, Exception):
            logger.warning(f"Fuzzy search failed: {fuzzy_results}")
            fuzzy_results = []
        # if isinstance(bm25_results, Exception):  # uncomment with 4-leg version
        #     logger.warning(f"BM25 search failed: {bm25_results}")
        #     bm25_results = []

        if not any([semantic_results, text_results, fuzzy_results]):
            logger.error("All searches failed")
            return []

        # Merge using RRF across all three signals.
        # Four-leg version: _reciprocal_rank_fusion([semantic_results, text_results, fuzzy_results, bm25_results], k=60)
        merged_results = self._reciprocal_rank_fusion(
            [semantic_results, text_results, fuzzy_results], k=60
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

            logger.info(f"Cleaned collections: {chunks_result}, {docs_result}")

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
