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

"""Tests for PostgreSQL/pgvector vector store."""

import pytest
import pytest_asyncio

from rag.config.settings import load_settings
from rag.ingestion.models import ChunkData, MetadataFilter
from rag.storage.vector_store.postgres import PostgresHybridStore


class TestPostgresConnection:
    """Test PostgreSQL connection and basic operations."""

    @pytest.fixture
    def settings(self):
        """Load settings fixture."""
        return load_settings()

    @pytest.fixture
    def store(self):
        """Create PostgresHybridStore fixture."""
        return PostgresHybridStore()

    def test_store_initialization(self, store):
        """Test that store initializes with correct settings."""
        assert store.settings is not None
        assert store.pool is None  # Not connected yet
        assert store._initialized is False

    def test_settings_have_database_url(self, settings):
        """Test that settings have database URL configured."""
        assert settings.database_url is not None
        assert len(settings.database_url) > 0

    def test_settings_have_table_names(self, settings):
        """Test that settings have table names configured."""
        assert settings.postgres_table_documents is not None
        assert settings.postgres_table_chunks is not None
        assert len(settings.postgres_table_documents) > 0
        assert len(settings.postgres_table_chunks) > 0


@pytest.mark.asyncio
class TestPostgresConnectionLive:
    """Test PostgreSQL connection - requires live PostgreSQL connection."""

    @pytest.fixture
    def settings(self):
        """Load settings fixture."""
        return load_settings()

    @pytest_asyncio.fixture
    async def connected_store(self):
        """Create and initialize a connected PostgresHybridStore."""
        store = PostgresHybridStore()
        await store.initialize()
        yield store
        await store.close()

    async def test_postgres_connection(self, connected_store):
        """Test that PostgreSQL connection is established."""
        assert connected_store._initialized is True
        assert connected_store.pool is not None

    async def test_tables_exist(self, connected_store, settings):
        """Test that required tables exist."""
        async with connected_store.pool.acquire() as conn:
            # Check documents table
            docs_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = $1
                )
            """, settings.postgres_table_documents)
            assert docs_exists, f"Documents table '{settings.postgres_table_documents}' not found"

            # Check chunks table
            chunks_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = $1
                )
            """, settings.postgres_table_chunks)
            assert chunks_exists, f"Chunks table '{settings.postgres_table_chunks}' not found"

    async def test_pgvector_extension_enabled(self, connected_store):
        """Test that pgvector extension is enabled."""
        async with connected_store.pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM pg_extension WHERE extname = 'vector'
                )
            """)
            assert result, "pgvector extension is not enabled"

    async def test_vector_index_exists(self, connected_store, settings):
        """Test that vector index exists on chunks table."""
        async with connected_store.pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM pg_indexes
                    WHERE tablename = $1
                    AND indexname = 'chunks_embedding_idx'
                )
            """, settings.postgres_table_chunks)
            assert result, "Vector index 'chunks_embedding_idx' not found on chunks table"

    async def test_text_search_index_exists(self, connected_store, settings):
        """Test that text search index exists on chunks table."""
        async with connected_store.pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM pg_indexes
                    WHERE tablename = $1
                    AND indexname = 'chunks_content_tsv_idx'
                )
            """, settings.postgres_table_chunks)
            assert result, "Text search index 'chunks_content_tsv_idx' not found on chunks table"

    async def test_get_document_count(self, connected_store):
        """Test getting document count."""
        count = await connected_store.get_document_count()
        assert isinstance(count, int)
        assert count >= 0

    async def test_get_chunk_count(self, connected_store):
        """Test getting chunk count."""
        count = await connected_store.get_chunk_count()
        assert isinstance(count, int)
        assert count >= 0


@pytest.mark.asyncio
class TestPostgresStoreOperations:
    """Test PostgreSQL store CRUD operations."""

    @pytest_asyncio.fixture
    async def connected_store(self):
        """Create and initialize a connected PostgresHybridStore."""
        store = PostgresHybridStore()
        await store.initialize()
        yield store
        await store.close()

    async def test_save_and_get_document(self, connected_store):
        """Test saving and retrieving a document."""
        # Save document
        doc_id = await connected_store.save_document(
            title="Test Document",
            source="test_doc.txt",
            content="This is test content for PostgreSQL store.",
            metadata={"test": True, "type": "test"}
        )
        assert doc_id is not None
        assert len(doc_id) > 0

        # Retrieve document
        doc = await connected_store.get_document_by_source("test_doc.txt")
        assert doc is not None
        assert doc["title"] == "Test Document"
        assert doc["content"] == "This is test content for PostgreSQL store."
        assert doc["metadata"]["test"] is True

        # Cleanup
        deleted = await connected_store.delete_document_and_chunks("test_doc.txt")
        assert deleted is True

        # Verify deletion
        doc = await connected_store.get_document_by_source("test_doc.txt")
        assert doc is None

    async def test_get_all_document_sources(self, connected_store):
        """Test getting all document sources."""
        sources = await connected_store.get_all_document_sources()
        assert isinstance(sources, list)

    async def test_get_document_hash(self, connected_store):
        """Test getting document hash."""
        # Save document with hash
        doc_id = await connected_store.save_document(
            title="Hash Test Doc",
            source="hash_test.txt",
            content="Content for hash test",
            metadata={"content_hash": "abc123hash"}
        )

        # Get hash
        hash_value = await connected_store.get_document_hash("hash_test.txt")
        assert hash_value == "abc123hash"

        # Cleanup
        await connected_store.delete_document_and_chunks("hash_test.txt")


@pytest.mark.asyncio
class TestPostgresSearchOperations:
    """Test PostgreSQL search operations - requires data in database."""

    @pytest_asyncio.fixture
    async def connected_store(self):
        """Create and initialize a connected PostgresHybridStore."""
        store = PostgresHybridStore()
        await store.initialize()
        yield store
        await store.close()

    async def test_semantic_search_empty_results(self, connected_store):
        """Test semantic search returns empty list when no data matches."""
        # Create a dummy embedding (768 dimensions)
        dummy_embedding = [0.0] * 768

        results = await connected_store.semantic_search(dummy_embedding, 5)
        assert isinstance(results, list)

    async def test_text_search_empty_results(self, connected_store):
        """Test text search returns empty list when no data matches."""
        results = await connected_store.text_search("xyznonexistentquery123", 5)
        assert isinstance(results, list)

    async def test_hybrid_search_empty_results(self, connected_store):
        """Test hybrid search returns empty list when no data matches."""
        dummy_embedding = [0.0] * 768

        results = await connected_store.hybrid_search(
            "xyznonexistentquery123",
            dummy_embedding,
            5
        )
        assert isinstance(results, list)


@pytest.mark.asyncio
class TestEmbeddingDimensionValidation:
    """Test embedding dimension validation."""

    @pytest.fixture
    def settings(self):
        """Load settings fixture."""
        return load_settings()

    async def test_embedding_dimension_is_positive(self, settings):
        """Test that embedding dimension is a positive integer."""
        assert settings.embedding_dimension > 0
        assert isinstance(settings.embedding_dimension, int)

    async def test_embedding_dimension_matches_model(self, settings):
        """Test that embedding dimension matches the configured model."""
        # Known dimensions for common models
        known_dimensions = {
            "nomic-embed-text": 768,
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }

        model_name = settings.embedding_model.lower()

        for model_key, expected_dim in known_dimensions.items():
            if model_key in model_name:
                assert settings.embedding_dimension == expected_dim, (
                    f"Model '{settings.embedding_model}' should have "
                    f"dimension {expected_dim}, but got {settings.embedding_dimension}"
                )
                break


class TestBuildFilterClause:
    """Unit tests for _build_filter_clause — no DB connection required.

    Verifies that the SQL fragments and bound parameter lists are generated
    correctly for every MetadataFilter field combination, including the
    'Q4 2024 earnings' temporal scenario.
    """

    @pytest.fixture
    def store(self):
        return PostgresHybridStore()

    # --- empty filter ---

    def test_empty_filter_produces_no_clauses(self, store):
        clauses, params = store._build_filter_clause(MetadataFilter(), param_offset=3)
        assert clauses == []
        assert params == []

    # --- metadata_eq ---

    def test_single_metadata_eq(self, store):
        f = MetadataFilter(metadata_eq={"doc_type": "policy"})
        clauses, params = store._build_filter_clause(f, param_offset=3)
        assert clauses == ["c.metadata->>$3 = $4"]
        assert params == ["doc_type", "policy"]

    def test_q4_2024_metadata_eq(self, store):
        """Q4 2024 earnings filter: two metadata_eq keys ANDed together."""
        f = MetadataFilter(metadata_eq={"quarter": "Q4", "year": "2024"})
        clauses, params = store._build_filter_clause(f, param_offset=3)
        # Two keys → two clauses, four params (key, value, key, value)
        assert len(clauses) == 2
        assert len(params) == 4
        # Clause 1: quarter
        assert clauses[0] == "c.metadata->>$3 = $4"
        assert params[0] == "quarter"
        assert params[1] == "Q4"
        # Clause 2: year
        assert clauses[1] == "c.metadata->>$5 = $6"
        assert params[2] == "year"
        assert params[3] == "2024"

    def test_metadata_eq_values_coerced_to_str(self, store):
        f = MetadataFilter(metadata_eq={"year": 2024})
        _, params = store._build_filter_clause(f, param_offset=3)
        assert params[1] == "2024"

    # --- metadata_in ---

    def test_metadata_in_single_key(self, store):
        f = MetadataFilter(metadata_in={"quarter": ["Q3", "Q4"]})
        clauses, params = store._build_filter_clause(f, param_offset=3)
        assert clauses == ["c.metadata->>$3 = ANY($4::text[])"]
        assert params[0] == "quarter"
        assert params[1] == ["Q3", "Q4"]

    def test_metadata_in_full_year_filter(self, store):
        """All quarters in 2024."""
        f = MetadataFilter(metadata_in={"quarter": ["Q1", "Q2", "Q3", "Q4"], "year": ["2024"]})
        clauses, params = store._build_filter_clause(f, param_offset=3)
        assert len(clauses) == 2
        assert params[0] == "quarter"
        assert set(params[1]) == {"Q1", "Q2", "Q3", "Q4"}
        assert params[2] == "year"
        assert params[3] == ["2024"]

    # --- document-level filters ---

    def test_document_source_filter(self, store):
        f = MetadataFilter(document_source="rag/documents/earnings/amazon_q4_2024.md")
        clauses, params = store._build_filter_clause(f, param_offset=3)
        assert clauses == ["d.source = $3"]
        assert params == ["rag/documents/earnings/amazon_q4_2024.md"]

    def test_document_sources_filter(self, store):
        sources = ["earnings/q4_2024.md", "earnings/q3_2024.md"]
        f = MetadataFilter(document_sources=sources)
        clauses, params = store._build_filter_clause(f, param_offset=3)
        assert clauses == ["d.source = ANY($3::text[])"]
        assert params == [sources]

    def test_document_title_filter(self, store):
        f = MetadataFilter(document_title="Q4 2024 Earnings Report")
        clauses, params = store._build_filter_clause(f, param_offset=3)
        assert clauses == ["d.title = $3"]
        assert params == ["Q4 2024 Earnings Report"]

    # --- combined filters ---

    def test_q4_2024_combined_with_doc_source(self, store):
        """metadata_eq (quarter+year) AND document_source — 5 params total."""
        f = MetadataFilter(
            metadata_eq={"quarter": "Q4", "year": "2024"},
            document_source="rag/documents/earnings/amazon_q4_2024.md",
        )
        clauses, params = store._build_filter_clause(f, param_offset=3)
        assert len(clauses) == 3
        assert len(params) == 5
        assert clauses[2] == "d.source = $7"
        assert params[4] == "rag/documents/earnings/amazon_q4_2024.md"

    # --- date range (metadata_gte / metadata_lte) ---

    def test_metadata_gte_single_key(self, store):
        f = MetadataFilter(metadata_gte={"date": "2024-10-01"})
        clauses, params = store._build_filter_clause(f, param_offset=3)
        assert clauses == ["c.metadata->>$3 >= $4"]
        assert params == ["date", "2024-10-01"]

    def test_metadata_lte_single_key(self, store):
        f = MetadataFilter(metadata_lte={"date": "2024-12-31"})
        clauses, params = store._build_filter_clause(f, param_offset=3)
        assert clauses == ["c.metadata->>$3 <= $4"]
        assert params == ["date", "2024-12-31"]

    def test_q4_2024_date_range(self, store):
        """Q4 2024 via date range: two range clauses, four params."""
        f = MetadataFilter(
            metadata_gte={"date": "2024-10-01"},
            metadata_lte={"date": "2024-12-31"},
        )
        clauses, params = store._build_filter_clause(f, param_offset=3)
        assert clauses == [
            "c.metadata->>$3 >= $4",
            "c.metadata->>$5 <= $6",
        ]
        assert params == ["date", "2024-10-01", "date", "2024-12-31"]

    def test_combined_eq_and_date_range(self, store):
        """metadata_eq + date range generates sequential param numbers."""
        f = MetadataFilter(
            metadata_eq={"report_type": "earnings"},
            metadata_gte={"date": "2024-10-01"},
            metadata_lte={"date": "2024-12-31"},
        )
        clauses, params = store._build_filter_clause(f, param_offset=3)
        # eq → $3=$4, gte → $5>=$6, lte → $7<=$8
        assert len(clauses) == 3
        assert clauses[0] == "c.metadata->>$3 = $4"
        assert clauses[1] == "c.metadata->>$5 >= $6"
        assert clauses[2] == "c.metadata->>$7 <= $8"
        assert params == [
            "report_type", "earnings",
            "date", "2024-10-01",
            "date", "2024-12-31",
        ]

    # --- param_offset respected ---

    def test_param_offset_shifts_placeholders(self, store):
        f = MetadataFilter(metadata_eq={"quarter": "Q4"})
        clauses_3, _ = store._build_filter_clause(f, param_offset=3)
        clauses_5, _ = store._build_filter_clause(f, param_offset=5)
        assert clauses_3 == ["c.metadata->>$3 = $4"]
        assert clauses_5 == ["c.metadata->>$5 = $6"]


@pytest.mark.integration
@pytest.mark.asyncio
class TestMetadataFilteredSearch:
    """Integration tests: metadata filters must limit the scope of retrieved chunks.

    Seeds three isolated test documents:
      - Q3/2024 earnings report  (4 chunks)
      - Q4/2024 earnings report  (4 chunks)  ← target for Q4 2024 filter
      - Q1/2025 earnings report  (4 chunks)

    All sources are prefixed with 'test_mf_' to avoid polluting real data.
    Cleanup runs after every test class via the autouse 'cleanup' fixture.

    Requires: live PostgreSQL with pgvector.
    """

    # Unique source prefix so these never clash with production docs
    _SOURCES = {
        "q3_2024": "test_mf_earnings_q3_2024.md",
        "q4_2024": "test_mf_earnings_q4_2024.md",
        "q1_2025": "test_mf_earnings_q1_2025.md",
    }
    _DIM = 768

    @pytest_asyncio.fixture(autouse=True)
    async def seed_and_cleanup(self):
        """Insert test documents + chunks, yield, then delete everything."""
        store = PostgresHybridStore()
        await store.initialize()

        self._store = store
        self._doc_ids: dict[str, str] = {}

        # quarter, year, first_day, last_day, key, title
        quarters = [
            ("Q3", "2024", "2024-07-01", "2024-09-30", "q3_2024", "Q3 2024 Earnings Report"),
            ("Q4", "2024", "2024-10-01", "2024-12-31", "q4_2024", "Q4 2024 Earnings Report"),
            ("Q1", "2025", "2025-01-01", "2025-03-31", "q1_2025", "Q1 2025 Earnings Report"),
        ]

        for quarter, year, first_day, last_day, key, title in quarters:
            doc_id = await store.save_document(
                title=title,
                source=self._SOURCES[key],
                content=f"Full text of {title}",
                metadata={"file_type": "md", "quarter": quarter, "year": year},
            )
            self._doc_ids[key] = doc_id
            # Each chunk gets a date spread across the quarter so date-range tests
            # have meaningful ISO 8601 values to compare.
            dates = [first_day, first_day, last_day, last_day]
            chunks = [
                ChunkData(
                    content=f"{title} — chunk {i}: revenue, expenses, net income summary.",
                    index=i,
                    start_char=i * 100,
                    end_char=(i + 1) * 100,
                    metadata={
                        "quarter": quarter,
                        "year": year,
                        "date": dates[i],
                        "report_type": "earnings",
                    },
                    embedding=[0.1 * (i + 1)] * self._DIM,
                )
                for i in range(4)
            ]
            await store.add(chunks, doc_id)

        yield

        for key in self._SOURCES:
            await store.delete_document_and_chunks(self._SOURCES[key])
        await store.close()

    # --- text search with metadata filter ---

    async def test_filter_q4_2024_limits_to_q4_chunks(self):
        """text_search with Q4/2024 filter returns only Q4 2024 chunks."""
        f = MetadataFilter(metadata_eq={"quarter": "Q4", "year": "2024"})
        results = await self._store.text_search("earnings", match_count=20, metadata_filter=f)
        assert results, "Expected Q4 2024 chunks but got none"
        for r in results:
            assert r.metadata.get("quarter") == "Q4", f"Non-Q4 chunk leaked: {r.metadata}"
            assert r.metadata.get("year") == "2024", f"Non-2024 chunk leaked: {r.metadata}"

    async def test_filter_excludes_other_quarters(self):
        """Q4 2024 filter must NOT return Q3 2024 or Q1 2025 chunks."""
        f = MetadataFilter(metadata_eq={"quarter": "Q4", "year": "2024"})
        results = await self._store.text_search("earnings", match_count=20, metadata_filter=f)
        sources = {r.document_source for r in results}
        assert self._SOURCES["q3_2024"] not in sources, "Q3 2024 doc leaked into Q4 2024 results"
        assert self._SOURCES["q1_2025"] not in sources, "Q1 2025 doc leaked into Q4 2024 results"

    async def test_filter_q4_all_years_via_metadata_eq(self):
        """Filter by quarter=Q4 alone returns only Q4 chunks regardless of year."""
        f = MetadataFilter(metadata_eq={"quarter": "Q4"})
        results = await self._store.text_search("earnings", match_count=20, metadata_filter=f)
        for r in results:
            assert r.metadata.get("quarter") == "Q4"

    async def test_filter_by_year_only_returns_both_2024_quarters(self):
        """year=2024 filter returns Q3 and Q4 chunks but not Q1 2025."""
        f = MetadataFilter(metadata_eq={"year": "2024"})
        results = await self._store.text_search("earnings", match_count=20, metadata_filter=f)
        sources = {r.document_source for r in results}
        assert self._SOURCES["q3_2024"] in sources
        assert self._SOURCES["q4_2024"] in sources
        assert self._SOURCES["q1_2025"] not in sources

    async def test_filter_metadata_in_multi_quarter(self):
        """metadata_in filter for [Q3, Q4] returns both quarters."""
        f = MetadataFilter(metadata_in={"quarter": ["Q3", "Q4"]})
        results = await self._store.text_search("earnings", match_count=20, metadata_filter=f)
        quarters_seen = {r.metadata.get("quarter") for r in results}
        assert "Q3" in quarters_seen
        assert "Q4" in quarters_seen
        assert "Q1" not in quarters_seen

    async def test_filter_by_document_source_limits_scope(self):
        """document_source filter returns only chunks from the specified file."""
        f = MetadataFilter(document_source=self._SOURCES["q4_2024"])
        results = await self._store.text_search("earnings", match_count=20, metadata_filter=f)
        assert results
        for r in results:
            assert r.document_source == self._SOURCES["q4_2024"]

    async def test_empty_filter_returns_all_test_documents(self):
        """No filter: all three seeded documents must appear in results."""
        results = await self._store.text_search("earnings", match_count=50)
        sources = {r.document_source for r in results}
        for key in self._SOURCES:
            assert self._SOURCES[key] in sources, f"Expected {self._SOURCES[key]} in results"

    # --- semantic search with metadata filter ---

    async def test_semantic_search_q4_2024_filter(self):
        """semantic_search with Q4/2024 filter returns only Q4 2024 chunks."""
        dummy_embedding = [0.1] * self._DIM
        f = MetadataFilter(metadata_eq={"quarter": "Q4", "year": "2024"})
        results = await self._store.semantic_search(
            dummy_embedding, match_count=20, metadata_filter=f
        )
        for r in results:
            assert r.metadata.get("quarter") == "Q4"
            assert r.metadata.get("year") == "2024"

    # --- hybrid search with metadata filter ---

    async def test_hybrid_search_q4_2024_filter(self):
        """hybrid_search with Q4/2024 filter returns only Q4 2024 chunks."""
        dummy_embedding = [0.1] * self._DIM
        f = MetadataFilter(metadata_eq={"quarter": "Q4", "year": "2024"})
        results = await self._store.hybrid_search(
            "earnings revenue", dummy_embedding, match_count=20, metadata_filter=f
        )
        for r in results:
            assert r.metadata.get("quarter") == "Q4"
            assert r.metadata.get("year") == "2024"

    # --- date range filters (metadata JSONB, ISO 8601 text comparison) ---

    async def test_date_range_q4_2024_excludes_other_quarters(self):
        """date >= 2024-10-01 AND date <= 2024-12-31 returns only Q4 2024 chunks."""
        f = MetadataFilter(
            metadata_gte={"date": "2024-10-01"},
            metadata_lte={"date": "2024-12-31"},
        )
        results = await self._store.text_search("earnings", match_count=20, metadata_filter=f)
        assert results, "Expected Q4 2024 chunks by date range but got none"
        for r in results:
            date = r.metadata.get("date", "")
            assert "2024-10" <= date <= "2024-12-31", (
                f"Chunk date {date!r} is outside Q4 2024 range"
            )
        sources = {r.document_source for r in results}
        assert self._SOURCES["q3_2024"] not in sources
        assert self._SOURCES["q1_2025"] not in sources

    async def test_date_range_full_2024_includes_q3_and_q4(self):
        """date range covering all of 2024 includes Q3 and Q4 but not Q1 2025."""
        f = MetadataFilter(
            metadata_gte={"date": "2024-01-01"},
            metadata_lte={"date": "2024-12-31"},
        )
        results = await self._store.text_search("earnings", match_count=50, metadata_filter=f)
        sources = {r.document_source for r in results}
        assert self._SOURCES["q3_2024"] in sources
        assert self._SOURCES["q4_2024"] in sources
        assert self._SOURCES["q1_2025"] not in sources

    async def test_date_gte_only_excludes_earlier_quarters(self):
        """date >= 2024-10-01 (no upper bound) includes Q4 2024 and Q1 2025."""
        f = MetadataFilter(metadata_gte={"date": "2024-10-01"})
        results = await self._store.text_search("earnings", match_count=50, metadata_filter=f)
        sources = {r.document_source for r in results}
        assert self._SOURCES["q4_2024"] in sources
        assert self._SOURCES["q1_2025"] in sources
        assert self._SOURCES["q3_2024"] not in sources

    async def test_date_and_quarter_combined_filter(self):
        """Combining date range with metadata_eq quarter is redundant but must not error."""
        f = MetadataFilter(
            metadata_eq={"quarter": "Q4"},
            metadata_gte={"date": "2024-10-01"},
            metadata_lte={"date": "2024-12-31"},
        )
        results = await self._store.text_search("earnings", match_count=20, metadata_filter=f)
        for r in results:
            assert r.metadata.get("quarter") == "Q4"
