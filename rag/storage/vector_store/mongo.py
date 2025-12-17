"""MongoDB Atlas vector store implementation with hybrid search."""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from pymongo import AsyncMongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, OperationFailure
from bson import ObjectId

from rag.config.settings import load_settings
from rag.ingestion.models import SearchResult, ChunkData

logger = logging.getLogger(__name__)


class MongoHybridStore:
    """MongoDB Atlas implementation with hybrid vector + text search."""

    def __init__(self):
        """Initialize MongoDB connection."""
        self.settings = load_settings()
        self.client: Optional[AsyncMongoClient] = None
        self.db = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize MongoDB connection."""
        if self._initialized:
            return

        try:
            self.client = AsyncMongoClient(
                self.settings.mongodb_uri,
                serverSelectionTimeoutMS=5000
            )
            self.db = self.client[self.settings.mongodb_database]

            # Verify connection
            await self.client.admin.command("ping")
            logger.info(f"Connected to MongoDB: {self.settings.mongodb_database}")
            self._initialized = True

        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise

    async def close(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            await self.client.close()
            self.client = None
            self.db = None
            self._initialized = False
            logger.info("MongoDB connection closed")

    async def add(
        self,
        chunks: List[ChunkData],
        document_id: str
    ) -> None:
        """
        Store document chunks with embeddings.

        Args:
            chunks: List of chunks with embeddings
            document_id: Parent document ID
        """
        await self.initialize()

        collection = self.db[self.settings.mongodb_collection_chunks]

        chunk_dicts = []
        for chunk in chunks:
            chunk_dict = {
                "document_id": ObjectId(document_id),
                "content": chunk.content,
                "embedding": chunk.embedding,
                "chunk_index": chunk.index,
                "metadata": chunk.metadata,
                "token_count": chunk.token_count,
                "created_at": datetime.now()
            }
            chunk_dicts.append(chunk_dict)

        if chunk_dicts:
            await collection.insert_many(chunk_dicts, ordered=False)
            logger.info(f"Inserted {len(chunk_dicts)} chunks")

    async def semantic_search(
        self,
        query_embedding: List[float],
        match_count: Optional[int] = None
    ) -> List[SearchResult]:
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

        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.settings.mongodb_vector_index,
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": 100,
                    "limit": match_count
                }
            },
            {
                "$lookup": {
                    "from": self.settings.mongodb_collection_documents,
                    "localField": "document_id",
                    "foreignField": "_id",
                    "as": "document_info"
                }
            },
            {"$unwind": "$document_info"},
            {
                "$project": {
                    "chunk_id": "$_id",
                    "document_id": 1,
                    "content": 1,
                    "similarity": {"$meta": "vectorSearchScore"},
                    "metadata": 1,
                    "document_title": "$document_info.title",
                    "document_source": "$document_info.source"
                }
            }
        ]

        try:
            collection = self.db[self.settings.mongodb_collection_chunks]
            cursor = collection.aggregate(pipeline)
            results = [doc async for doc in cursor][:match_count]

            return [
                SearchResult(
                    chunk_id=str(doc["chunk_id"]),
                    document_id=str(doc["document_id"]),
                    content=doc["content"],
                    similarity=doc["similarity"],
                    metadata=doc.get("metadata", {}),
                    document_title=doc["document_title"],
                    document_source=doc["document_source"]
                )
                for doc in results
            ]

        except OperationFailure as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    async def text_search(
        self,
        query: str,
        match_count: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Perform full-text search using MongoDB Atlas Search.

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

        pipeline = [
            {
                "$search": {
                    "index": self.settings.mongodb_text_index,
                    "text": {
                        "query": query,
                        "path": "content",
                        "fuzzy": {
                            "maxEdits": 2,
                            "prefixLength": 3
                        }
                    }
                }
            },
            {"$limit": match_count * 2},
            {
                "$lookup": {
                    "from": self.settings.mongodb_collection_documents,
                    "localField": "document_id",
                    "foreignField": "_id",
                    "as": "document_info"
                }
            },
            {"$unwind": "$document_info"},
            {
                "$project": {
                    "chunk_id": "$_id",
                    "document_id": 1,
                    "content": 1,
                    "similarity": {"$meta": "searchScore"},
                    "metadata": 1,
                    "document_title": "$document_info.title",
                    "document_source": "$document_info.source"
                }
            }
        ]

        try:
            collection = self.db[self.settings.mongodb_collection_chunks]
            cursor = collection.aggregate(pipeline)
            results = [doc async for doc in cursor][:match_count * 2]

            return [
                SearchResult(
                    chunk_id=str(doc["chunk_id"]),
                    document_id=str(doc["document_id"]),
                    content=doc["content"],
                    similarity=doc["similarity"],
                    metadata=doc.get("metadata", {}),
                    document_title=doc["document_title"],
                    document_source=doc["document_source"]
                )
                for doc in results
            ]

        except OperationFailure as e:
            logger.error(f"Text search failed: {e}")
            return []

    async def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        match_count: Optional[int] = None
    ) -> List[SearchResult]:
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
            return_exceptions=True
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
            [semantic_results, text_results],
            k=60
        )

        return merged_results[:match_count]

    def _reciprocal_rank_fusion(
        self,
        search_results_list: List[List[SearchResult]],
        k: int = 60
    ) -> List[SearchResult]:
        """
        Merge multiple ranked lists using Reciprocal Rank Fusion.

        Args:
            search_results_list: List of ranked result lists
            k: RRF constant (default: 60)

        Returns:
            Unified list sorted by combined RRF score
        """
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, SearchResult] = {}

        for results in search_results_list:
            for rank, result in enumerate(results):
                chunk_id = result.chunk_id
                rrf_score = 1.0 / (k + rank)

                if chunk_id in rrf_scores:
                    rrf_scores[chunk_id] += rrf_score
                else:
                    rrf_scores[chunk_id] = rrf_score
                    chunk_map[chunk_id] = result

        sorted_chunks = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

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
                document_source=result.document_source
            )
            merged_results.append(merged_result)

        logger.info(f"RRF merged {len(search_results_list)} lists into {len(merged_results)} results")
        return merged_results

    async def save_document(
        self,
        title: str,
        source: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Save a document to MongoDB.

        Args:
            title: Document title
            source: Document source path
            content: Document content
            metadata: Document metadata

        Returns:
            Document ID (ObjectId as string)
        """
        await self.initialize()

        collection = self.db[self.settings.mongodb_collection_documents]

        document_dict = {
            "title": title,
            "source": source,
            "content": content,
            "metadata": metadata,
            "created_at": datetime.now()
        }

        result = await collection.insert_one(document_dict)
        logger.info(f"Saved document with ID: {result.inserted_id}")
        return str(result.inserted_id)

    async def clean_collections(self) -> None:
        """Clean all data from collections."""
        await self.initialize()

        chunks_collection = self.db[self.settings.mongodb_collection_chunks]
        docs_collection = self.db[self.settings.mongodb_collection_documents]

        chunks_result = await chunks_collection.delete_many({})
        docs_result = await docs_collection.delete_many({})

        logger.info(f"Deleted {chunks_result.deleted_count} chunks, {docs_result.deleted_count} documents")
