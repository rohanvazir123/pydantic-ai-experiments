#!/usr/bin/env python
# coding: utf-8







import asyncio
import logging
from typing import Optional, List, Dict, Any
from pydantic_ai import RunContext
from pydantic import BaseModel, Field
from pymongo.errors import OperationFailure
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from dotenv import load_dotenv
from typing import Optional

from typing import Optional
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging
from pymongo import AsyncMongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import openai


from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
from typing import Optional
from pydantic_ai.ag_ui import StateDeps

import os
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument

from typing import List, Optional
from datetime import datetime

from dotenv import load_dotenv
import openai

import os
import asyncio
import logging
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import argparse
from dataclasses import dataclass

from pymongo import AsyncMongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from bson import ObjectId
from dotenv import load_dotenv




logger = logging.getLogger(__name__)




# Load environment variables from .env file
load_dotenv()
import os
cwd=os.getcwd()
print(cwd)
print(os.listdir(cwd))
import pprint
from dotenv import dotenv_values
pprint.pprint(dotenv_values())




# Settings configuration for MongoDB RAG Agent.

class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = ConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # MongoDB Configuration
    mongodb_uri: str = Field(..., description="MongoDB Atlas connection string")

    mongodb_database: str = Field(default="rag_db", description="MongoDB database name")

    mongodb_collection_documents: str = Field(
        default="documents", description="Collection for source documents"
    )

    mongodb_collection_chunks: str = Field(
        default="chunks", description="Collection for document chunks with embeddings"
    )

    mongodb_vector_index: str = Field(
        default="vector_index",
        description="Vector search index name (must be created in Atlas UI)",
    )

    mongodb_text_index: str = Field(
        default="text_index",
        description="Full-text search index name (must be created in Atlas UI)",
    )

    # LLM Configuration (OpenAI-compatible)
    llm_provider: str = Field(
        default="openrouter",
        description="LLM provider (openai, anthropic, gemini, ollama, etc.)",
    )

    llm_api_key: str = Field(..., description="API key for the LLM provider")

    llm_model: str = Field(
        default="anthropic/claude-haiku-4.5",
        description="Model to use for search and summarization",
    )

    llm_base_url: Optional[str] = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for the LLM API (for OpenAI-compatible providers)",
    )

    # Embedding Configuration
    embedding_provider: str = Field(default="openai", description="Embedding provider")

    embedding_api_key: str = Field(..., description="API key for embedding provider")

    embedding_model: str = Field(
        default="text-embedding-3-small", description="Embedding model to use"
    )

    embedding_base_url: Optional[str] = Field(
        default="https://api.openai.com/v1", description="Base URL for embedding API"
    )

    embedding_dimension: int = Field(
        default=1536,
        description="Embedding vector dimension (1536 for text-embedding-3-small)",
    )

    # Search Configuration
    default_match_count: int = Field(
        default=10, description="Default number of search results to return"
    )

    max_match_count: int = Field(
        default=50, description="Maximum number of search results allowed"
    )

    default_text_weight: float = Field(
        default=0.3, description="Default text weight for hybrid search (0-1)"
    )




def load_settings() -> Settings:
    """Load settings with proper error handling."""
    try:
        return Settings()
    except Exception as e:
        error_msg = f"Failed to load settings: {e}"
        if "mongodb_uri" in str(e).lower():
            error_msg += "\nMake sure to set MONGODB_URI in your .env file"
        if "llm_api_key" in str(e).lower():
            error_msg += "\nMake sure to set LLM_API_KEY in your .env file"
        if "embedding_api_key" in str(e).lower():
            error_msg += "\nMake sure to set EMBEDDING_API_KEY in your .env file"
        raise ValueError(error_msg) from e




"""Model providers for Semantic Search Agent."""

def get_llm_model(model_choice: Optional[str] = None) -> OpenAIModel:
    """
    Get LLM model configuration based on environment variables.
    Supports any OpenAI-compatible API provider.

    Args:
        model_choice: Optional override for model choice

    Returns:
        Configured OpenAI-compatible model
    """
    settings = load_settings()

    llm_choice = model_choice or settings.llm_model
    base_url = settings.llm_base_url
    api_key = settings.llm_api_key

    # Create provider based on configuration
    provider = OpenAIProvider(base_url=base_url, api_key=api_key)

    return OpenAIModel(llm_choice, provider=provider)


def get_embedding_model() -> OpenAIModel:
    """
    Get embedding model configuration.
    Uses OpenAI embeddings API (or compatible provider).

    Returns:
        Configured embedding model
    """
    settings = load_settings()

    # For embeddings, use the same provider configuration
    provider = OpenAIProvider(
        base_url=settings.llm_base_url, api_key=settings.llm_api_key
    )

    return OpenAIModel(settings.embedding_model, provider=provider)


def get_model_info() -> dict:
    """
    Get information about current model configuration.

    Returns:
        Dictionary with model configuration info
    """
    settings = load_settings()

    return {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "llm_base_url": settings.llm_base_url,
        "embedding_model": settings.embedding_model,
    }


def validate_llm_configuration() -> bool:
    """
    Validate that LLM configuration is properly set.

    Returns:
        True if configuration is valid
    """
    try:
        # Check if we can create a model instance
        get_llm_model()
        return True
    except Exception as e:
        print(f"LLM configuration validation failed: {e}")
        return False




"""Configuration validation script for MongoDB RAG Agent."""


def mask_credential(value: str) -> str:
    """Mask credentials for safe display."""
    if not value or len(value) < 8:
        return "***"
    return value[:4] + "..." + value[-4:]


def validate_config() -> bool:
    """
    Validate configuration and display settings.

    Returns:
        True if configuration is valid, False otherwise
    """
    try:

        print("MongoDB RAG Agent - Configuration Validation")


        # Load settings
        print("[1/4] Loading settings...")
        settings = load_settings()
        print("[OK] Settings loaded successfully")
        print()

        # Validate MongoDB configuration
        print("[2/4] Validating MongoDB configuration...")
        print(f"  MongoDB URI: {mask_credential(settings.mongodb_uri)}")
        print(f"  Database: {settings.mongodb_database}")
        print(f"  Documents Collection: {settings.mongodb_collection_documents}")
        print(f"  Chunks Collection: {settings.mongodb_collection_chunks}")
        print(f"  Vector Index: {settings.mongodb_vector_index}")
        print(f"  Text Index: {settings.mongodb_text_index}")
        print("[OK] MongoDB configuration present")
        print()

        # Validate LLM configuration
        print("[3/4] Validating LLM configuration...")
        model_info = get_model_info()
        print(f"  Provider: {model_info['llm_provider']}")
        print(f"  Model: {model_info['llm_model']}")
        print(f"  Base URL: {model_info['llm_base_url']}")
        print(f"  API Key: {mask_credential(settings.llm_api_key)}")
        print("[OK] LLM configuration present")
        print()

        # Validate Embedding configuration
        print("[4/4] Validating Embedding configuration...")
        print(f"  Provider: {settings.embedding_provider}")
        print(f"  Model: {settings.embedding_model}")
        print(f"  Dimension: {settings.embedding_dimension}")
        print(f"  API Key: {mask_credential(settings.embedding_api_key)}")
        print("[OK] Embedding configuration present")
        print()

        # Success summary
        print("=" * 60)
        print("[OK] ALL CONFIGURATION CHECKS PASSED")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Add documents to the ./documents/ folder")
        print("2. Run ingestion: uv run python -m src.ingestion.ingest -d ./documents")
        print("3. Create search indexes in MongoDB Atlas (after ingestion completes)")
        print("   See README.md for index creation instructions")
        print()

        return True

    except ValueError as e:
        print()
        print("=" * 60)
        print("[FAIL] CONFIGURATION VALIDATION FAILED")
        print("=" * 60)
        print()
        print(f"Error: {e}")
        print()
        print("Please check your .env file and ensure all required variables are set.")
        print("See .env.example for required variables.")
        print()
        return False

    except Exception as e:
        print()
        print("=" * 60)
        print("[FAIL] UNEXPECTED ERROR")
        print("=" * 60)
        print()
        print(f"Error: {e}")
        print()
        import traceback

        traceback.print_exc()
        return False




success = validate_config()
print(success)




"""
Docling HybridChunker implementation for intelligent document splitting.

This module uses Docling's built-in HybridChunker which combines:
- Token-aware chunking (uses actual tokenizer)
- Document structure preservation (headings, sections, tables)
- Semantic boundary respect (paragraphs, code blocks)
- Contextualized output (chunks include heading hierarchy)

Benefits over custom chunking:
- Fast (no LLM API calls)
- Token-precise (not character-based estimates)
- Better for RAG (chunks include document context)
- Battle-tested (maintained by Docling team)
"""




@dataclass
class ChunkingConfig:
    """Configuration for DoclingHybridChunker."""
    chunk_size: int = 1000  # Target characters per chunk (used in fallback)
    chunk_overlap: int = 200  # Character overlap between chunks (used in fallback)
    max_chunk_size: int = 2000  # Maximum chunk size (used in fallback)
    min_chunk_size: int = 100  # Minimum chunk size (used in fallback)
    max_tokens: int = 512  # Maximum tokens for embedding models

    def __post_init__(self):
        """Validate configuration."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        if self.min_chunk_size <= 0:
            raise ValueError("Minimum chunk size must be positive")


@dataclass
class DocumentChunk:
    """Represents a document chunk with optional embedding."""
    content: str
    index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]
    token_count: Optional[int] = None
    embedding: Optional[List[float]] = None  # For embedder compatibility

    def __post_init__(self):
        """Calculate token count if not provided."""
        if self.token_count is None:
            # Rough estimation: ~4 characters per token
            self.token_count = len(self.content) // 4


class DoclingHybridChunker:
    """
    Docling HybridChunker wrapper for intelligent document splitting.

    This chunker uses Docling's built-in HybridChunker which:
    - Respects document structure (sections, paragraphs, tables)
    - Is token-aware (fits embedding model limits)
    - Preserves semantic coherence
    - Includes heading context in chunks
    """

    def __init__(self, config: ChunkingConfig):
        """
        Initialize chunker.

        Args:
            config: Chunking configuration
        """
        self.config = config

        # Initialize tokenizer for token-aware chunking
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        logger.info(f"Initializing tokenizer: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Create HybridChunker
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=config.max_tokens,
            merge_peers=True  # Merge small adjacent chunks
        )

        logger.info(f"HybridChunker initialized (max_tokens={config.max_tokens})")

    async def chunk_document(
        self,
        content: str,
        title: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        docling_doc: Optional[DoclingDocument] = None
    ) -> List[DocumentChunk]:
        """
        Chunk a document using Docling's HybridChunker.

        Args:
            content: Document content (markdown format)
            title: Document title
            source: Document source
            metadata: Additional metadata
            docling_doc: Optional pre-converted DoclingDocument (for efficiency)

        Returns:
            List of document chunks with contextualized content
        """
        if not content.strip():
            return []

        base_metadata = {
            "title": title,
            "source": source,
            "chunk_method": "hybrid",
            **(metadata or {})
        }

        # If we don't have a DoclingDocument, we need to create one from markdown
        if docling_doc is None:
            # For markdown content, we need to convert it to DoclingDocument
            # This is a simplified version - in practice, content comes from
            # Docling's document converter in the ingestion pipeline
            logger.warning("No DoclingDocument provided, using simple chunking fallback")
            return self._simple_fallback_chunk(content, base_metadata)

        try:
            # Use HybridChunker to chunk the DoclingDocument
            chunk_iter = self.chunker.chunk(dl_doc=docling_doc)
            chunks = list(chunk_iter)

            # Convert Docling chunks to DocumentChunk objects
            document_chunks = []
            current_pos = 0

            for i, chunk in enumerate(chunks):
                # Get contextualized text (includes heading hierarchy)
                contextualized_text = self.chunker.contextualize(chunk=chunk)

                # Count actual tokens
                token_count = len(self.tokenizer.encode(contextualized_text))

                # Create chunk metadata
                chunk_metadata = {
                    **base_metadata,
                    "total_chunks": len(chunks),
                    "token_count": token_count,
                    "has_context": True  # Flag indicating contextualized chunk
                }

                # Estimate character positions
                start_char = current_pos
                end_char = start_char + len(contextualized_text)

                document_chunks.append(DocumentChunk(
                    content=contextualized_text.strip(),
                    index=i,
                    start_char=start_char,
                    end_char=end_char,
                    metadata=chunk_metadata,
                    token_count=token_count
                ))

                current_pos = end_char

            logger.info(f"Created {len(document_chunks)} chunks using HybridChunker")
            return document_chunks

        except Exception as e:
            logger.error(f"HybridChunker failed: {e}, falling back to simple chunking")
            return self._simple_fallback_chunk(content, base_metadata)

    def _simple_fallback_chunk(
        self,
        content: str,
        base_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Simple fallback chunking when HybridChunker can't be used.

        This is used when:
        - No DoclingDocument is provided
        - HybridChunker fails

        Args:
            content: Content to chunk
            base_metadata: Base metadata for chunks

        Returns:
            List of document chunks
        """
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        # Simple sliding window approach
        start = 0
        chunk_index = 0

        while start < len(content):
            end = start + chunk_size

            if end >= len(content):
                # Last chunk
                chunk_text = content[start:]
            else:
                # Try to end at sentence boundary
                chunk_end = end
                for i in range(end, max(start + self.config.min_chunk_size, end - 200), -1):
                    if i < len(content) and content[i] in '.!?\n':
                        chunk_end = i + 1
                        break
                chunk_text = content[start:chunk_end]
                end = chunk_end

            if chunk_text.strip():
                token_count = len(self.tokenizer.encode(chunk_text))

                chunks.append(DocumentChunk(
                    content=chunk_text.strip(),
                    index=chunk_index,
                    start_char=start,
                    end_char=end,
                    metadata={
                        **base_metadata,
                        "chunk_method": "simple_fallback",
                        "total_chunks": -1  # Will update after
                    },
                    token_count=token_count
                ))

                chunk_index += 1

            # Move forward with overlap
            start = end - overlap

        # Update total chunks
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)

        logger.info(f"Created {len(chunks)} chunks using simple fallback")
        return chunks


def create_chunker(config: ChunkingConfig):
    """
    Create DoclingHybridChunker for intelligent document splitting.

    Args:
        config: Chunking configuration

    Returns:
        DoclingHybridChunker instance
    """
    return DoclingHybridChunker(config)




"""System prompts for MongoDB RAG Agent."""

MAIN_SYSTEM_PROMPT = """You are a helpful assistant with access to a knowledge base that you can search when needed.

ALWAYS Start with Hybrid search

## Your Capabilities:
1. **Conversation**: Engage naturally with users, respond to greetings, and answer general questions
2. **Semantic Search**: When users ask for information from the knowledge base, use hybrid_search for conceptual queries
3. **Hybrid Search**: For specific facts or technical queries, use hybrid_search
4. **Information Synthesis**: Transform search results into coherent responses

## When to Search:
- ONLY search when users explicitly ask for information that would be in the knowledge base
- For greetings (hi, hello, hey) → Just respond conversationally, no search needed
- For general questions about yourself → Answer directly, no search needed
- For requests about specific topics or information → Use the appropriate search tool

## Search Strategy (when searching):
- Conceptual/thematic queries → Use hybrid_search
- Specific facts/technical terms → Use hybrid_search with appropriate text_weight
- Start with lower match_count (5-10) for focused results

## Response Guidelines:
- Be conversational and natural
- Only cite sources when you've actually performed a search
- If no search is needed, just respond directly
- Be helpful and friendly

Remember: Not every interaction requires a search. Use your judgment about when to search the knowledge base."""




"""Dependencies for MongoDB RAG Agent."""

@dataclass
class LLMongoAgentContextDependencies:
    """LLMMongoContext dependencies to be inserted into the agent context."""

    # Core dependencies
    mongo_client: Optional[AsyncMongoClient] = None
    db: Optional[Any] = None
    openai_client: Optional[openai.AsyncOpenAI] = None
    settings: Optional[Any] = None

    # Session context
    session_id: Optional[str] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    query_history: list = field(default_factory=list)

    async def initialize(self) -> None:
        """
        Initialize external connections.

        Raises:
            ConnectionFailure: If MongoDB connection fails
            ServerSelectionTimeoutError: If MongoDB server selection times out
            ValueError: If settings cannot be loaded
        """
        if not self.settings:
            self.settings = load_settings()
            logger.info("settings_loaded", database=self.settings.mongodb_database)

        # Initialize MongoDB client
        if not self.mongo_client:
            try:
                self.mongo_client = AsyncMongoClient(
                    self.settings.mongodb_uri, serverSelectionTimeoutMS=5000
                )
                self.db = self.mongo_client[self.settings.mongodb_database]

                # Verify connection with ping
                await self.mongo_client.admin.command("ping")
                logger.info(
                    "mongodb_connected",
                    database=self.settings.mongodb_database,
                    collections={
                        "documents": self.settings.mongodb_collection_documents,
                        "chunks": self.settings.mongodb_collection_chunks,
                    },
                )
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                logger.exception("mongodb_connection_failed", error=str(e))
                raise

        # Initialize OpenAI client for embeddings
        if not self.openai_client:
            self.openai_client = openai.AsyncOpenAI(
                api_key=self.settings.embedding_api_key,
                base_url=self.settings.embedding_base_url,
            )
            logger.info(
                "openai_client_initialized",
                model=self.settings.embedding_model,
                dimension=self.settings.embedding_dimension,
            )

    async def cleanup(self) -> None:
        """Clean up external connections."""
        if self.mongo_client:
            await self.mongo_client.close()
            self.mongo_client = None
            self.db = None
            logger.info("mongodb_connection_closed")

    async def get_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for text using OpenAI.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            Exception: If embedding generation fails
        """
        if not self.openai_client:
            await self.initialize()

        response = await self.openai_client.embeddings.create(
            model=self.settings.embedding_model, input=text
        )
        # Return as list of floats - MongoDB stores as native array
        return response.data[0].embedding

    def set_user_preference(self, key: str, value: Any) -> None:
        """
        Set a user preference for the session.

        Args:
            key: Preference key
            value: Preference value
        """
        self.user_preferences[key] = value

    def add_to_history(self, query: str) -> None:
        """
        Add a query to the search history.

        Args:
            query: Search query to add to history
        """
        self.query_history.append(query)
        # Keep only last 10 queries
        if len(self.query_history) > 10:
            self.query_history.pop(0)




"""Search tools for MongoDB RAG Agent."""

class SearchResult(BaseModel):
    """Model for search results."""

    chunk_id: str = Field(..., description="MongoDB ObjectId of chunk as string")
    document_id: str = Field(..., description="Parent document ObjectId as string")
    content: str = Field(..., description="Chunk text content")
    similarity: float = Field(..., description="Relevance score (0-1)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    document_title: str = Field(..., description="Title from document lookup")
    document_source: str = Field(..., description="Source from document lookup")


async def semantic_search(
    ctx: RunContext[LLMongoAgentContextDependencies],
    query: str,
    match_count: Optional[int] = None
) -> List[SearchResult]:
    """
    Perform pure semantic search using MongoDB vector similarity.

    Args:
        ctx: Agent runtime context with dependencies
        query: Search query text
        match_count: Number of results to return (default: 10)

    Returns:
        List of search results ordered by similarity

    Raises:
        OperationFailure: If MongoDB operation fails (e.g., missing index)
    """
    try:
        deps = ctx.deps

        # Use default if not specified
        if match_count is None:
            match_count = deps.settings.default_match_count

        # Validate match count
        match_count = min(match_count, deps.settings.max_match_count)

        # Generate embedding for query (already returns list[float])
        query_embedding = await deps.get_embedding(query)

        # Build MongoDB aggregation pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": deps.settings.mongodb_vector_index,
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": 100,  # Search space (10x limit is good default)
                    "limit": match_count
                }
            },
            {
                "$lookup": {
                    "from": deps.settings.mongodb_collection_documents,
                    "localField": "document_id",
                    "foreignField": "_id",
                    "as": "document_info"
                }
            },
            {
                "$unwind": "$document_info"
            },
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

        # Execute aggregation
        collection = deps.db[deps.settings.mongodb_collection_chunks]
        cursor = await collection.aggregate(pipeline)
        results = [doc async for doc in cursor][:match_count]

        # Convert to SearchResult objects (ObjectId → str conversion)
        search_results = [
            SearchResult(
                chunk_id=str(doc['chunk_id']),
                document_id=str(doc['document_id']),
                content=doc['content'],
                similarity=doc['similarity'],
                metadata=doc.get('metadata', {}),
                document_title=doc['document_title'],
                document_source=doc['document_source']
            )
            for doc in results
        ]

        logger.info(
            f"semantic_search_completed: query={query}, results={len(search_results)}, match_count={match_count}"
        )

        return search_results

    except OperationFailure as e:
        error_code = e.code if hasattr(e, 'code') else None
        logger.error(
            f"semantic_search_failed: query={query}, error={str(e)}, code={error_code}"
        )
        # Return empty list on error (graceful degradation)
        return []
    except Exception as e:
        logger.exception(f"semantic_search_error: query={query}, error={str(e)}")
        return []


async def text_search(
    ctx: RunContext[LLMongoAgentContextDependencies],
    query: str,
    match_count: Optional[int] = None
) -> List[SearchResult]:
    """
    Perform full-text search using MongoDB Atlas Search.

    Uses $search operator for keyword matching, fuzzy matching, and phrase matching.
    Works on all Atlas tiers including M0 (free tier).

    Args:
        ctx: Agent runtime context with dependencies
        query: Search query text
        match_count: Number of results to return (default: 10)

    Returns:
        List of search results ordered by text relevance

    Raises:
        OperationFailure: If MongoDB operation fails (e.g., missing index)
    """
    try:
        deps = ctx.deps

        # Use default if not specified
        if match_count is None:
            match_count = deps.settings.default_match_count

        # Validate match count
        match_count = min(match_count, deps.settings.max_match_count)

        # Build MongoDB Atlas Search aggregation pipeline
        pipeline = [
            {
                "$search": {
                    "index": deps.settings.mongodb_text_index,
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
            {
                "$limit": match_count * 2  # Over-fetch for better RRF results
            },
            {
                "$lookup": {
                    "from": deps.settings.mongodb_collection_documents,
                    "localField": "document_id",
                    "foreignField": "_id",
                    "as": "document_info"
                }
            },
            {
                "$unwind": "$document_info"
            },
            {
                "$project": {
                    "chunk_id": "$_id",
                    "document_id": 1,
                    "content": 1,
                    "similarity": {"$meta": "searchScore"},  # Text relevance score
                    "metadata": 1,
                    "document_title": "$document_info.title",
                    "document_source": "$document_info.source"
                }
            }
        ]

        # Execute aggregation
        collection = deps.db[deps.settings.mongodb_collection_chunks]
        cursor = await collection.aggregate(pipeline)
        results = [doc async for doc in cursor][:match_count * 2]

        # Convert to SearchResult objects (ObjectId → str conversion)
        search_results = [
            SearchResult(
                chunk_id=str(doc['chunk_id']),
                document_id=str(doc['document_id']),
                content=doc['content'],
                similarity=doc['similarity'],
                metadata=doc.get('metadata', {}),
                document_title=doc['document_title'],
                document_source=doc['document_source']
            )
            for doc in results
        ]

        logger.info(
            f"text_search_completed: query={query}, results={len(search_results)}, match_count={match_count}"
        )

        return search_results

    except OperationFailure as e:
        error_code = e.code if hasattr(e, 'code') else None
        logger.error(
            f"text_search_failed: query={query}, error={str(e)}, code={error_code}"
        )
        # Return empty list on error (graceful degradation)
        return []
    except Exception as e:
        logger.exception(f"text_search_error: query={query}, error={str(e)}")
        return []


def reciprocal_rank_fusion(
    search_results_list: List[List[SearchResult]],
    k: int = 60
) -> List[SearchResult]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF is a simple yet effective algorithm for combining results from different
    search methods. It works by scoring each document based on its rank position
    in each result list.

    Args:
        search_results_list: List of ranked result lists from different searches
        k: RRF constant (default: 60, standard in literature)

    Returns:
        Unified list of results sorted by combined RRF score

    Algorithm:
        For each document d appearing in result lists:
            RRF_score(d) = Σ(1 / (k + rank_i(d)))
        Where rank_i(d) is the position of document d in result list i.

    References:
        - Cormack et al. (2009): "Reciprocal Rank Fusion outperforms the best system"
        - Standard k=60 performs well across various datasets
    """
    # Build score dictionary by chunk_id
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, SearchResult] = {}

    # Process each search result list
    for results in search_results_list:
        for rank, result in enumerate(results):
            chunk_id = result.chunk_id

            # Calculate RRF contribution: 1 / (k + rank)
            rrf_score = 1.0 / (k + rank)

            # Accumulate score (automatic deduplication)
            if chunk_id in rrf_scores:
                rrf_scores[chunk_id] += rrf_score
            else:
                rrf_scores[chunk_id] = rrf_score
                chunk_map[chunk_id] = result

    # Sort by combined RRF score (descending)
    sorted_chunks = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Build final result list with updated similarity scores
    merged_results = []
    for chunk_id, rrf_score in sorted_chunks:
        result = chunk_map[chunk_id]
        # Create new result with updated similarity (RRF score)
        merged_result = SearchResult(
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            content=result.content,
            similarity=rrf_score,  # Combined RRF score
            metadata=result.metadata,
            document_title=result.document_title,
            document_source=result.document_source
        )
        merged_results.append(merged_result)

    logger.info(f"RRF merged {len(search_results_list)} result lists into {len(merged_results)} unique results")

    return merged_results


async def hybrid_search(
    ctx: RunContext[LLMongoAgentContextDependencies],
    query: str,
    match_count: Optional[int] = None,
    text_weight: Optional[float] = None
) -> List[SearchResult]:
    """
    Perform hybrid search combining semantic and keyword matching.

    Uses manual Reciprocal Rank Fusion (RRF) to merge vector and text search results.
    Works on all Atlas tiers including M0 (free tier) - no M10+ required!

    Args:
        ctx: Agent runtime context with dependencies
        query: Search query text
        match_count: Number of results to return (default: 10)
        text_weight: Weight for text matching (0-1, not used with RRF)

    Returns:
        List of search results sorted by combined RRF score

    Algorithm:
        1. Run semantic search (vector similarity)
        2. Run text search (keyword/fuzzy matching)
        3. Merge results using Reciprocal Rank Fusion
        4. Return top N results by combined score
    """
    try:
        deps = ctx.deps

        # Use defaults if not specified
        if match_count is None:
            match_count = deps.settings.default_match_count

        # Validate match count
        match_count = min(match_count, deps.settings.max_match_count)

        # Over-fetch for better RRF results (2x requested count)
        fetch_count = match_count * 2

        logger.info(f"hybrid_search starting: query='{query}', match_count={match_count}")

        # Run both searches concurrently for performance
        semantic_results, text_results = await asyncio.gather(
            semantic_search(ctx, query, fetch_count),
            text_search(ctx, query, fetch_count),
            return_exceptions=True  # Don't fail if one search errors
        )

        # Handle errors gracefully
        if isinstance(semantic_results, Exception):
            logger.warning(f"Semantic search failed: {semantic_results}, using text results only")
            semantic_results = []
        if isinstance(text_results, Exception):
            logger.warning(f"Text search failed: {text_results}, using semantic results only")
            text_results = []

        # If both failed, return empty
        if not semantic_results and not text_results:
            logger.error("Both semantic and text search failed")
            return []

        # Merge results using Reciprocal Rank Fusion
        merged_results = reciprocal_rank_fusion(
            [semantic_results, text_results],
            k=60  # Standard RRF constant
        )

        # Return top N results
        final_results = merged_results[:match_count]

        logger.info(
            f"hybrid_search_completed: query='{query}', "
            f"semantic={len(semantic_results)}, text={len(text_results)}, "
            f"merged={len(merged_results)}, returned={len(final_results)}"
        )

        return final_results

    except Exception as e:
        logger.exception(f"hybrid_search_error: query={query}, error={str(e)}")
        # Graceful degradation: try semantic-only as last resort
        try:
            logger.info("Falling back to semantic search only")
            return await semantic_search(ctx, query, match_count)
        except:
            return []




"""Main MongoDB RAG agent implementation with shared state."""

class RAGState(BaseModel):
    """Minimal shared state for the RAG agent."""
    pass


# Create the RAG agent with AGUI support
rag_agent = Agent(
    get_llm_model(),
    deps_type=StateDeps[RAGState],
    system_prompt=MAIN_SYSTEM_PROMPT
)


@rag_agent.tool
async def search_knowledge_base(
    ctx: RunContext[StateDeps[RAGState]],
    query: str,
    match_count: Optional[int] = 5,
    search_type: Optional[str] = "hybrid"
) -> str:
    """
    Search the knowledge base for relevant information.

    Args:
        ctx: Agent runtime context with state dependencies
        query: Search query text
        match_count: Number of results to return (default: 5)
        search_type: Type of search - "semantic" or "text" or "hybrid" (default: hybrid)

    Returns:
        String containing the retrieved information formatted for the LLM
    """
    try:
        # Initialize database connection
        agent_initializer = LLMMongoInitializer()
        await agent_initializer.initialize()

        # Create a context wrapper for the search tools
        class DepsWrapper:
            def __init__(self, deps):
                self.deps = deps

        deps_ctx = DepsWrapper(agent_deps)

        # Perform the search based on type
        if search_type == "hybrid":
            results = await hybrid_search(
                ctx=deps_ctx,
                query=query,
                match_count=match_count
            )
        elif search_type == "semantic":
            results = await semantic_search(
                ctx=deps_ctx,
                query=query,
                match_count=match_count
            )
        else:
            results = await text_search(
                ctx=deps_ctx,
                query=query,
                match_count=match_count
            )

        # Clean up
        await agent_initializer.cleanup()

        # Format results as a simple string
        if not results:
            return "No relevant information found in the knowledge base."

        # Build a formatted response
        response_parts = [f"Found {len(results)} relevant documents:\n"]

        for i, result in enumerate(results, 1):
            response_parts.append(f"\n--- Document {i}: {result.document_title} (relevance: {result.similarity:.2f}) ---")
            response_parts.append(result.content)

        return "\n".join(response_parts)

    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"




embedding_client = openai.AsyncOpenAI(
    api_key=settings.embedding_api_key,
    base_url=settings.embedding_base_url
)
EMBEDDING_MODEL = settings.embedding_model




class EmbeddingGenerator:
    """Generates embeddings for document chunks."""

    def __init__(
        self,
        model: str = EMBEDDING_MODEL,settings = load_settings()
        embedding_client = openai.AsyncOpenAI(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url
        )
        EMBEDDING_MODEL = settings.embedding_model

        batch_size: int = 100
    ):
        """
        Initialize embedding generator.

        Args:
            model: Embedding model to use
            batch_size: Number of texts to process in parallel
        """
        self.model = model
        self.batch_size = batch_size

        # Model-specific configurations
        self.model_configs = {
            "text-embedding-3-small": {"dimensions": 1536, "max_tokens": 8191},
            "text-embedding-3-large": {"dimensions": 3072, "max_tokens": 8191},
            "text-embedding-ada-002": {"dimensions": 1536, "max_tokens": 8191}
        }

        self.config = self.model_configs.get(
            model,
            {"dimensions": 1536, "max_tokens": 8191}
        )

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Truncate text if too long (rough estimation: 4 chars per token)
        if len(text) > self.config["max_tokens"] * 4:
            text = text[:self.config["max_tokens"] * 4]

        response = await embedding_client.embeddings.create(
            model=self.model,
            input=text
        )

        return response.data[0].embedding

    async def generate_embeddings_batch(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        # Truncate texts if too long
        processed_texts = []
        for text in texts:
            if len(text) > self.config["max_tokens"] * 4:
                text = text[:self.config["max_tokens"] * 4]
            processed_texts.append(text)

        response = await embedding_client.embeddings.create(
            model=self.model,
            input=processed_texts
        )

        return [data.embedding for data in response.data]

    async def embed_chunks(
        self,
        chunks: List[DocumentChunk],
        progress_callback: Optional[callable] = None
    ) -> List[DocumentChunk]:
        """
        Generate embeddings for document chunks.

        Args:
            chunks: List of document chunks
            progress_callback: Optional callback for progress updates

        Returns:
            Chunks with embeddings added
        """
        if not chunks:
            return chunks

        logger.info(f"Generating embeddings for {len(chunks)} chunks")

        # Process chunks in batches
        embedded_chunks = []
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i:i + self.batch_size]
            batch_texts = [chunk.content for chunk in batch_chunks]

            # Generate embeddings for this batch
            embeddings = await self.generate_embeddings_batch(batch_texts)

            # Add embeddings to chunks
            for chunk, embedding in zip(batch_chunks, embeddings):
                embedded_chunk = DocumentChunk(
                    content=chunk.content,
                    index=chunk.index,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    metadata={
                        **chunk.metadata,
                        "embedding_model": self.model,
                        "embedding_generated_at": datetime.now().isoformat()
                    },
                    token_count=chunk.token_count
                )
                embedded_chunk.embedding = embedding
                embedded_chunks.append(embedded_chunk)

            # Progress update
            current_batch = (i // self.batch_size) + 1
            if progress_callback:
                progress_callback(current_batch, total_batches)

            logger.info(f"Processed batch {current_batch}/{total_batches}")

        logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
        return embedded_chunks

    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.

        Args:
            query: Search query

        Returns:
            Query embedding
        """
        return await self.generate_embedding(query)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for this model."""
        return self.config["dimensions"]


def create_embedder(model: str = EMBEDDING_MODEL, **kwargs) -> EmbeddingGenerator:
    """
    Create embedding generator.

    Args:
        model: Embedding model to use
        **kwargs: Additional arguments for EmbeddingGenerator

    Returns:
        EmbeddingGenerator instance
    """
    return EmbeddingGenerator(model=model, **kwargs)




"""
Main ingestion script for processing documents into MongoDB vector database.
"""

logger = logging.getLogger(__name__)


@dataclass
class IngestionConfig:
    """Configuration for document ingestion."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunk_size: int = 2000
    max_tokens: int = 512


@dataclass
class IngestionResult:
    """Result of document ingestion."""
    document_id: str
    title: str
    chunks_created: int
    processing_time_ms: float
    errors: List[str]


class DocumentIngestionPipeline:
    """Pipeline for ingesting documents into MongoDB vector database."""

    def __init__(
        self,
        config: IngestionConfig,
        documents_folder: str = "documents",
        clean_before_ingest: bool = True
    ):
        """
        Initialize ingestion pipeline.

        Args:
            config: Ingestion configuration
            documents_folder: Folder containing documents
            clean_before_ingest: Whether to clean existing data before ingestion
        """
        self.config = config
        self.documents_folder = documents_folder
        self.clean_before_ingest = clean_before_ingest

        # Load settings
        self.settings = load_settings()

        # Initialize MongoDB client and database references
        self.mongo_client: Optional[AsyncMongoClient] = None
        self.db: Optional[Any] = None

        # Initialize components
        self.chunker_config = ChunkingConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            max_chunk_size=config.max_chunk_size,
            max_tokens=config.max_tokens
        )

        self.chunker = create_chunker(self.chunker_config)
        self.embedder = create_embedder()

        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize MongoDB connections.

        Raises:
            ConnectionFailure: If MongoDB connection fails
            ServerSelectionTimeoutError: If MongoDB server selection times out
        """
        if self._initialized:
            return

        logger.info("Initializing ingestion pipeline...")

        try:
            # Initialize MongoDB client
            self.mongo_client = AsyncMongoClient(
                self.settings.mongodb_uri,
                serverSelectionTimeoutMS=5000
            )
            self.db = self.mongo_client[self.settings.mongodb_database]

            # Verify connection
            await self.mongo_client.admin.command("ping")
            logger.info(
                f"Connected to MongoDB database: {self.settings.mongodb_database}"
            )

        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.exception("mongodb_connection_failed", error=str(e))
            raise

        self._initialized = True
        logger.info("Ingestion pipeline initialized")

    async def close(self) -> None:
        """Close MongoDB connections."""
        if self._initialized and self.mongo_client:
            await self.mongo_client.close()
            self.mongo_client = None
            self.db = None
            self._initialized = False
            logger.info("MongoDB connection closed")

    def _find_document_files(self) -> List[str]:
        """
        Find all supported document files in the documents folder.

        Returns:
            List of file paths
        """
        if not os.path.exists(self.documents_folder):
            logger.error(f"Documents folder not found: {self.documents_folder}")
            return []

        # Supported file patterns - Docling + text formats + audio
        patterns = [
            "*.md", "*.markdown", "*.txt",  # Text formats
            "*.pdf",  # PDF
            "*.docx", "*.doc",  # Word
            "*.pptx", "*.ppt",  # PowerPoint
            "*.xlsx", "*.xls",  # Excel
            "*.html", "*.htm",  # HTML
            "*.mp3", "*.wav", "*.m4a", "*.flac",  # Audio formats
        ]
        files = []

        for pattern in patterns:
            files.extend(
                glob.glob(
                    os.path.join(self.documents_folder, "**", pattern),
                    recursive=True
                )
            )

        return sorted(files)

    def _read_document(self, file_path: str) -> tuple[str, Optional[Any]]:
        """
        Read document content from file - supports multiple formats via Docling.

        Args:
            file_path: Path to the document file

        Returns:
            Tuple of (markdown_content, docling_document).
            docling_document is None only for text files.
        """
        file_ext = os.path.splitext(file_path)[1].lower()

        # Audio formats - transcribe with Whisper ASR
        audio_formats = ['.mp3', '.wav', '.m4a', '.flac']
        if file_ext in audio_formats:
            # Returns tuple: (markdown_content, docling_document)
            return self._transcribe_audio(file_path)

        # Docling-supported formats (convert to markdown)
        docling_formats = [
            '.pdf', '.docx', '.doc', '.pptx', '.ppt',
            '.xlsx', '.xls', '.html', '.htm',
            '.md', '.markdown'  # Markdown files for HybridChunker
        ]

        if file_ext in docling_formats:
            try:
                from docling.document_converter import DocumentConverter

                logger.info(
                    f"Converting {file_ext} file using Docling: "
                    f"{os.path.basename(file_path)}"
                )

                converter = DocumentConverter()
                result = converter.convert(file_path)

                # Export to markdown for consistent processing
                markdown_content = result.document.export_to_markdown()
                logger.info(
                    f"Successfully converted {os.path.basename(file_path)} "
                    f"to markdown"
                )

                # Return both markdown and DoclingDocument for HybridChunker
                return (markdown_content, result.document)

            except Exception as e:
                logger.error(f"Failed to convert {file_path} with Docling: {e}")
                # Fall back to raw text if Docling fails
                logger.warning(f"Falling back to raw text extraction for {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return (f.read(), None)
                except Exception:
                    return (
                        f"[Error: Could not read file {os.path.basename(file_path)}]",
                        None
                    )

        # Text-based formats (read directly)
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return (f.read(), None)
            except UnicodeDecodeError:
                # Try with different encoding
                with open(file_path, 'r', encoding='latin-1') as f:
                    return (f.read(), None)

    def _transcribe_audio(self, file_path: str) -> tuple[str, Optional[Any]]:
        """
        Transcribe audio file using Whisper ASR via Docling.

        Args:
            file_path: Path to the audio file

        Returns:
            Tuple of (markdown_content, docling_document)
        """
        try:
            from pathlib import Path
            from docling.document_converter import (
                DocumentConverter,
                AudioFormatOption
            )
            from docling.datamodel.pipeline_options import AsrPipelineOptions
            from docling.datamodel import asr_model_specs
            from docling.datamodel.base_models import InputFormat
            from docling.pipeline.asr_pipeline import AsrPipeline

            # Use Path object - Docling expects this
            audio_path = Path(file_path).resolve()
            logger.info(
                f"Transcribing audio file using Whisper Turbo: {audio_path.name}"
            )

            # Verify file exists
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            # Configure ASR pipeline with Whisper Turbo model
            pipeline_options = AsrPipelineOptions()
            pipeline_options.asr_options = asr_model_specs.WHISPER_TURBO

            converter = DocumentConverter(
                format_options={
                    InputFormat.AUDIO: AudioFormatOption(
                        pipeline_cls=AsrPipeline,
                        pipeline_options=pipeline_options,
                    )
                }
            )

            # Transcribe the audio file
            result = converter.convert(audio_path)

            # Export to markdown with timestamps
            markdown_content = result.document.export_to_markdown()
            logger.info(f"Successfully transcribed {os.path.basename(file_path)}")

            # Return both markdown and DoclingDocument for HybridChunker
            return (markdown_content, result.document)

        except Exception as e:
            logger.error(f"Failed to transcribe {file_path} with Whisper ASR: {e}")
            return (
                f"[Error: Could not transcribe audio file "
                f"{os.path.basename(file_path)}]",
                None
            )

    def _extract_title(self, content: str, file_path: str) -> str:
        """
        Extract title from document content or filename.

        Args:
            content: Document content
            file_path: Path to the document file

        Returns:
            Document title
        """
        # Try to find markdown title
        lines = content.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()

        # Fallback to filename
        return os.path.splitext(os.path.basename(file_path))[0]

    def _extract_document_metadata(
        self,
        content: str,
        file_path: str
    ) -> Dict[str, Any]:
        """
        Extract metadata from document content.

        Args:
            content: Document content
            file_path: Path to the document file

        Returns:
            Document metadata dictionary
        """
        metadata = {
            "file_path": file_path,
            "file_size": len(content),
            "ingestion_date": datetime.now().isoformat()
        }

        # Try to extract YAML frontmatter
        if content.startswith('---'):
            try:
                import yaml
                end_marker = content.find('\n---\n', 4)
                if end_marker != -1:
                    frontmatter = content[4:end_marker]
                    yaml_metadata = yaml.safe_load(frontmatter)
                    if isinstance(yaml_metadata, dict):
                        metadata.update(yaml_metadata)
            except ImportError:
                logger.warning(
                    "PyYAML not installed, skipping frontmatter extraction"
                )
            except Exception as e:
                logger.warning(f"Failed to parse frontmatter: {e}")

        # Extract some basic metadata from content
        lines = content.split('\n')
        metadata['line_count'] = len(lines)
        metadata['word_count'] = len(content.split())

        return metadata

    async def _save_to_mongodb(
        self,
        title: str,
        source: str,
        content: str,
        chunks: List[DocumentChunk],
        metadata: Dict[str, Any]
    ) -> str:
        """
        Save document and chunks to MongoDB.

        Args:
            title: Document title
            source: Document source path
            content: Document content
            chunks: List of document chunks with embeddings
            metadata: Document metadata

        Returns:
            Document ID (ObjectId as string)

        Raises:
            Exception: If MongoDB operations fail
        """
        # Get collection references
        documents_collection = self.db[
            self.settings.mongodb_collection_documents
        ]
        chunks_collection = self.db[self.settings.mongodb_collection_chunks]

        # Insert document
        document_dict = {
            "title": title,
            "source": source,
            "content": content,
            "metadata": metadata,
            "created_at": datetime.now()
        }

        document_result = await documents_collection.insert_one(document_dict)
        document_id = document_result.inserted_id

        logger.info(f"Inserted document with ID: {document_id}")

        # Insert chunks with embeddings as Python lists
        chunk_dicts = []
        for chunk in chunks:
            chunk_dict = {
                "document_id": document_id,
                "content": chunk.content,
                "embedding": chunk.embedding,  # Python list, NOT string!
                "chunk_index": chunk.index,
                "metadata": chunk.metadata,
                "token_count": chunk.token_count,
                "created_at": datetime.now()
            }
            chunk_dicts.append(chunk_dict)

        # Batch insert with ordered=False for partial success
        if chunk_dicts:
            await chunks_collection.insert_many(chunk_dicts, ordered=False)
            logger.info(f"Inserted {len(chunk_dicts)} chunks")

        return str(document_id)

    async def _clean_databases(self) -> None:
        """Clean existing data from MongoDB collections."""
        logger.warning("Cleaning existing data from MongoDB...")

        # Get collection references
        documents_collection = self.db[
            self.settings.mongodb_collection_documents
        ]
        chunks_collection = self.db[self.settings.mongodb_collection_chunks]

        # Delete all chunks first (to respect FK relationships)
        chunks_result = await chunks_collection.delete_many({})
        logger.info(f"Deleted {chunks_result.deleted_count} chunks")

        # Delete all documents
        docs_result = await documents_collection.delete_many({})
        logger.info(f"Deleted {docs_result.deleted_count} documents")

    async def _ingest_single_document(self, file_path: str) -> IngestionResult:
        """
        Ingest a single document.

        Args:
            file_path: Path to the document file

        Returns:
            Ingestion result
        """
        start_time = datetime.now()

        # Read document (returns tuple: content, docling_doc)
        document_content, docling_doc = self._read_document(file_path)
        document_title = self._extract_title(document_content, file_path)
        document_source = os.path.relpath(file_path, self.documents_folder)

        # Extract metadata from content
        document_metadata = self._extract_document_metadata(
            document_content,
            file_path
        )

        logger.info(f"Processing document: {document_title}")

        # Chunk the document - pass DoclingDocument for HybridChunker
        chunks = await self.chunker.chunk_document(
            content=document_content,
            title=document_title,
            source=document_source,
            metadata=document_metadata,
            docling_doc=docling_doc  # Pass DoclingDocument for HybridChunker
        )

        if not chunks:
            logger.warning(f"No chunks created for {document_title}")
            return IngestionResult(
                document_id="",
                title=document_title,
                chunks_created=0,
                processing_time_ms=(
                    datetime.now() - start_time
                ).total_seconds() * 1000,
                errors=["No chunks created"]
            )

        logger.info(f"Created {len(chunks)} chunks")

        # Generate embeddings
        embedded_chunks = await self.embedder.embed_chunks(chunks)
        logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")

        # Save to MongoDB
        document_id = await self._save_to_mongodb(
            document_title,
            document_source,
            document_content,
            embedded_chunks,
            document_metadata
        )

        logger.info(f"Saved document to MongoDB with ID: {document_id}")

        # Calculate processing time
        processing_time = (
            datetime.now() - start_time
        ).total_seconds() * 1000

        return IngestionResult(
            document_id=document_id,
            title=document_title,
            chunks_created=len(chunks),
            processing_time_ms=processing_time,
            errors=[]
        )

    async def ingest_documents(
        self,
        progress_callback: Optional[callable] = None
    ) -> List[IngestionResult]:
        """
        Ingest all documents from the documents folder.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            List of ingestion results
        """
        if not self._initialized:
            await self.initialize()

        # Clean existing data if requested
        if self.clean_before_ingest:
            await self._clean_databases()

        # Find all supported document files
        document_files = self._find_document_files()

        if not document_files:
            logger.warning(
                f"No supported document files found in {self.documents_folder}"
            )
            return []

        logger.info(f"Found {len(document_files)} document files to process")

        results = []

        for i, file_path in enumerate(document_files):
            try:
                logger.info(
                    f"Processing file {i+1}/{len(document_files)}: {file_path}"
                )

                result = await self._ingest_single_document(file_path)
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, len(document_files))

            except Exception as e:
                logger.exception(f"Failed to process {file_path}: {e}")
                results.append(IngestionResult(
                    document_id="",
                    title=os.path.basename(file_path),
                    chunks_created=0,
                    processing_time_ms=0,
                    errors=[str(e)]
                ))

        # Log summary
        total_chunks = sum(r.chunks_created for r in results)
        total_errors = sum(len(r.errors) for r in results)

        logger.info(
            f"Ingestion complete: {len(results)} documents, "
            f"{total_chunks} chunks, {total_errors} errors"
        )

        return results


async def main() -> None:
    """Main function for running ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into MongoDB vector database"
    )
    parser.add_argument(
        "--documents", "-d",
        default="documents",
        help="Documents folder path"
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip cleaning existing data before ingestion"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for splitting documents"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap size"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per chunk for embeddings"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create ingestion configuration
    config = IngestionConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_chunk_size=args.chunk_size * 2,
        max_tokens=args.max_tokens
    )

    # Create and run pipeline - clean by default unless --no-clean is specified
    pipeline = DocumentIngestionPipeline(
        config=config,
        documents_folder=args.documents,
        clean_before_ingest=not args.no_clean  # Clean by default
    )

    def progress_callback(current: int, total: int) -> None:
        print(f"Progress: {current}/{total} documents processed")

    try:
        start_time = datetime.now()

        results = await pipeline.ingest_documents(progress_callback)

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        # Print summary
        print("\n" + "="*50)
        print("INGESTION SUMMARY")
        print("="*50)
        print(f"Documents processed: {len(results)}")
        print(f"Total chunks created: {sum(r.chunks_created for r in results)}")
        print(f"Total errors: {sum(len(r.errors) for r in results)}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print()

        # Print individual results
        for result in results:
            status = "[OK]" if not result.errors else "[FAILED]"
            print(f"{status} {result.title}: {result.chunks_created} chunks")

            if result.errors:
                for error in result.errors:
                    print(f"  Error: {error}")

        # Print next steps
        print("\n" + "="*50)
        print("NEXT STEPS")
        print("="*50)
        print("1. Create vector search index in Atlas UI:")
        print("   - Index name: vector_index")
        print("   - Collection: chunks")
        print("   - Field: embedding")
        print("   - Dimensions: 1536 (for text-embedding-3-small)")
        print()
        print("2. Create text search index in Atlas UI:")
        print("   - Index name: text_index")
        print("   - Collection: chunks")
        print("   - Field: content")
        print()
        print("See .claude/reference/mongodb-patterns.md for detailed instructions")

    except KeyboardInterrupt:
        print("\nIngestion interrupted by user")
    except Exception as e:
        logger.exception(f"Ingestion failed: {e}")
        raise
    finally:
        await pipeline.close()




asyncio.run(main())

