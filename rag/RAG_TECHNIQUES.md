# RAG Techniques Implementation Guide

This guide documents how to implement various RAG (Retrieval-Augmented Generation) techniques in this codebase. Each section covers a technique, the classes to modify, and implementation examples.

---

## Table of Contents

1. [Current Architecture](#1-current-architecture)
2. [Chunking Strategies](#2-chunking-strategies)
3. [Reranking](#3-reranking)
4. [Query Expansion & Transformation](#4-query-expansion--transformation)
5. [Contextual Retrieval](#5-contextual-retrieval)
6. [Parent-Child Document Retrieval](#6-parent-child-document-retrieval)
7. [Metadata Filtering](#7-metadata-filtering)
8. [Multi-Vector Retrieval](#8-multi-vector-retrieval)
9. [Implementation Roadmap](#9-implementation-roadmap)

---

## 1. Current Architecture

### System Flow
```
Documents → Ingestion Pipeline → Chunking → Embedding → MongoDB → Retrieval → Agent
```

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| Ingestion | `rag/ingestion/pipeline.py` | Multi-format document processing, incremental indexing |
| Chunking | `rag/ingestion/chunkers/docling.py` | Docling HybridChunker (token-aware, structure-preserving) |
| Embedding | `rag/ingestion/embedder.py` | OpenAI-compatible API (Ollama, OpenAI) |
| Storage | `rag/storage/vector_store/mongo.py` | MongoDB Atlas with vector + text search |
| Retrieval | `rag/retrieval/retriever.py` | Semantic, text, and hybrid (RRF) search |
| Agent | `rag/agent/rag_agent.py` | Pydantic AI agent with search tool |
| Config | `rag/config/settings.py` | Environment-based configuration |

### Current Search Methods

| Method | Score Range | Best For |
|--------|-------------|----------|
| Semantic | 0.0 - 1.0 | Conceptual queries, paraphrases |
| Text | 0.0 - 10.0+ | Exact matches, keywords, acronyms |
| Hybrid (RRF) | 0.01 - 0.03 | Balanced retrieval (default) |

---

## 2. Chunking Strategies

### Current Implementation
- **Method**: Docling HybridChunker
- **Location**: `rag/ingestion/chunkers/docling.py`
- **Config**: `chunk_size=1000`, `chunk_overlap=200`, `max_tokens=512`

### Available Strategies

#### 2.1 Fixed-Size Chunking
Already implemented as fallback in `_simple_fallback_chunk()`.

```python
# rag/ingestion/chunkers/docling.py
def _simple_fallback_chunk(self, text: str) -> list[ChunkData]:
    # Sliding window with sentence boundary detection
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + self.config.chunk_size, len(text))
        # Find sentence boundary...
        chunks.append(ChunkData(content=text[start:end], ...))
        start = end - self.config.chunk_overlap
    return chunks
```

#### 2.2 Semantic Chunking
**Goal**: Split at semantic boundaries using embeddings.

**Files to modify**:
- Create `rag/ingestion/chunkers/semantic.py`

```python
# rag/ingestion/chunkers/semantic.py
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticChunker:
    def __init__(self, threshold: float = 0.5):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = threshold

    async def chunk(self, content: str) -> list[ChunkData]:
        # Split into sentences
        sentences = self._split_sentences(content)

        # Embed each sentence
        embeddings = self.model.encode(sentences)

        # Find semantic breaks (low similarity between adjacent sentences)
        chunks = []
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            similarity = np.dot(embeddings[i-1], embeddings[i])
            if similarity < self.threshold:
                # Semantic break - start new chunk
                chunks.append(ChunkData(content=" ".join(current_chunk), ...))
                current_chunk = []
            current_chunk.append(sentences[i])

        if current_chunk:
            chunks.append(ChunkData(content=" ".join(current_chunk), ...))

        return chunks
```

#### 2.3 Hierarchical Chunking
**Goal**: Create parent-child chunk relationships.

**Files to modify**:
- Create `rag/ingestion/chunkers/hierarchical.py`
- Update MongoDB schema in `rag/storage/vector_store/mongo.py`

```python
# rag/ingestion/chunkers/hierarchical.py
class HierarchicalChunker:
    def __init__(self, levels: list[int] = [2000, 500]):
        self.levels = levels  # [parent_size, child_size]

    async def chunk(self, content: str) -> list[ChunkData]:
        chunks = []

        # Level 0: Large parent chunks
        parent_chunks = self._chunk_at_size(content, self.levels[0])

        for parent_idx, parent in enumerate(parent_chunks):
            parent.metadata["hierarchy_level"] = 0
            parent.metadata["parent_chunk_id"] = None
            chunks.append(parent)

            # Level 1: Smaller child chunks
            children = self._chunk_at_size(parent.content, self.levels[1])
            for child in children:
                child.metadata["hierarchy_level"] = 1
                child.metadata["parent_chunk_id"] = parent_idx
                chunks.append(child)

        return chunks
```

**Schema extension** in MongoDB:
```python
# Add to chunk document
{
    "parent_chunk_id": ObjectId | None,
    "children_chunk_ids": [ObjectId],
    "hierarchy_level": 0  # 0=parent, 1=child
}
```

### Switching Chunking Strategy

**Modify**: `rag/ingestion/pipeline.py`

```python
def _get_chunker(self, strategy: str):
    match strategy:
        case "hybrid":
            return DoclingHybridChunker(self.chunking_config)
        case "semantic":
            return SemanticChunker()
        case "hierarchical":
            return HierarchicalChunker()
        case "fixed":
            return FixedSizeChunker(self.chunking_config)
        case _:
            return DoclingHybridChunker(self.chunking_config)
```

---

## 3. Reranking

### Current State
Only RRF (Reciprocal Rank Fusion) scoring, no dedicated reranker.

### Adding Reranking

**Files to create/modify**:
- Create `rag/retrieval/rerankers.py`
- Modify `rag/retrieval/retriever.py`
- Update `rag/config/settings.py`

#### 3.1 Cross-Encoder Reranker

```python
# rag/retrieval/rerankers.py
from sentence_transformers import CrossEncoder
from rag.ingestion.models import SearchResult

class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        self.model = CrossEncoder(model_name)

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int
    ) -> list[SearchResult]:
        if not results:
            return results

        # Create query-document pairs
        pairs = [(query, r.content) for r in results]

        # Score with cross-encoder
        scores = self.model.predict(pairs)

        # Sort by score and update results
        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        for result, score in scored_results:
            result.similarity = float(score)

        return [r for r, _ in scored_results[:top_k]]
```

#### 3.2 LLM Reranker

```python
# rag/retrieval/rerankers.py
class LLMReranker:
    def __init__(self, llm_client):
        self.llm = llm_client

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int
    ) -> list[SearchResult]:
        prompt = f"""Rate the relevance of each document to the query on a scale of 0-10.

Query: {query}

Documents:
{self._format_documents(results)}

Return only the document numbers in order of relevance (most relevant first):"""

        response = await self.llm.generate(prompt)
        ranking = self._parse_ranking(response)

        return [results[i] for i in ranking[:top_k]]
```

#### 3.3 Integration in Retriever

```python
# rag/retrieval/retriever.py
class Retriever:
    def __init__(
        self,
        store: MongoHybridStore | None = None,
        embedder: EmbeddingGenerator | None = None,
        reranker: Reranker | None = None,  # Add this
    ):
        self.store = store or MongoHybridStore()
        self.embedder = embedder or EmbeddingGenerator()
        self.reranker = reranker

    async def retrieve(
        self,
        query: str,
        match_count: int | None = None,
        search_type: str = "hybrid",
        rerank: bool = False,  # Add this
    ) -> list[SearchResult]:
        # ... existing search logic ...

        # Add reranking step
        if rerank and self.reranker:
            # Over-fetch for reranking
            results = await self.store.hybrid_search(
                query, query_embedding, match_count * 3
            )
            results = await self.reranker.rerank(query, results, match_count)
        else:
            results = await self.store.hybrid_search(
                query, query_embedding, match_count
            )

        return results
```

---

## 4. Query Expansion & Transformation

### Current State
Direct query search, no processing.

### Adding Query Processors

**Files to create/modify**:
- Create `rag/retrieval/query_processors.py`
- Modify `rag/retrieval/retriever.py`

#### 4.1 LLM Query Expansion

```python
# rag/retrieval/query_processors.py
class LLMQueryExpander:
    def __init__(self, llm_client):
        self.llm = llm_client

    async def expand(self, query: str, num_expansions: int = 3) -> list[str]:
        prompt = f"""Generate {num_expansions} alternative phrasings of this query:
"{query}"

Return only the alternative queries, one per line."""

        response = await self.llm.generate(prompt)
        expansions = response.strip().split("\n")
        return [query] + expansions[:num_expansions]
```

#### 4.2 HyDE (Hypothetical Document Embeddings)

```python
# rag/retrieval/query_processors.py
class HyDEProcessor:
    def __init__(self, llm_client, embedder):
        self.llm = llm_client
        self.embedder = embedder

    async def generate_hypothetical(self, query: str) -> str:
        prompt = f"""Write a short passage that would answer this question:
"{query}"

Write as if you are quoting from a document that contains the answer."""

        return await self.llm.generate(prompt)

    async def get_hyde_embedding(self, query: str) -> list[float]:
        hypothetical = await self.generate_hypothetical(query)
        return await self.embedder.embed_query(hypothetical)
```

#### 4.3 Multi-Query Retrieval

```python
# rag/retrieval/retriever.py
async def retrieve_multi_query(
    self,
    query: str,
    match_count: int | None = None
) -> list[SearchResult]:
    """Retrieve using multiple query variations."""
    # Expand query
    expanded_queries = await self.query_processor.expand(query)

    all_results = []
    seen_chunk_ids = set()

    for q in expanded_queries:
        embedding = await self.embedder.embed_query(q)
        results = await self.store.semantic_search(embedding, match_count)

        for r in results:
            if r.chunk_id not in seen_chunk_ids:
                all_results.append(r)
                seen_chunk_ids.add(r.chunk_id)

    # Re-rank combined results
    all_results.sort(key=lambda x: x.similarity, reverse=True)
    return all_results[:match_count]
```

---

## 5. Contextual Retrieval

### Current State
Returns only matched chunks, no surrounding context.

### Adding Context Expansion

**Files to create/modify**:
- Create `rag/retrieval/context_expanders.py`
- Modify `rag/storage/vector_store/mongo.py`

#### 5.1 Adjacent Chunk Expander

```python
# rag/retrieval/context_expanders.py
class AdjacentChunkExpander:
    def __init__(self, store: MongoHybridStore):
        self.store = store

    async def expand(
        self,
        result: SearchResult,
        context_before: int = 1,
        context_after: int = 1
    ) -> dict:
        """Get surrounding chunks for context."""
        # Get the matched chunk's index
        chunk = await self.store.get_chunk_by_id(result.chunk_id)
        chunk_index = chunk["chunk_index"]
        document_id = chunk["document_id"]

        # Fetch adjacent chunks
        chunks = await self.store.get_chunks_by_document(
            document_id,
            start_index=max(0, chunk_index - context_before),
            end_index=chunk_index + context_after + 1
        )

        return {
            "main": result,
            "context_before": [c for c in chunks if c["chunk_index"] < chunk_index],
            "context_after": [c for c in chunks if c["chunk_index"] > chunk_index],
            "combined_content": self._combine_chunks(chunks)
        }

    def _combine_chunks(self, chunks: list[dict]) -> str:
        sorted_chunks = sorted(chunks, key=lambda x: x["chunk_index"])
        return "\n\n".join(c["content"] for c in sorted_chunks)
```

#### 5.2 MongoDB Helper Methods

```python
# rag/storage/vector_store/mongo.py
async def get_chunk_by_id(self, chunk_id: str) -> dict:
    """Get a single chunk by ID."""
    collection = self.db[self.settings.mongodb_collection_chunks]
    return await collection.find_one({"_id": ObjectId(chunk_id)})

async def get_chunks_by_document(
    self,
    document_id: str,
    start_index: int,
    end_index: int
) -> list[dict]:
    """Get chunks for a document within index range."""
    collection = self.db[self.settings.mongodb_collection_chunks]
    cursor = collection.find({
        "document_id": ObjectId(document_id),
        "chunk_index": {"$gte": start_index, "$lt": end_index}
    }).sort("chunk_index", 1)

    return await cursor.to_list(length=end_index - start_index)
```

---

## 6. Parent-Child Document Retrieval

### Current State
Flat chunk structure, no hierarchical relationships.

### Adding Hierarchical Retrieval

**Files to modify**:
- `rag/storage/vector_store/mongo.py`
- `rag/ingestion/pipeline.py`
- `rag/retrieval/retriever.py`

#### 6.1 Schema Extension

```python
# Extended chunk schema
{
    "_id": ObjectId,
    "document_id": ObjectId,
    "content": str,
    "embedding": list[float],
    "chunk_index": int,

    # New hierarchical fields
    "parent_chunk_id": ObjectId | None,
    "children_chunk_ids": list[ObjectId],
    "hierarchy_level": int,  # 0=leaf, 1=parent, 2=grandparent
    "section_path": str,  # e.g., "1.2.3" for nested sections
}
```

#### 6.2 Parent Retrieval

```python
# rag/storage/vector_store/mongo.py
async def semantic_search_with_parents(
    self,
    query_embedding: list[float],
    match_count: int
) -> list[SearchResult]:
    """Search and include parent chunks for context."""
    # Get base results
    results = await self.semantic_search(query_embedding, match_count)

    # Fetch parent chunks
    enriched_results = []
    for result in results:
        enriched = {"result": result, "parent": None}

        chunk = await self.get_chunk_by_id(result.chunk_id)
        if chunk.get("parent_chunk_id"):
            parent = await self.get_chunk_by_id(str(chunk["parent_chunk_id"]))
            enriched["parent"] = parent

        enriched_results.append(enriched)

    return enriched_results
```

#### 6.3 Two-Stage Retrieval

```python
# rag/retrieval/retriever.py
async def retrieve_hierarchical(
    self,
    query: str,
    match_count: int = 5
) -> list[dict]:
    """Two-stage retrieval: find children, return with parents."""
    embedding = await self.embedder.embed_query(query)

    # Stage 1: Search leaf chunks (fine-grained)
    leaf_results = await self.store.semantic_search(
        embedding,
        match_count * 2,
        filter={"hierarchy_level": 1}  # Only search children
    )

    # Stage 2: Get parent context
    enriched = []
    for result in leaf_results[:match_count]:
        parent = await self.store.get_parent_chunk(result.chunk_id)
        enriched.append({
            "matched_chunk": result,
            "parent_chunk": parent,
            "context": parent["content"] if parent else result.content
        })

    return enriched
```

---

## 7. Metadata Filtering

### Current State
No filtering, returns all matching chunks.

### Adding Metadata Filters

**Files to modify**:
- `rag/storage/vector_store/mongo.py`
- `rag/agent/rag_agent.py`

#### 7.1 Filter Implementation

```python
# rag/storage/vector_store/mongo.py
async def semantic_search(
    self,
    query_embedding: list[float],
    match_count: int | None = None,
    filters: dict | None = None  # Add this
) -> list[SearchResult]:
    # Build filter stage
    filter_stage = self._build_filter_stage(filters) if filters else None

    pipeline = [
        {
            "$vectorSearch": {
                "index": self.settings.mongodb_vector_index,
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 100,
                "limit": match_count,
                "filter": filter_stage  # Add filter here
            }
        },
        # ... rest of pipeline
    ]

def _build_filter_stage(self, filters: dict) -> dict:
    """Convert filter dict to MongoDB filter expression."""
    mongo_filter = {}

    if "source_pattern" in filters:
        mongo_filter["metadata.source"] = {"$regex": filters["source_pattern"]}

    if "created_after" in filters:
        mongo_filter["created_at"] = {"$gte": filters["created_after"]}

    if "document_type" in filters:
        mongo_filter["metadata.file_type"] = filters["document_type"]

    if "title_contains" in filters:
        mongo_filter["metadata.title"] = {"$regex": filters["title_contains"], "$options": "i"}

    return mongo_filter
```

#### 7.2 Agent Tool Update

```python
# rag/agent/rag_agent.py
@rag_agent.tool
async def search_knowledge_base(
    ctx: PydanticRunContext,
    query: str,
    match_count: int | None = 5,
    search_type: str | None = "hybrid",
    source_filter: str | None = None,  # Add filters
    doc_type_filter: str | None = None,
) -> str:
    filters = {}
    if source_filter:
        filters["source_pattern"] = source_filter
    if doc_type_filter:
        filters["document_type"] = doc_type_filter

    result = await retriever.retrieve_as_context(
        query=query,
        match_count=match_count,
        search_type=search_type,
        filters=filters if filters else None
    )
    return result
```

---

## 8. Multi-Vector Retrieval

### Current State
Single embedding per chunk.

### Adding Multi-Vector Support

**Files to modify**:
- `rag/ingestion/embedder.py`
- `rag/storage/vector_store/mongo.py`
- `rag/ingestion/pipeline.py`

#### 8.1 Multi-Embedding Generation

```python
# rag/ingestion/embedder.py
class MultiEmbeddingGenerator:
    def __init__(self):
        self.embedders = {
            "primary": EmbeddingGenerator(model="nomic-embed-text"),
            "summary": EmbeddingGenerator(model="text-embedding-3-small"),
        }
        self.llm = None  # For summary generation

    async def embed_chunk_multi(self, chunk: ChunkData) -> dict[str, list[float]]:
        """Generate multiple embeddings for a chunk."""
        embeddings = {}

        # Primary embedding
        embeddings["primary"] = await self.embedders["primary"].embed_text(chunk.content)

        # Summary embedding (embed a summary of the chunk)
        if self.llm:
            summary = await self._generate_summary(chunk.content)
            embeddings["summary"] = await self.embedders["summary"].embed_text(summary)

        return embeddings

    async def _generate_summary(self, content: str) -> str:
        prompt = f"Summarize in 1-2 sentences:\n{content}"
        return await self.llm.generate(prompt)
```

#### 8.2 Extended Storage Schema

```python
# MongoDB chunk schema with multiple embeddings
{
    "_id": ObjectId,
    "document_id": ObjectId,
    "content": str,

    # Multiple embeddings
    "embeddings": {
        "primary": list[float],    # 768-dim nomic
        "summary": list[float],    # 1536-dim OpenAI
        "hyde": list[float],       # Hypothetical doc embedding
    },

    # Keep backward compatibility
    "embedding": list[float],  # Alias for primary
}
```

#### 8.3 Multi-Vector Search

```python
# rag/storage/vector_store/mongo.py
async def multi_vector_search(
    self,
    query_embedding: list[float],
    embedding_type: str = "primary",
    match_count: int = 10
) -> list[SearchResult]:
    """Search using a specific embedding type."""
    pipeline = [
        {
            "$vectorSearch": {
                "index": f"vector_index_{embedding_type}",  # Separate index per type
                "queryVector": query_embedding,
                "path": f"embeddings.{embedding_type}",
                "numCandidates": 100,
                "limit": match_count,
            }
        },
        # ... rest of pipeline
    ]
```

---

## 9. Implementation Roadmap

### Phase 1: Reranking (High Impact, Medium Effort)
1. Create `rag/retrieval/rerankers.py` with `CrossEncoderReranker`
2. Add `reranker` parameter to `Retriever.__init__()`
3. Update `retrieve()` to optionally rerank
4. Add settings for reranker model
5. Update agent tool to expose reranking

**Expected improvement**: 5-15% relevance

### Phase 2: Query Processing (Medium Impact, Low Effort)
1. Create `rag/retrieval/query_processors.py`
2. Implement `LLMQueryExpander`
3. Add `retrieve_multi_query()` method
4. Create HyDE processor

**Expected improvement**: 3-10% for ambiguous queries

### Phase 3: Context Expansion (Medium Impact, Low Effort)
1. Create `rag/retrieval/context_expanders.py`
2. Add `get_chunks_by_document()` to MongoDB store
3. Update `retrieve_as_context()` to include surrounding chunks
4. Add context size to settings

**Expected improvement**: Better answer quality

### Phase 4: Metadata Filtering (Low Impact, Low Effort)
1. Add `filters` parameter to search methods
2. Build filter pipeline in MongoDB
3. Update agent tool parameters

**Expected improvement**: Domain-specific retrieval

### Phase 5: Hierarchical Chunking (High Impact, High Effort)
1. Create `HierarchicalChunker`
2. Extend MongoDB schema
3. Add two-stage retrieval
4. Update ingestion pipeline
5. Create data migration script

**Expected improvement**: Better context preservation

---

## Quick Reference: Files to Modify by Technique

| Technique | Primary Files | Supporting Files |
|-----------|---------------|------------------|
| Chunking | `chunkers/docling.py` | `pipeline.py`, `models.py` |
| Reranking | `retrieval/rerankers.py` (new) | `retriever.py`, `settings.py` |
| Query Expansion | `retrieval/query_processors.py` (new) | `retriever.py` |
| Context Expansion | `retrieval/context_expanders.py` (new) | `mongo.py`, `retriever.py` |
| Parent-Child | `mongo.py`, `pipeline.py` | `chunkers/`, `retriever.py` |
| Metadata Filtering | `mongo.py` | `rag_agent.py` |
| Multi-Vector | `embedder.py`, `mongo.py` | `pipeline.py` |

---

## Configuration Template

Add these to `rag/config/settings.py` for new features:

```python
class Settings(BaseSettings):
    # Existing settings...

    # Chunking
    chunking_strategy: str = "hybrid"  # hybrid, semantic, hierarchical, fixed

    # Reranking
    reranker_enabled: bool = False
    reranker_type: str = "cross_encoder"  # cross_encoder, colbert, llm
    reranker_model: str = "BAAI/bge-reranker-large"
    reranker_top_k_multiplier: int = 3  # Over-fetch factor

    # Query Processing
    query_expansion_enabled: bool = False
    query_expansion_count: int = 3
    hyde_enabled: bool = False

    # Context Expansion
    context_expansion_enabled: bool = False
    context_chunks_before: int = 1
    context_chunks_after: int = 1

    # Hierarchical
    hierarchical_retrieval_enabled: bool = False
    hierarchy_levels: list[int] = [2000, 500]
```
