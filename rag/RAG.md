# RAG Techniques Implementation Guide

This guide documents how to implement various RAG (Retrieval-Augmented Generation) techniques in this codebase. Each section covers a technique, the classes to modify, and implementation examples.

---

## Table of Contents

1. [Current Architecture](#1-current-architecture)
2. [Streamlit Web UI](#2-streamlit-wei)
3. [Chunking Strategies](#3-chunking-strategies)
4. [Reranking](#4-reranking)
5. [Query Expansion & Transformation](#5-query-expansion--transformation)
6. [Contextual Retrieval](#6-contextual-retrieval)
7. [Parent-Child Document Retrieval](#7-parent-child-document-retrieval)
8. [Metadata Filtering](#8-metadata-filtering)
9. [Multi-Vector Retrieval](#9-multi-vector-retrieval)
10. [Knowledge Graph RAG with Graphiti](#10-knowledge-graph-rag-with-graphiti)
11. [Langfuse Tracing & Observability](#11-langfuse-tracing--observability)
12. [Implementation Roadmap](#12-implementation-roadmap)
13. [Testing](#13-testing)

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

## 2. Streamlit Web UI

The RAG agent includes a Streamlit-based web interface for interactive chat with the knowledge base.

### Features

- **Real-time Streaming**: See responses as they're generated token-by-token
- **Tool Call Visibility**: Watch the agent search the knowledge base in real-time
- **Conversation History**: Multi-turn conversations with full context
- **Configuration Display**: View current LLM and embedding settings
- **Session Management**: Clear conversation and start fresh

### Screenshot

```
┌─────────────────────────────────────────────────────────────────────┐
│  🔍 MongoDB RAG Agent          │  💬 Chat with RAG Agent            │
│  ─────────────────────         │  ───────────────────────           │
│  Configuration                 │                                     │
│  LLM Provider: ollama          │  User: What is the PTO policy?     │
│  LLM Model: llama3.1:8b        │                                     │
│  Embedding: nomic-embed-text   │  🔧 Calling: search_knowledge_base │
│                                │     Query: PTO policy               │
│  [🗑️ Clear Conversation]       │     Type: hybrid                    │
│                                │  ✅ Search completed                │
│  ℹ️ Help                        │                                     │
│  ────────                      │  Assistant: The PTO policy allows  │
│  How to use:                   │  employees to take...               │
│  1. Type your question...      │                                     │
│                                │  ────────────────────────────────   │
│                                │  [Ask a question...]                │
└─────────────────────────────────────────────────────────────────────┘
```

### Running the App

#### Prerequisites

1. Ensure MongoDB Atlas is configured with vector and text indexes
2. Ensure Ollama is running (or configure another LLM provider)
3. Install Streamlit:

```bash
pip install streamlit>=1.40.0
```

#### Start the App

From the project root directory:

```bash
streamlit run rag/agent/streamlit_app.py
```

Or with specific options:

```bash
# Run on a different port
streamlit run rag/agent/streamlit_app.py --server.port 8502

# Run in headless mode (no browser auto-open)
streamlit run rag/agent/streamlit_app.py --server.headless true

# Run with specific address binding
streamlit run rag/agent/streamlit_app.py --server.address 0.0.0.0
```

The app will be available at: **http://localhost:8501**

### File Structure

| File | Purpose |
|------|---------|
| `rag/agent/streamlit_app.py` | Main Streamlit application |
| `rag/agent/rag_agent.py` | RAG agent with search tool |
| `rag/agent/agent_main.py` | CLI version (alternative interface) |

### Key Functions

| Function | Description |
|----------|-------------|
| `init_session_state()` | Initialize chat history and agent state |
| `stream_agent_response()` | Stream agent response with real-time updates |
| `render_sidebar()` | Display configuration and controls |
| `render_chat()` | Main chat interface with message history |
| `extract_tool_info()` | Parse tool call events for display |

### Example Queries

Once the app is running, try these queries:

```
What does NeuralFlow AI do?
What is the PTO policy?
What technologies does the company use?
How many engineers work at the company?
What is the learning budget for employees?
```

### Customization

#### Changing the Page Title

Edit `streamlit_app.py`:

```python
st.set_page_config(
    page_title="Your Custom Title",
    page_icon="🤖",  # Change emoji
    layout="wide",
)
```

#### Adding Custom Sidebar Content

```python
def render_sidebar():
    with st.sidebar:
        st.title("Your App Name")
        # Add custom widgets
        st.slider("Temperature", 0.0, 1.0, 0.7)
```

#### Styling with Custom CSS

```python
st.markdown("""
<style>
    .stChat { background-color: #f0f2f6; }
    .stChatMessage { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'rag'` | Run from project root, not `rag/agent/` |
| App won't start | Check if port 8501 is in use: `lsof -i :8501` |
| No response from agent | Verify Ollama is running: `ollama list` |
| MongoDB connection error | Check `MONGODB_URI` in `.env` file |
| Slow responses | Consider using a faster model or reducing `match_count` |

### Running with Docker (Optional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -e .
RUN pip install streamlit>=1.40.0

EXPOSE 8501

CMD ["streamlit", "run", "rag/agent/streamlit_app.py", "--server.headless", "true"]
```

```bash
docker build -t rag-streamlit .
docker run -p 8501:8501 --env-file .env rag-streamlit
```

---

## 3. Chunking Strategies

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

## 4. Reranking

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

## 5. Query Expansion & Transformation

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

## 6. Contextual Retrieval

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

## 7. Parent-Child Document Retrieval

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

## 8. Metadata Filtering

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

## 9. Multi-Vector Retrieval

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

## 10. Knowledge Graph RAG with Graphiti

### Overview

[Graphiti](https://github.com/getzep/graphiti) is a Python framework by Zep AI for building temporally-aware knowledge graphs designed for AI agents. Unlike traditional RAG which retrieves chunks of text, Knowledge Graph RAG (GraphRAG) retrieves structured facts and relationships between entities.

### Why Knowledge Graphs for RAG?

| Aspect | Traditional RAG | Knowledge Graph RAG |
|--------|-----------------|---------------------|
| **Data Structure** | Flat text chunks | Entities + Relationships (Triplets) |
| **Query Type** | Semantic similarity | Graph traversal + Semantic |
| **Temporal Handling** | Basic timestamps | Bi-temporal (event time + ingestion time) |
| **Contradiction Handling** | None | Edge invalidation with history |
| **Context** | Sliding window | Multi-hop relationships |
| **Updates** | Re-embed chunks | Incremental graph updates |

### Graphiti Key Features

- **Bi-temporal data model**: Tracks both when events occurred and when they were ingested
- **Hybrid retrieval**: Combines semantic embeddings, BM25 keyword search, and graph traversal
- **Custom entity types**: Define entities via Pydantic models
- **Multiple graph backends**: Neo4j, FalkorDB, Kuzu, Amazon Neptune
- **Real-time updates**: Incremental updates without batch recomputation

### Installation

```bash
# Basic installation (Neo4j backend)
pip install graphiti-core

# With FalkorDB backend
pip install graphiti-core[falkordb]

# With Ollama support (local LLM)
pip install graphiti-core
```

### Integration Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              Hybrid RAG System              │
                    └─────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
            ┌───────▼───────┐                     ┌─────────▼─────────┐
            │  Vector RAG   │                     │  Knowledge Graph  │
            │  (MongoDB)    │                     │  RAG (Graphiti)   │
            └───────┬───────┘                     └─────────┬─────────┘
                    │                                       │
            ┌───────▼───────┐                     ┌─────────▼─────────┐
            │ Chunk-based   │                     │ Entity-based      │
            │ Retrieval     │                     │ Retrieval         │
            │ - Semantic    │                     │ - Facts/Triplets  │
            │ - Keyword     │                     │ - Relationships   │
            │ - Hybrid RRF  │                     │ - Graph Traversal │
            └───────┬───────┘                     └─────────┬─────────┘
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        │
                              ┌─────────▼─────────┐
                              │   Merge Results   │
                              │   (RRF Fusion)    │
                              └─────────┬─────────┘
                                        │
                              ┌─────────▼─────────┐
                              │   LLM Response    │
                              └───────────────────┘
```

### Files to Create/Modify

| File | Purpose |
|------|---------|
| `rag/knowledge_graph/graphiti_store.py` (new) | Graphiti wrapper for graph operations |
| `rag/knowledge_graph/entity_types.py` (new) | Custom Pydantic entity definitions |
| `rag/retrieval/retriever.py` | Add graph retrieval method |
| `rag/ingestion/pipeline.py` | Add graph ingestion alongside vector ingestion |
| `rag/agent/rag_agent.py` | Add graph search tool |
| `rag/config/settings.py` | Add Graphiti configuration |

### Implementation

#### 9.1 Configuration

```python
# rag/config/settings.py
class Settings(BaseSettings):
    # Existing settings...

    # Graphiti / Knowledge Graph
    graphiti_enabled: bool = False
    graph_db_type: str = "neo4j"  # neo4j, falkordb, kuzu
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # For FalkorDB
    falkordb_host: str = "localhost"
    falkordb_port: int = 6379
```

#### 9.2 Graphiti Store Wrapper

```python
# rag/knowledge_graph/graphiti_store.py
import logging
from datetime import datetime, timezone
from typing import Any

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import (
    NODE_HYBRID_SEARCH_RRF,
    EDGE_HYBRID_SEARCH_RRF,
)

from rag.config.settings import load_settings

logger = logging.getLogger(__name__)


class GraphitiStore:
    """Wrapper for Graphiti knowledge graph operations."""

    def __init__(self):
        self.settings = load_settings()
        self.graphiti: Graphiti | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Graphiti connection."""
        if self._initialized:
            return

        if self.settings.graph_db_type == "neo4j":
            self.graphiti = Graphiti(
                self.settings.neo4j_uri,
                self.settings.neo4j_user,
                self.settings.neo4j_password,
            )
        elif self.settings.graph_db_type == "falkordb":
            from graphiti_core.driver.falkordb_driver import FalkorDriver

            driver = FalkorDriver(
                host=self.settings.falkordb_host,
                port=self.settings.falkordb_port,
            )
            self.graphiti = Graphiti(graph_driver=driver)

        self._initialized = True
        logger.info(f"Graphiti initialized with {self.settings.graph_db_type}")

    async def close(self) -> None:
        """Close Graphiti connection."""
        if self.graphiti:
            await self.graphiti.close()
            self.graphiti = None
            self._initialized = False

    async def add_episode(
        self,
        content: str,
        name: str,
        source_description: str = "document",
        episode_type: EpisodeType = EpisodeType.text,
        reference_time: datetime | None = None,
    ) -> None:
        """
        Add an episode (document/text) to the knowledge graph.

        Graphiti will automatically:
        - Extract entities (nodes)
        - Extract relationships (edges)
        - Handle temporal information
        - Deduplicate entities
        """
        await self.initialize()

        if reference_time is None:
            reference_time = datetime.now(timezone.utc)

        await self.graphiti.add_episode(
            name=name,
            episode_body=content,
            source=episode_type,
            source_description=source_description,
            reference_time=reference_time,
        )
        logger.info(f"Added episode to graph: {name}")

    async def search_edges(
        self,
        query: str,
        limit: int = 10,
        center_node_uuid: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for relationships (edges/facts) in the knowledge graph.

        Returns triplets like: "Kamala Harris" -[WAS_ATTORNEY_GENERAL_OF]-> "California"
        """
        await self.initialize()

        results = await self.graphiti.search(
            query=query,
            num_results=limit,
            center_node_uuid=center_node_uuid,
        )

        return [
            {
                "uuid": r.uuid,
                "fact": r.fact,
                "source_node": r.source_node_uuid,
                "target_node": r.target_node_uuid,
                "valid_at": r.valid_at if hasattr(r, 'valid_at') else None,
                "invalid_at": r.invalid_at if hasattr(r, 'invalid_at') else None,
            }
            for r in results
        ]

    async def search_nodes(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search for entities (nodes) in the knowledge graph.

        Returns entities like: "Kamala Harris", "California", "Attorney General"
        """
        await self.initialize()

        config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        config.limit = limit

        results = await self.graphiti._search(query=query, config=config)

        return [
            {
                "uuid": node.uuid,
                "name": node.name,
                "summary": node.summary,
                "labels": node.labels,
                "created_at": node.created_at,
            }
            for node in results.nodes
        ]

    async def search_as_context(
        self,
        query: str,
        limit: int = 10,
    ) -> str:
        """
        Search and format results as context for LLM.
        """
        edges = await self.search_edges(query, limit)

        if not edges:
            return "No relevant facts found in knowledge graph."

        context_parts = ["## Knowledge Graph Facts\n"]
        for i, edge in enumerate(edges, 1):
            fact = edge["fact"]
            validity = ""
            if edge.get("valid_at"):
                validity = f" (from {edge['valid_at']}"
                if edge.get("invalid_at"):
                    validity += f" to {edge['invalid_at']}"
                validity += ")"
            context_parts.append(f"{i}. {fact}{validity}")

        return "\n".join(context_parts)
```

#### 9.3 Custom Entity Types (Optional)

```python
# rag/knowledge_graph/entity_types.py
from pydantic import BaseModel, Field


class Person(BaseModel):
    """Custom entity type for people."""
    name: str = Field(..., description="Full name of the person")
    role: str | None = Field(None, description="Job title or role")
    organization: str | None = Field(None, description="Associated organization")


class Organization(BaseModel):
    """Custom entity type for organizations."""
    name: str = Field(..., description="Organization name")
    type: str | None = Field(None, description="Type: company, government, nonprofit")
    location: str | None = Field(None, description="Headquarters location")


class Technology(BaseModel):
    """Custom entity type for technologies/tools."""
    name: str = Field(..., description="Technology name")
    category: str | None = Field(None, description="Category: language, framework, database")
    version: str | None = Field(None, description="Version if applicable")
```

#### 9.4 Integrated Retriever

```python
# rag/retrieval/retriever.py (additions)
from rag.knowledge_graph.graphiti_store import GraphitiStore


class Retriever:
    def __init__(
        self,
        store: MongoHybridStore | None = None,
        embedder: EmbeddingGenerator | None = None,
        graph_store: GraphitiStore | None = None,  # Add this
    ):
        self.store = store or MongoHybridStore()
        self.embedder = embedder or EmbeddingGenerator()
        self.graph_store = graph_store  # Optional graph store

    async def retrieve_hybrid_with_graph(
        self,
        query: str,
        match_count: int = 5,
        graph_weight: float = 0.3,
    ) -> dict[str, Any]:
        """
        Retrieve from both vector store and knowledge graph.

        Args:
            query: Search query
            match_count: Number of results per source
            graph_weight: Weight for graph results in final ranking (0-1)

        Returns:
            Combined results with both chunks and graph facts
        """
        # Vector search
        vector_results = await self.retrieve(query, match_count)

        # Graph search (if enabled)
        graph_results = []
        if self.graph_store:
            graph_results = await self.graph_store.search_edges(query, match_count)

        return {
            "chunks": vector_results,
            "facts": graph_results,
            "combined_context": self._merge_contexts(
                vector_results, graph_results, graph_weight
            ),
        }

    def _merge_contexts(
        self,
        chunks: list[SearchResult],
        facts: list[dict],
        graph_weight: float,
    ) -> str:
        """Merge vector chunks and graph facts into unified context."""
        context_parts = []

        # Add graph facts first (structured knowledge)
        if facts:
            context_parts.append("## Established Facts")
            for fact in facts:
                context_parts.append(f"- {fact['fact']}")
            context_parts.append("")

        # Add document chunks (detailed content)
        if chunks:
            context_parts.append("## Document Excerpts")
            for chunk in chunks:
                context_parts.append(f"### From: {chunk.document_title}")
                context_parts.append(chunk.content)
                context_parts.append("")

        return "\n".join(context_parts)
```

#### 9.5 Agent Tool Integration

```python
# rag/agent/rag_agent.py (additions)
@rag_agent.tool
async def search_knowledge_graph(
    ctx: RunContext[RAGState],
    query: str,
    limit: int = 10,
    search_type: str = "edges",  # "edges" for facts, "nodes" for entities
) -> str:
    """
    Search the knowledge graph for facts and relationships.

    Use this for:
    - Finding relationships between entities ("Who works at company X?")
    - Getting temporal facts ("When did X happen?")
    - Understanding entity connections ("How are X and Y related?")

    Args:
        query: Natural language query
        limit: Maximum results to return
        search_type: "edges" for facts/relationships, "nodes" for entities

    Returns:
        Formatted facts or entities from the knowledge graph
    """
    graph_store = GraphitiStore()

    try:
        if search_type == "edges":
            results = await graph_store.search_edges(query, limit)
            if not results:
                return "No facts found matching your query."

            formatted = ["Found the following facts:"]
            for r in results:
                formatted.append(f"- {r['fact']}")
            return "\n".join(formatted)

        else:  # nodes
            results = await graph_store.search_nodes(query, limit)
            if not results:
                return "No entities found matching your query."

            formatted = ["Found the following entities:"]
            for r in results:
                formatted.append(f"- {r['name']}: {r['summary'][:100]}...")
            return "\n".join(formatted)

    finally:
        await graph_store.close()
```

#### 9.6 Ingestion Pipeline Integration

```python
# rag/ingestion/pipeline.py (additions)
from rag.knowledge_graph.graphiti_store import GraphitiStore


class IngestionPipeline:
    def __init__(self, ...):
        # Existing init...
        self.graph_store: GraphitiStore | None = None
        if self.settings.graphiti_enabled:
            self.graph_store = GraphitiStore()

    async def ingest_document(self, file_path: Path) -> dict[str, Any]:
        """Ingest document into both vector store and knowledge graph."""

        # Existing vector ingestion...
        result = await self._ingest_to_vector_store(file_path)

        # Knowledge graph ingestion (if enabled)
        if self.graph_store and self.settings.graphiti_enabled:
            content = result.get("content", "")
            await self.graph_store.add_episode(
                content=content,
                name=file_path.stem,
                source_description=f"Document: {file_path.name}",
            )
            result["graph_ingested"] = True

        return result
```

### When to Use Knowledge Graph RAG

| Scenario | Use Vector RAG | Use Graph RAG | Use Both |
|----------|---------------|---------------|----------|
| Document Q&A | ✓ | | |
| Entity relationships | | ✓ | |
| Temporal queries | | ✓ | |
| Multi-hop reasoning | | ✓ | |
| Detailed explanations | ✓ | | |
| Fact verification | | ✓ | |
| Complex enterprise data | | | ✓ |
| Chatbot with memory | | ✓ | |

### Graph Database Options

| Database | Best For | Notes |
|----------|----------|-------|
| **Neo4j** | Production, enterprise | Most mature, requires Neo4j Desktop or cloud |
| **FalkorDB** | Quick start, Redis-based | Simple Docker setup, good for dev |
| **Kuzu** | Embedded, lightweight | No server needed, file-based |
| **Amazon Neptune** | AWS deployments | Managed service, enterprise scale |

### Quick Start with FalkorDB (Docker)

```bash
# Start FalkorDB
docker run -p 6379:6379 -p 3000:3000 -it --rm falkordb/falkordb:latest

# Install with FalkorDB support
pip install graphiti-core[falkordb]
```

```python
# Quick test
import asyncio
from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver

async def test_graphiti():
    driver = FalkorDriver(host="localhost", port=6379)
    graphiti = Graphiti(graph_driver=driver)

    # Add some data
    await graphiti.add_episode(
        name="test",
        episode_body="Alice is an engineer at TechCorp. Bob is her manager.",
        source=EpisodeType.text,
        source_description="test data",
    )

    # Search
    results = await graphiti.search("Who works at TechCorp?")
    for r in results:
        print(f"Fact: {r.fact}")

    await graphiti.close()

asyncio.run(test_graphiti())
```

### Using with Ollama (Local LLM)

```python
from graphiti_core import Graphiti
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

# Configure for Ollama
llm_config = LLMConfig(
    api_key="ollama",
    model="llama3.1:8b",
    small_model="llama3.1:8b",
    base_url="http://localhost:11434/v1",
)

graphiti = Graphiti(
    "bolt://localhost:7687",
    "neo4j",
    "password",
    llm_client=OpenAIGenericClient(config=llm_config),
    embedder=OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key="ollama",
            embedding_model="nomic-embed-text",
            embedding_dim=768,
            base_url="http://localhost:11434/v1",
        )
    ),
)
```

---

## 11. Langfuse Tracing & Observability

### Overview

[Langfuse](https://langfuse.com) is an open-source LLM observability platform that provides tracing, analytics, and evaluation for LLM applications. This integration enables real-time monitoring of RAG agent performance, including:

- **Trace Visualization**: See the complete execution flow of agent runs
- **Latency Tracking**: Monitor response times for LLM calls and tool executions
- **Cost Analysis**: Track token usage and associated costs
- **Error Monitoring**: Identify and debug failed requests
- **User Analytics**: Group traces by user and session for behavioral insights

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RAG Agent Request                        │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Langfuse Trace                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Span: Agent Run                                     │    │
│  │  ├── input: user query                               │    │
│  │  ├── Span: Tool Call (search_knowledge_base)         │    │
│  │  │   ├── input: query, search_type, match_count      │    │
│  │  │   ├── output: retrieved context                   │    │
│  │  │   └── duration: 150ms                             │    │
│  │  ├── Generation: LLM Response                        │    │
│  │  │   ├── model: llama3.1:8b                          │    │
│  │  │   ├── tokens: 450 input, 280 output               │    │
│  │  │   └── duration: 2.3s                              │    │
│  │  └── output: final response                          │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Setup

#### 1. Install Langfuse

```bash
pip install langfuse>=2.0.0
```

#### 2. Get API Keys

1. Sign up at [cloud.langfuse.com](https://cloud.langfuse.com) (free tier available)
2. Create a new project
3. Copy your **Public Key** and **Secret Key** from Settings → API Keys

#### 3. Configure Environment Variables

Add to your `.env` file:

```bash
# Langfuse Configuration
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
LANGFUSE_HOST=https://cloud.langfuse.com  # or your self-hosted URL
```

### Usage

#### Basic Usage with Traced Agent Run

The simplest way to enable tracing is to use the `traced_agent_run` function:

```python
from rag.agent.rag_agent import traced_agent_run

# Run with automatic tracing
result = await traced_agent_run(
    query="What is RAG?",
    user_id="user_123",        # Optional: group traces by user
    session_id="session_456",  # Optional: group traces by session
)

print(result.output)
```

#### Manual Trace Control

For more control over tracing, use the observability module directly:

```python
from rag.observability import (
    get_langfuse,
    trace_agent_run,
    trace_retrieval,
    trace_tool_call,
    shutdown_langfuse,
)

# Get Langfuse instance
langfuse = get_langfuse()

# Use context manager for tracing
with trace_agent_run("What is the PTO policy?", user_id="user_123") as trace:
    # Your agent logic here
    result = await agent.run(query)

    # Add custom spans
    trace_retrieval(
        trace=trace,
        query="PTO policy",
        search_type="hybrid",
        results_count=5,
    )

# Graceful shutdown
shutdown_langfuse()
```

#### Using the @observe Decorator

For custom functions that should be traced:

```python
from rag.observability import observe

@observe("custom_processing")
async def process_documents(docs: list) -> list:
    # Your processing logic
    processed = [transform(doc) for doc in docs]
    return processed
```

### Files Structure

| File | Purpose |
|------|---------|
| `rag/observability/__init__.py` | Module exports |
| `rag/observability/langfuse_integration.py` | Core Langfuse wrapper and utilities |
| `rag/config/settings.py` | Langfuse configuration settings |
| `rag/agent/rag_agent.py` | Integrated tracing in agent and tools |

### Key Functions

| Function | Description |
|----------|-------------|
| `get_langfuse()` | Get or create the global Langfuse instance |
| `trace_agent_run()` | Context manager for tracing agent runs |
| `trace_retrieval()` | Add retrieval span to a trace |
| `trace_tool_call()` | Add tool call span to a trace |
| `trace_llm_call()` | Add LLM generation span to a trace |
| `observe()` | Decorator for tracing function execution |
| `shutdown_langfuse()` | Gracefully flush and close Langfuse |
| `is_langfuse_enabled()` | Check if Langfuse is enabled and configured |

### Viewing Traces

Once configured, traces appear in your Langfuse dashboard:

1. Go to [cloud.langfuse.com](https://cloud.langfuse.com)
2. Select your project
3. Navigate to **Traces** to see all agent runs
4. Click on a trace to see:
   - Full execution timeline
   - Input/output for each step
   - Latency breakdown
   - Token usage and costs

### Self-Hosting Langfuse

For production or data privacy requirements, you can self-host Langfuse:

```bash
# Docker Compose
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d
```

Then update your `.env`:

```bash
LANGFUSE_HOST=http://localhost:3000
```

### Best Practices

1. **Use User IDs**: Always pass `user_id` to group traces by user for analytics
2. **Use Session IDs**: Pass `session_id` for multi-turn conversations
3. **Add Metadata**: Include relevant context in trace metadata
4. **Graceful Shutdown**: Call `shutdown_langfuse()` on application exit
5. **Error Handling**: Traces automatically capture errors with stack traces

### Configuration Reference

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| `langfuse_enabled` | `LANGFUSE_ENABLED` | `false` | Enable/disable Langfuse |
| `langfuse_public_key` | `LANGFUSE_PUBLIC_KEY` | `None` | Your Langfuse public key |
| `langfuse_secret_key` | `LANGFUSE_SECRET_KEY` | `None` | Your Langfuse secret key |
| `langfuse_host` | `LANGFUSE_HOST` | `https://cloud.langfuse.com` | Langfuse API host |

---

## 12. Implementation Roadmap

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
| Knowledge Graph | `knowledge_graph/graphiti_store.py` (new) | `retriever.py`, `pipeline.py`, `rag_agent.py` |
| Langfuse Tracing | `observability/langfuse_integration.py` | `rag_agent.py`, `settings.py` |
| Streamlit Web UI | `agent/streamlit_app.py` | `rag_agent.py` |

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

    # Knowledge Graph (Graphiti)
    graphiti_enabled: bool = False
    graph_db_type: str = "neo4j"  # neo4j, falkordb, kuzu
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    falkordb_host: str = "localhost"
    falkordb_port: int = 6379

    # Langfuse Observability (Already Implemented)
    langfuse_enabled: bool = False
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
```

---

## 13. Testing

### Run All Tests

```bash
python -m pytest rag/tests/ -v
```

### Run Specific Test Categories

```bash
# Configuration tests (fast, no external deps)
python -m pytest rag/tests/test_config.py -v

# Ingestion model tests (fast, no external deps)
python -m pytest rag/tests/test_ingestion.py -v

# MongoDB connection & index tests (requires MongoDB)
python -m pytest rag/tests/test_mongo_store.py -v

# RAG agent integration tests (requires MongoDB + Ollama)
python -m pytest rag/tests/test_rag_agent.py -v
python -m pytest rag/tests/test_rag_agent.py -v --log-cli-level=INFO --tb=short # log.info
```

### Debug Agent Flow

#### Execution Flow Summary

```
test_agent_flow_verbose (test_agent_flow.py)
    |
    +--> set_verbose_debug(True)
    |
    +--> stream_agent_interaction() (agent_main.py:57)
              |
              +--> _stream_agent() (agent_main.py:331)
                        |
                        +--> agent.iter(query) --> yields nodes
                                  |
                                  +--> NODE: UserPromptNode
                                  |         --> _debug_print()
                                  |
                                  +--> NODE: ModelRequestNode
                                  |         --> _handle_model_request_node() (agent_main.py:185)
                                  |                   |
                                  |                   +--> node.stream() --> yields events
                                  |                             |
                                  |                             +--> PartStartEvent (tool-call or text)
                                  |                             +--> PartDeltaEvent (TextPartDelta)
                                  |                             +--> FinalResultEvent
                                  |                             +--> PartEndEvent
                                  |
                                  +--> NODE: CallToolsNode
                                  |         --> _handle_tool_call_node() (agent_main.py:266)
                                  |                   |
                                  |                   +--> node.stream() --> yields events
                                  |                             |
                                  |                             +--> FunctionToolCallEvent
                                  |                             |         --> _extract_tool_info()
                                  |                             |         --> _display_tool_args()
                                  |                             |
                                  |                             +--> FunctionToolResultEvent
                                  |
                                  +--> NODE: End
                                            --> _debug_print("Execution complete")
```

#### Running the Tests

To verify the agent execution flow and see all Pydantic AI events:

```bash
# Run all agent flow tests
python -m pytest rag/tests/test_agent_flow.py -v -s

# Run a single test by full path
python -m pytest rag/tests/test_agent_flow.py::TestAgentFlow::test_agent_flow_verbose -v -s
python -m pytest rag/tests/test_agent_flow.py::TestAgentFlow::test_agent_flow_with_tool_call -v -s
python -m pytest rag/tests/test_agent_flow.py::TestAgentFlow::test_agent_flow_no_verbose -v -s

# Run tests matching a pattern
python -m pytest rag/tests/test_agent_flow.py -k "verbose" -v -s
```

The test enables verbose debugging via `set_verbose_debug(True)` from `agent_main.py`, which prints:

**Node Types:**
- `NODE #1: UserPromptNode` - Initial user input
- `NODE #2: ModelRequestNode` - LLM generating response
- `NODE #3: CallToolsNode` - Tool execution (search_knowledge_base)
- `NODE #N: End` - Execution complete

**Streaming Events (ModelRequestNode):**
- `PartStartEvent` - Start of text/tool-call part with initial content
- `PartDeltaEvent (TextPartDelta)` - Incremental text updates
- `FinalResultEvent` - Final result ready
- `PartEndEvent` - Part complete

**Tool Events (CallToolsNode):**
- `FunctionToolCallEvent` - Tool invocation with args (query, match_count, search_type)
- `FunctionToolResultEvent` - Tool result with search results

**Enabling Verbose Debug Programmatically:**
```python
from rag.agent.agent_main import set_verbose_debug, stream_agent_interaction

set_verbose_debug(True)  # Enable verbose output
# ... run agent ...
set_verbose_debug(False)  # Disable when done
```

### Test Categories

| Test File | What It Tests | Requirements |
|-----------|--------------|--------------|
| `test_config.py` | Settings loading, credential masking | None |
| `test_ingestion.py` | Data models, chunking config validation | None |
| `test_mongo_store.py` | MongoDB connection, vector/text indexes | MongoDB Atlas |
| `test_rag_agent.py` | Retriever queries, agent integration | MongoDB + Ollama |
| `test_agent_flow.py` | Agent flow execution, debug prints | MongoDB + Ollama |

### Sample Test Queries (from test_rag_agent.py)

The tests query the ingested NeuralFlow AI documents:

```python
# Company information
"What does NeuralFlow AI do?"
"How many engineers work at the company?"

# Employee benefits
"What is the PTO policy?"
"What is the learning budget for employees?"

# Technology
"What technologies and tools does the company use?"
```

### Expected Test Results

After successful ingestion of `rag/documents/`:
- `test_config.py`: 13 tests pass
- `test_ingestion.py`: 14 tests pass
- `test_mongo_store.py`: 5+ tests pass (some may skip if indexes not created)
- `test_rag_agent.py`: All tests pass (requires indexes + Ollama running)
