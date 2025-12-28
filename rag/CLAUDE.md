# MongoDB RAG Agent Development Instructions

## Project Overview

Agentic RAG system combining MongoDB Atlas Vector Search with Pydantic AI for intelligent document retrieval. Uses Docling for multi-format ingestion (PDF, DOCX, audio via Whisper ASR), async MongoDB operations, and hybrid search (vector + text with RRF). Built with Python 3.13, Ollama for local LLM/embeddings, and type-safe Pydantic models.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Validate configuration
python -m rag.main --validate

# Ingest documents
python -m rag.main --ingest --documents rag/documents

# Run tests
python -m pytest rag/tests/ -v
```

## Core Principles

1. **TYPE SAFETY IS NON-NEGOTIABLE**
   - All functions, methods, and variables MUST have type annotations
   - Use Pydantic models for all data structures
   - Import `Callable` from `collections.abc`, not lowercase `callable`

2. **KISS** (Keep It Simple, Stupid)
   - Prefer simple, readable solutions over clever abstractions
   - Trust MongoDB RRF - no manual score combination

3. **ASYNC ALL THE WAY**
   - All I/O operations MUST be async (MongoDB, embeddings, LLM calls)
   - Use `asyncio` for concurrent operations

---

## Project Structure

```
rag/
├── config/
│   └── settings.py          # Pydantic Settings configuration
├── ingestion/
│   ├── pipeline.py           # Document ingestion pipeline
│   ├── embedder.py           # Embedding generation (OpenAI-compatible)
│   ├── models.py             # Data models (ChunkData, SearchResult, etc.)
│   └── chunkers/
│       └── docling.py        # Docling HybridChunker integration
├── storage/
│   └── vector_store/
│       └── mongo.py          # MongoHybridStore (vector + text search)
├── retrieval/
│   └── retriever.py          # Search orchestrator
├── agent/
│   ├── rag_agent.py          # Pydantic AI agent with search tool
│   └── prompts.py            # System prompts
├── documents/                # Sample documents for ingestion
├── tests/                    # Test suite
│   ├── test_config.py        # Configuration tests
│   ├── test_ingestion.py     # Ingestion model tests
│   ├── test_mongo_store.py   # MongoDB connection & index tests
│   └── test_rag_agent.py     # RAG agent integration tests
└── main.py                   # CLI entry point
```

---

## Configuration

### Environment Variables (.env)

```bash
# MongoDB Atlas
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/?appName=MyApp

# LLM (Ollama local)
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama

# Embeddings (Ollama local)
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text:latest
EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_API_KEY=ollama
EMBEDDING_DIMENSION=768
```

### MongoDB Atlas Setup

Create these indexes in Atlas UI on the `chunks` collection:

**Vector Search Index** (`vector_index`):
```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 768,
      "similarity": "cosine"
    }
  ]
}
```

**Text Search Index** (`text_index`):
```json
{
  "mappings": {
    "dynamic": false,
    "fields": {
      "content": {
        "type": "string",
        "analyzer": "lucene.standard"
      }
    }
  }
}
```

---

## CLI Usage

### Validate Configuration
```bash
python -m rag.main --validate
```

### Ingest Documents
```bash
# Full ingestion (cleans existing data)
python -m rag.main --ingest --documents rag/documents

# Incremental ingestion (keeps existing data)
python -m rag.main --ingest --documents rag/documents --no-clean

# With custom chunking
python -m rag.main --ingest --documents rag/documents \
    --chunk-size 1000 \
    --chunk-overlap 200 \
    --max-tokens 512

# Verbose output
python -m rag.main --ingest --documents rag/documents --verbose
```

### Supported File Formats
- Text: `.md`, `.markdown`, `.txt`
- Documents: `.pdf`, `.docx`, `.doc`, `.pptx`, `.ppt`, `.xlsx`, `.xls`, `.html`
- Audio: `.mp3`, `.wav`, `.m4a`, `.flac` (requires `openai-whisper`)

---

## Testing

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

### Test Categories

| Test File | What It Tests | Requirements |
|-----------|--------------|--------------|
| `test_config.py` | Settings loading, credential masking | None |
| `test_ingestion.py` | Data models, chunking config validation | None |
| `test_mongo_store.py` | MongoDB connection, vector/text indexes | MongoDB Atlas |
| `test_rag_agent.py` | Retriever queries, agent integration | MongoDB + Ollama |

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

---

## Code Quality

### Linting & Formatting
```bash
# Check and fix
ruff check --fix rag/

# Format
ruff format rag/
```

### Makefile Commands
```bash
make ruff    # Run ruff check --fix and format
```

---

## Architecture

### Two-Collection Pattern

**`documents` collection**: Full document metadata
```python
{
    "_id": ObjectId,
    "title": str,
    "source": str,  # Relative file path
    "content": str,  # Full document text
    "metadata": dict,
    "created_at": datetime
}
```

**`chunks` collection**: Embedded chunks for search
```python
{
    "_id": ObjectId,
    "document_id": ObjectId,  # Reference to documents
    "content": str,
    "embedding": list[float],  # 768-dim for nomic-embed-text
    "chunk_index": int,
    "metadata": dict,
    "token_count": int,
    "created_at": datetime
}
```

### Hybrid Search (RRF)

The system combines vector and text search using Reciprocal Rank Fusion:

```python
# Vector search for semantic similarity
semantic_results = await store.semantic_search(query_embedding, count)

# Text search for keyword matching
text_results = await store.text_search(query, count)

# Merge with RRF (k=60)
merged = reciprocal_rank_fusion([semantic_results, text_results])
```

---

## Common Issues

### 1. "Vector index not found"
Create the `vector_index` in MongoDB Atlas UI (see Configuration section).

### 2. "callable is not subscriptable"
Use `Callable` from `collections.abc`:
```python
from collections.abc import Callable
def func(callback: Callable | None = None): ...
```

### 3. Audio transcription fails
Audio transcription requires both FFmpeg (in PATH) and Whisper:
```bash
# Install FFmpeg (system-level) - must be in PATH
# Windows (Chocolatey): choco install ffmpeg
#   Default path: C:\ProgramData\chocolatey\lib\ffmpeg\tools\ffmpeg\bin
# Windows (WinGet): winget install ffmpeg
# macOS: brew install ffmpeg
# Linux: sudo apt install ffmpeg

# Verify FFmpeg is in PATH
ffmpeg -version

# Install Whisper
pip install openai-whisper
```
If dependencies are missing, audio files are stored with `[Error: Could not transcribe audio file ...]` placeholder.

### 4. Ollama connection refused
Start Ollama server:
```bash
ollama serve
```

### 5. Embedding dimension mismatch
Ensure `EMBEDDING_DIMENSION` matches your model:
- `nomic-embed-text`: 768
- `text-embedding-3-small`: 1536
- `text-embedding-ada-002`: 1536

---

## Development Workflow

1. **Make changes** to source files in `rag/`
2. **Run linting**: `ruff check --fix rag/ && ruff format rag/`
3. **Run tests**: `python -m pytest rag/tests/ -v`
4. **Test manually** if needed:
   ```python
   import asyncio
   from rag.retrieval.retriever import Retriever
   from rag.storage.vector_store.mongo import MongoHybridStore

   async def test():
       store = MongoHybridStore()
       retriever = Retriever(store=store)
       results = await retriever.retrieve("What does the company do?")
       for r in results:
           print(f"{r.document_title}: {r.content[:100]}...")
       await store.close()

   asyncio.run(test())
   ```

---

## Quick Reference

### Run the Agent Programmatically
```python
import asyncio
from rag.agent.rag_agent import agent

async def main():
    result = await agent.run("What services does NeuralFlow AI provide?")
    print(result.output)

asyncio.run(main())
```

### Search Without Agent
```python
import asyncio
from rag.retrieval.retriever import Retriever
from rag.storage.vector_store.mongo import MongoHybridStore

async def search(query: str):
    store = MongoHybridStore()
    retriever = Retriever(store=store)

    # Get search results
    results = await retriever.retrieve(query, match_count=5)

    # Or get formatted context for LLM
    context = await retriever.retrieve_as_context(query)

    await store.close()
    return results, context

asyncio.run(search("employee benefits"))
```
