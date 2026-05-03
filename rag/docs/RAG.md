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
10. [Langfuse Tracing & Observability](#10-langfuse-tracing--observability)
11. [Implementation Roadmap](#11-implementation-roadmap)
12. [Testing](#12-testing)
13. [Performance Tuning](#13-performance-tuning)
14. [Caching](#14-caching)
15. [Mem0 Memory Layer](#15-mem0-memory-layer)
16. [RAG-Anything Modal Processors](#16-rag-anything-modal-processors)

---

## 1. Current Architecture

### System Flow
```
Documents → Ingestion Pipeline → Chunking → Embedding → PostgreSQL → Retrieval → Agent
```

#### Ingestion Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE WORKFLOW                           │
└─────────────────────────────────────────────────────────────────────────────┘

python -m rag.main --ingest --documents rag/documents
    │
    ▼
run_ingestion_pipeline() [pipeline.py:496]
    │
    ├──► argparse.ArgumentParser()
    ├──► IngestionConfig(chunk_size, chunk_overlap, max_tokens)
    │
    ▼
DocumentIngestionPipeline.__init__() [pipeline.py:32]
    │
    ├──► load_settings()                        [settings.py]
    ├──► ChunkingConfig()                       [models.py]
    ├──► create_chunker(config)                 [docling.py]
    │       └──► DoclingHybridChunker()
    │               ├──► AutoTokenizer.from_pretrained()
    │               └──► HybridChunker()
    ├──► create_embedder()                      [embedder.py]
    │       └──► EmbeddingGenerator()
    │               └──► openai.AsyncOpenAI()
    └──► PostgresHybridStore()                     [postgres.py]
            └──► load_settings()
    │
    ▼
pipeline.ingest_documents() [pipeline.py:366]
    │
    ├──► pipeline.initialize()
    │       └──► store.initialize()
    │               └──► asyncpg.connect()
    │               └──► CREATE EXTENSION IF NOT EXISTS vector
    │
    ├──► [if clean_before_ingest]:
    │       └──► store.clean_collections()
    │               ├──► TRUNCATE TABLE chunks
    │               └──► TRUNCATE TABLE documents
    │
    ├──► _find_document_files()
    │       └──► glob.glob("**/*.{md,pdf,docx,...}")
    │
    ▼
[FOR EACH FILE] ─────────────────────────────────────────────────────────────┐
    │                                                                         │
    ├──► _compute_file_hash()                                                 │
    │       └──► hashlib.md5()                                                │
    │                                                                         │
    ├──► [INCREMENTAL MODE - if not clean_before_ingest]:                     │
    │       ├──► store.get_document_hash(source)                              │
    │       │       └──► [UNCHANGED?] ──► SKIP file                           │
    │       │       └──► [CHANGED?] ──► store.delete_document_and_chunks()    │
    │       └──► [NEW?] ──► continue to ingest                                │
    │                                                                         │
    ▼                                                                         │
_ingest_single_document(file_path) [pipeline.py:300]                          │
    │                                                                         │
    ├──► _read_document()                                                     │
    │       │                                                                 │
    │       ├──► [AUDIO: .mp3, .wav, .m4a, .flac]:                            │
    │       │       └──► _transcribe_audio()                                  │
    │       │               └──► DocumentConverter() with AsrPipeline         │
    │       │                       └──► Whisper ASR transcription            │
    │       │                                                                 │
    │       ├──► [DOCLING: .pdf, .docx, .pptx, .xlsx, .html, .md]:            │
    │       │       └──► DocumentConverter.convert()                          │
    │       │               └──► result.document.export_to_markdown()         │
    │       │                                                                 │
    │       └──► [TEXT: .txt, other]:                                         │
    │               └──► open(file_path).read()                               │
    │                                                                         │
    ├──► _extract_title()                                                     │
    │       └──► [Find "# " heading or use filename]                          │
    │                                                                         │
    ├──► _extract_document_metadata()                                         │
    │       ├──► _compute_file_hash()                                         │
    │       └──► [Parse YAML frontmatter if present]                          │
    │                                                                         │
    ▼                                                                         │
chunker.chunk_document() [docling.py:61]                                      │
    │                                                                         │
    ├──► [WITH DoclingDocument]:                                              │
    │       ├──► self.chunker.chunk(dl_doc)     ──► yields chunks             │
    │       ├──► self.chunker.contextualize(chunk)  ──► adds heading context  │
    │       ├──► self.tokenizer.count_tokens()                                │
    │       └──► ChunkData(content, index, metadata, token_count)             │
    │                                                                         │
    └──► [WITHOUT DoclingDocument - fallback]:                                │
            └──► _simple_fallback_chunk()                                     │
                    └──► [Sliding window with sentence boundary detection]    │
    │                                                                         │
    ▼                                                                         │
embedder.embed_chunks(chunks) [embedder.py:122]                               │
    │                                                                         │
    ├──► [FOR EACH BATCH of batch_size]:                                      │
    │       ├──► generate_embeddings_batch(texts)                             │
    │       │       └──► client.embeddings.create(model, input=texts)         │
    │       │               └──► [Ollama/OpenAI API call]                     │
    │       │                                                                 │
    │       └──► ChunkData(content, embedding, metadata)                      │
    │                                                                         │
    ▼                                                                         │
store.save_document() [postgres.py:340]                                          │
    │                                                                         │
    └──► INSERT INTO documents (title, source, content, metadata)             │
        └──► Returns document_id (UUID)                                       │
    │                                                                         │
    ▼                                                                         │
store.add(chunks, document_id) [postgres.py:61]                                  │
    │                                                                         │
    └──► executemany INSERT INTO chunks (                                    │
            document_id, content, embedding, chunk_index,                     │
            metadata, token_count                                              │
        ) [batch insert]                                                      │
    │                                                                         │
    ▼                                                                         │
IngestionResult(document_id, title, chunks_created, processing_time_ms)       │
    │                                                                         │
◄─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
[INCREMENTAL MODE - handle deleted files]:
    ├──► store.get_all_document_sources()
    └──► [For each source not in current files]:
            └──► store.delete_document_and_chunks(source)
    │
    ▼
pipeline.close()
    └──► store.close()
            └──► pool.close()
    │
    ▼
INGESTION COMPLETE
    ├──► Log: "{N} documents processed, {M} chunks created"
    └──► Tables and indexes created automatically by PostgresHybridStore
```

#### Ingestion Modes

| Mode | Flag | Behavior |
|------|------|----------|
| **Full** (default) | `--ingest` | Deletes all existing data, re-ingests everything |
| **Incremental** | `--ingest --no-clean` | Only processes new/changed files, removes deleted files |

#### Supported File Formats

| Category | Extensions | Processing Method |
|----------|------------|-------------------|
| **Text** | `.md`, `.markdown`, `.txt` | Direct read / Docling |
| **Documents** | `.pdf`, `.docx`, `.doc`, `.pptx`, `.xlsx`, `.html` | Docling DocumentConverter |
| **Audio** | `.mp3`, `.wav`, `.m4a`, `.flac` | Whisper ASR via Docling (requires additional setup) |

> **Audio Transcription Prerequisites**
>
> Audio file processing requires:
> 1. **FFmpeg** - System-level audio decoder (must be in PATH)
>    - Windows (Chocolatey): `choco install ffmpeg`
>      - Default path: `C:\ProgramData\chocolatey\lib\ffmpeg\tools\ffmpeg\bin`
>    - Windows (WinGet): `winget install ffmpeg`
>    - macOS: `brew install ffmpeg`
>    - Linux: `sudo apt install ffmpeg`
>
>    **Important**: Ensure FFmpeg is in your system PATH. Verify with: `ffmpeg -version`
>
> 2. **OpenAI Whisper** - Speech recognition model
>    ```bash
>    pip install openai-whisper
>    ```
>
> Without these dependencies, audio files will be ingested with an error placeholder instead of transcribed content.

#### Chunking Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                    HYBRID CHUNKER FLOW                           │
└─────────────────────────────────────────────────────────────────┘

DoclingDocument (structured)
    │
    ▼
HybridChunker.chunk(dl_doc)
    │
    ├──► Respects document structure (sections, tables, lists)
    ├──► Token-aware splitting (max_tokens=512)
    ├──► Merges small peer sections (merge_peers=True)
    │
    ▼
HybridChunker.contextualize(chunk)
    │
    └──► Prepends heading hierarchy to chunk
         Example: "## Company Overview\n### Mission\nOur mission is..."
    │
    ▼
ChunkData(content, index, metadata, token_count, embedding=None)
```

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| Ingestion | `rag/ingestion/pipeline.py` | Multi-format document processing, incremental indexing |
| Chunking | `rag/ingestion/chunkers/docling.py` | Docling HybridChunker (token-aware, structure-preserving) |
| Embedding | `rag/ingestion/embedder.py` | OpenAI-compatible API (Ollama, OpenAI) |
| Storage | `rag/storage/vector_store/postgres.py` | PostgreSQL/pgvector with vector + text search |
| Retrieval | `rag/retrieval/retriever.py` | Semantic, text, and hybrid (RRF) search |
| Agent | `rag/agent/rag_agent.py` | Pydantic AI agent with search tool |
| Config | `rag/config/settings.py` | Environment-based configuration |

### Current Search Methods

| Method | Score Range | Best For |
|--------|-------------|----------|
| Semantic | 0.0 - 1.0 | Conceptual queries, paraphrases |
| Text | 0.0 - 10.0+ | Exact matches, keywords, acronyms |
| Hybrid (RRF) | 0.01 - 0.03 | Balanced retrieval (default) |

### Component Call Graphs

Detailed call graphs showing how class methods interact within each component.

#### 1. Settings (`rag/config/settings.py`)

```
Class: Settings(BaseSettings)
├── model_config (class var)
└── [Fields: database_url, postgres_table_documents, llm_model, etc.]

Functions:
┌──────────────────────┐
│  load_settings()     │ ──────────► Settings()
└──────────────────────┘                │
                                        └──► BaseSettings.__init__()
                                             └──► load_dotenv()

┌──────────────────────┐
│  mask_credential()   │ ──────────► str[:4] + "..." + str[-4:]
└──────────────────────┘

Module-level:
settings = load_settings()   # Singleton instance
```

#### 2. Ingestion Pipeline (`rag/ingestion/pipeline.py`)

```
Class: DocumentIngestionPipeline

__init__(config, documents_folder, clean_before_ingest)
    ├──► load_settings()                    [settings.py]
    ├──► ChunkingConfig()                   [models.py]
    ├──► create_chunker(config)             [chunkers/docling.py]
    ├──► create_embedder()                  [embedder.py]
    └──► PostgresHybridStore()                 [postgres.py]

initialize()
    └──► self.store.initialize()            [postgres.py]

close()
    └──► self.store.close()                 [postgres.py]

ingest_documents(progress_callback)
    ├──► self.initialize()
    ├──► self.store.clean_collections()     (if clean_before_ingest)
    ├──► self._find_document_files()
    │       └──► glob.glob()
    ├──► [For each file]:
    │       ├──► self._compute_file_hash()
    │       │       └──► hashlib.md5()
    │       ├──► self.store.get_document_hash()      (incremental mode)
    │       ├──► self.store.delete_document_and_chunks()  (if updated)
    │       └──► self._ingest_single_document()
    ├──► self.store.get_all_document_sources()       (incremental mode)
    └──► self.store.delete_document_and_chunks()     (for deleted files)

_ingest_single_document(file_path)
    ├──► self._read_document()
    │       ├──► self._transcribe_audio()           (for audio files)
    │       │       └──► DocumentConverter.convert()
    │       └──► DocumentConverter.convert()        (for docling formats)
    ├──► self._extract_title()
    ├──► self._extract_document_metadata()
    │       └──► self._compute_file_hash()
    ├──► self.chunker.chunk_document()              [docling.py]
    ├──► self.embedder.embed_chunks()               [embedder.py]
    ├──► self.store.save_document()                 [postgres.py]
    └──► self.store.add()                           [postgres.py]

run_ingestion_pipeline() [async main]
    ├──► argparse.ArgumentParser()
    ├──► IngestionConfig()
    ├──► DocumentIngestionPipeline()
    ├──► pipeline.ingest_documents()
    └──► pipeline.close()
```

#### 3. Chunker (`rag/ingestion/chunkers/docling.py`)

```
Class: DoclingHybridChunker

__init__(config: ChunkingConfig)
    ├──► AutoTokenizer.from_pretrained(TOKENIZER_MODEL)  [transformers]
    └──► HybridChunker(tokenizer, max_tokens, merge_peers)  [docling]

chunk_document(content, title, source, metadata, docling_doc)
    ├──► [if no docling_doc]:
    │       └──► self._simple_fallback_chunk()
    │
    └──► [with docling_doc]:
            ├──► self.chunker.chunk(dl_doc)             [docling]
            ├──► self.chunker.contextualize(chunk)      [docling]
            ├──► self.tokenizer.count_tokens()          [transformers]
            └──► ChunkData()                            [models.py]

            [on exception]:
            └──► self._simple_fallback_chunk()

_simple_fallback_chunk(content, base_metadata)
    ├──► [sliding window loop]:
    │       ├──► self.tokenizer.count_tokens()
    │       └──► ChunkData()
    └──► [update total_chunks in metadata]

create_chunker(config) ──────────► DoclingHybridChunker(config)
```

#### 4. Embedder (`rag/ingestion/embedder.py`)

```
Module-level:
_client: AsyncOpenAI | None
_settings = None

_get_client()
    ├──► load_settings()                    [settings.py]
    └──► openai.AsyncOpenAI()               [openai]

@alru_cache(maxsize=1000)
_cached_embed(text, model)
    ├──► _get_client()
    └──► client.embeddings.create()         [openai]

Class: EmbeddingGenerator

__init__(model, batch_size)
    ├──► load_settings()                    [settings.py]
    └──► openai.AsyncOpenAI()               [openai]

generate_embedding(text)
    └──► self.client.embeddings.create()    [openai]

generate_embeddings_batch(texts)
    └──► self.client.embeddings.create()    [openai]

embed_chunks(chunks, progress_callback)
    └──► [For each batch]:
            ├──► self.generate_embeddings_batch()
            └──► ChunkData()                [models.py]

embed_query(query, use_cache)
    ├──► [if use_cache]:
    │       └──► _cached_embed()            ──► [CACHE HIT or MISS]
    └──► [else]:
            └──► self.generate_embedding()

get_cache_stats() [static]
    └──► _cached_embed.cache_info()         [@alru_cache]

clear_cache() [static]
    └──► _cached_embed.cache_clear()        [@alru_cache]

get_embedding_dimension()
    └──► self.config["dimensions"]

create_embedder(model, **kwargs) ──────────► EmbeddingGenerator(model, **kwargs)
```

#### 5. PostgreSQL Store (`rag/storage/vector_store/postgres.py`)

```
Class: PostgresHybridStore

__init__()
    └──► load_settings()                    [settings.py]

initialize()
    ├──► asyncpg.connect()                 [asyncpg]
    └──► CREATE EXTENSION IF NOT EXISTS vector
    └──► CREATE TABLE IF NOT EXISTS documents, chunks

close()
    └──► self.pool.close()                  [asyncpg]

add(chunks, document_id)
    ├──► self.initialize()
    └──► INSERT INTO chunks (...)           [asyncpg]

semantic_search(query_embedding, match_count)
    ├──► self.initialize()
    ├──► SELECT ... ORDER BY embedding <=> $1::vector  [asyncpg/pgvector]
    └──► SearchResult()                     [models.py]

text_search(query, match_count)
    ├──► self.initialize()
    ├──► SELECT ... WHERE content_tsv @@ plainto_tsquery(...)  [asyncpg]
    └──► SearchResult()                     [models.py]

hybrid_search(query, query_embedding, match_count)
    ├──► self.initialize()
    ├──► asyncio.gather(
    │       self.semantic_search(),
    │       self.text_search()
    │   )
    └──► self._reciprocal_rank_fusion()

_reciprocal_rank_fusion(search_results_list, k=60)
    ├──► [calculate RRF scores]
    ├──► sorted()
    └──► SearchResult()                     [models.py]

save_document(title, source, content, metadata)
    ├──► self.initialize()
    └──► INSERT INTO documents (...)        [asyncpg]

clean_collections()
    ├──► self.initialize()
    └──► TRUNCATE TABLE chunks, documents   [asyncpg]

get_document_by_source(source)
    ├──► self.initialize()
    └──► SELECT * FROM documents WHERE source = $1  [asyncpg]

get_document_hash(source)
    └──► self.get_document_by_source()

delete_document_and_chunks(source)
    ├──► self.initialize()
    └──► DELETE FROM documents WHERE source = $1 (CASCADE deletes chunks)

get_all_document_sources()
    ├──► self.initialize()
    └──► SELECT source FROM documents       [asyncpg]
```

#### 6. Retriever (`rag/retrieval/retriever.py`)

```
Class: ResultCache

__init__(max_size, ttl_seconds)

_get_key(query, search_type, match_count)
    └──► hashlib.sha256()

get(query, search_type, match_count)
    ├──► self._get_key()
    ├──► [Check TTL]
    └──► [Return cached results or None]

set(query, search_type, match_count, results)
    ├──► self._get_key()
    └──► [LRU eviction if over max_size]

stats()
    └──► [Return hit/miss statistics]

clear()

Module-level:
_result_cache = ResultCache(max_size=100, ttl_seconds=300)

─────────────────────────────────────────────────────────────────

Class: Retriever

__init__(store, embedder, reranker, hyde)
    ├──► load_settings()                    [settings.py]
    ├──► PostgresHybridStore()              (if store not provided)
    ├──► EmbeddingGenerator()              (if embedder not provided)
    ├──► self._reranker = reranker          (lazy-init from settings if None)
    └──► self._hyde = hyde                  (lazy-init from settings if None)

_get_hyde()  [lazy-init]
    └──► HyDEProcessor(model, base_url, api_key, embedding_model, ...)

_get_reranker()  [lazy-init]
    ├──► [reranker_type == "cross_encoder"]:
    │       └──► CrossEncoderReranker(model_name)
    └──► [reranker_type == "llm"]:
            └──► LLMReranker(model, base_url, api_key)

retrieve(query, match_count, search_type, use_cache)
    ├──► 1. [if use_cache]:
    │           └──► _result_cache.get()
    │                   └──► [CACHE HIT] ──► return cached results
    │
    ├──► 2. Query embedding:
    │       ├──► [hyde_enabled=True]:
    │       │       ├──► _get_hyde().generate_hypothetical(query)  [LLM call]
    │       │       └──► embedder.generate_embedding(hypothetical)
    │       └──► [hyde_enabled=False]:
    │               └──► embedder.embed_query(query)               [embedder.py]
    │
    ├──► 3. fetch_count = match_count × reranker_overfetch_factor  (if reranking)
    │
    ├──► 4. [based on search_type]:
    │       ├──► self.store.semantic_search()   [postgres.py]
    │       ├──► self.store.text_search()       [postgres.py]
    │       └──► self.store.hybrid_search()     [postgres.py] (default)
    │
    ├──► 5. [if reranker_enabled]:
    │       └──► _get_reranker().rerank(query, results, top_k=match_count)
    │
    └──► 6. [if use_cache]:
            └──► _result_cache.set()

get_cache_stats() [static]
    └──► _result_cache.stats()

clear_cache() [static]
    └──► _result_cache.clear()

retrieve_as_context(query, match_count, search_type)
    ├──► self.retrieve()
    └──► [Format results as string]

close()
    └──► self.store.close()                 [postgres.py]
```

#### 7. RAG Agent (`rag/agent/rag_agent.py`)

```
Module-level:
_trace_context: ContextVar  (per-coroutine trace isolation)

get_llm_model(model_choice)
    ├──► load_settings()                    [settings.py]
    ├──► OpenAIProvider()                   [pydantic_ai]
    └──► OpenAIChatModel()                  [pydantic_ai]

get_model_info()
    └──► load_settings()                    [settings.py]

agent = PydanticAgent(get_llm_model(), system_prompt=MAIN_SYSTEM_PROMPT)

─────────────────────────────────────────────────────────────────

Class: RAGState(BaseModel)
    _store:       PostgresHybridStore  (PrivateAttr)
    _retriever:   Retriever            (PrivateAttr)
    _mem0:        Mem0Store            (PrivateAttr, if mem0_enabled)
    _initialized: bool                 (PrivateAttr)
    _init_lock:   asyncio.Lock         (PrivateAttr)

get_retriever()
    ├──► [if not initialized, under _init_lock]:
    │       ├──► PostgresHybridStore()         [postgres.py]
    │       ├──► store.initialize()
    │       ├──► EmbeddingGenerator()          [embedder.py]
    │       ├──► Retriever(store, embedder)    [retriever.py]
    │       └──► Mem0Store()  (if mem0_enabled) [mem0_store.py]
    └──► return self._retriever

close()
    └──► self._store.close()                [postgres.py]

─────────────────────────────────────────────────────────────────

@agent.tool
search_knowledge_base(ctx, query, match_count, search_type)
    │
    ├──► RAGState.get_retriever()           (from ctx.deps)
    │
    ├──► retriever.retrieve_as_context()    [retriever.py]
    │
    ├──► mem0_store.get_context_string()    [mem0_store.py]  (if mem0_enabled)
    │
    └──► return combined context string

─────────────────────────────────────────────────────────────────

traced_agent_run(query, user_id, session_id, message_history)
    │
    ├──► _trace_context.set(trace)          (ContextVar — per coroutine)
    │
    ├──► get_langfuse()                     [observability]
    │       └──► [if langfuse enabled]: langfuse.trace()
    │
    ├──► state = RAGState(user_id=user_id)
    │
    ├──► agent.run(query, deps=state, message_history)  [pydantic_ai]
    │       └──► [Internally calls]: search_knowledge_base()
    │
    └──► [finally]:
            ├──► state.close()
            ├──► langfuse.flush()
            └──► _trace_context.set(None)
```

### Complete System Call Flow

End-to-end flow when a user query is processed:

```
USER QUERY
    │
    ▼
traced_agent_run() [rag_agent.py]
    │
    ├──► get_langfuse() [observability]
    │
    ▼
agent.run() [pydantic_ai]
    │
    ├──► LLM decides to use tool
    │
    ▼
search_knowledge_base() [rag_agent.py:101]
    │
    ├──► RAGState.get_retriever() ─────────────────────────────┐
    │       ├──► PostgresHybridStore() [postgres.py]                  │
    │       │       └──► load_settings() [settings.py]          │
    │       ├──► store.initialize()                             │
    │       │       └──► asyncpg.connect()                     │
    │       └──► Retriever(store) [retriever.py]                │
    │               ├──► load_settings()                        │
    │               └──► EmbeddingGenerator() [embedder.py]     │
    │                       └──► load_settings()                │
    │                                                           │
    ▼◄──────────────────────────────────────────────────────────┘
retriever.retrieve_as_context() [retriever.py:189]
    │
    ├──► retriever.retrieve()
    │       │
    │       ├──► _result_cache.get() ──► [CACHE HIT?] ──► return cached
    │       │
    │       ├──► embedder.embed_query() [embedder.py:178]
    │       │       └──► _cached_embed() ──► [CACHE HIT?] ──► return cached
    │       │               └──► client.embeddings.create() [openai]
    │       │
    │       └──► store.hybrid_search() [postgres.py:238]
    │               │
    │               ├──► asyncio.gather(
    │               │       semantic_search(),
    │               │       text_search()
    │               │   )
    │               │
    │               ├──► semantic_search() ──► ORDER BY embedding <=> $1::vector
    │               ├──► text_search() ──► WHERE content_tsv @@ plainto_tsquery
    │               │
    │               └──► _reciprocal_rank_fusion()
    │
    └──► [Format results as context string]
            │
            ▼
    Return to LLM for response generation
            │
            ▼
    LLM generates final response
            │
            ▼
    RESPONSE TO USER
```

### Key Design Patterns

| Pattern | Where Used | Purpose |
|---------|------------|---------|
| **Lazy Initialization** | `RAGState.get_retriever()`, `PostgresHybridStore.initialize()` | Avoid event loop issues, defer connection until needed |
| **Two-Level Caching** | `@alru_cache` (embeddings), `ResultCache` (search results) | Reduce API calls and DB queries |
| **Dependency Injection** | `Retriever(store, embedder)` | Testability, flexibility |
| **Factory Functions** | `create_chunker()`, `create_embedder()` | Encapsulate instantiation logic |
| **Singleton** | `settings = load_settings()`, `_result_cache` | Share configuration and cache globally |
| **RRF Fusion** | `_reciprocal_rank_fusion()` | Combine semantic + text search results |

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
│  🔍 RAG Agent                  │  💬 Chat with RAG Agent            │
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

1. Ensure PostgreSQL is configured with the pgvector extension and indexes
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
| PostgreSQL connection error | Check `DATABASE_URL` in `.env` file |
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

### Current Implementation: DoclingHybridChunker

**Location**: [`rag/ingestion/chunkers/docling.py`](../rag/ingestion/chunkers/docling.py)

#### What It Does

`DoclingHybridChunker` wraps Docling's built-in `HybridChunker`. Rather than splitting text blindly at character counts, it understands the document's structure — sections, headings, paragraphs, tables, code blocks — and uses that structure to find natural chunk boundaries.

#### Initialisation

```python
# docling.py:130-135
TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer_obj = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
self.chunker = HybridChunker(
    tokenizer=tokenizer_obj,
    max_tokens=config.max_tokens,   # hard token ceiling per chunk
    merge_peers=True,               # merge small sibling sections
)
```

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `max_tokens` | `512` | Hard token ceiling — chunks never exceed this |
| `merge_peers` | `True` | Merges short adjacent sibling sections into one chunk |
| Tokenizer | `all-MiniLM-L6-v2` | Used for accurate token counting (not for embeddings) |
| `chunk_size` | `1000` | Character limit for fallback chunker |
| `chunk_overlap` | `200` | Overlap for fallback sliding window |

#### Requires a `DoclingDocument`

The primary path requires a `DoclingDocument` object — the structured representation produced by Docling's `DocumentConverter`. This is passed in from the ingestion pipeline directly (the converter result is reused, not re-run):

```python
# pipeline.py — converter result passed directly to chunker
chunks = await self.chunker.chunk_document(
    content=markdown_text,
    title=title,
    source=str(file_path),
    metadata=metadata,
    docling_doc=docling_doc,   # ← structured doc from DocumentConverter
)
```

If `docling_doc` is `None` (e.g. for plain `.txt` files), the chunker logs a warning and falls back to `_simple_fallback_chunk()`.

#### `contextualize()` — Heading Context Prepended

After chunking, each chunk is passed through `HybridChunker.contextualize()`:

```python
# docling.py:191
contextualized_text = self.chunker.contextualize(chunk=chunk)
```

This prepends the full heading hierarchy to the chunk text:

```
## Company Overview
### Mission Statement
Our mission is to build AI tools that...
```

Without contextualization, the second chunk of a section would contain only the body text, with no indication of which section it belongs to. With it, every chunk is self-contained — the embedding captures both the topic (heading) and the content (body).

#### Fallback: `_simple_fallback_chunk()`

Used when no `DoclingDocument` is available or when `HybridChunker` raises an exception:

```python
# docling.py:249-292
# Sliding window with sentence-boundary detection
while start < len(content):
    end = start + chunk_size
    # Walk back from end to find ".!?\n" sentence boundary
    for i in range(end, max(start + min_chunk_size, end - 200), -1):
        if content[i] in ".!?\n":
            chunk_end = i + 1
            break
    chunks.append(ChunkData(content=chunk_text, ...))
    start = end - overlap   # advance with overlap
```

Chunks are tagged `chunk_method: "simple_fallback"` in metadata so you can distinguish them.

#### Output: `ChunkData`

Each chunk carries:

| Field | Contents |
|-------|----------|
| `content` | Contextualized chunk text (heading hierarchy + body) |
| `index` | Position within document |
| `start_char` / `end_char` | Estimated character offsets |
| `token_count` | Exact token count via the tokenizer |
| `metadata.chunk_method` | `"hybrid"` or `"simple_fallback"` |
| `metadata.has_context` | `True` when heading context was prepended |
| `metadata.total_chunks` | Total chunk count for the document |

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
- Update PostgreSQL schema in `rag/storage/vector_store/postgres.py`

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

**Schema extension** in PostgreSQL:
```sql
-- Add to chunks table
ALTER TABLE chunks ADD COLUMN parent_chunk_id UUID REFERENCES chunks(id);
ALTER TABLE chunks ADD COLUMN hierarchy_level INTEGER DEFAULT 0;  -- 0=parent, 1=child
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
        store: PostgresHybridStore | None = None,
        embedder: EmbeddingGenerator | None = None,
        reranker: Reranker | None = None,  # Add this
    ):
        self.store = store or PostgresHybridStore()
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
- Modify `rag/storage/vector_store/postgres.py`

#### 5.1 Adjacent Chunk Expander

```python
# rag/retrieval/context_expanders.py
class AdjacentChunkExpander:
    def __init__(self, store: PostgresHybridStore):
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

#### 5.2 PostgreSQL Helper Methods

```python
# rag/storage/vector_store/postgres.py
async def get_chunk_by_id(self, chunk_id: str) -> dict:
    """Get a single chunk by ID."""
    row = await self._pool.fetchrow(
        "SELECT * FROM chunks WHERE id = $1", chunk_id
    )
    return dict(row) if row else None

async def get_chunks_by_document(
    self,
    document_id: str,
    start_index: int,
    end_index: int
) -> list[dict]:
    """Get chunks for a document within index range."""
    rows = await self._pool.fetch(
        """SELECT * FROM chunks
           WHERE document_id = $1
             AND chunk_index >= $2 AND chunk_index < $3
           ORDER BY chunk_index""",
        document_id, start_index, end_index
    )
    return [dict(r) for r in rows]
```

---

## 7. Parent-Child Document Retrieval

### Current State
Flat chunk structure, no hierarchical relationships.

### Adding Hierarchical Retrieval

**Files to modify**:
- `rag/storage/vector_store/postgres.py`
- `rag/ingestion/pipeline.py`
- `rag/retrieval/retriever.py`

#### 6.1 Schema Extension

```sql
-- Extended chunks table with hierarchical fields
ALTER TABLE chunks ADD COLUMN parent_chunk_id UUID REFERENCES chunks(id);
ALTER TABLE chunks ADD COLUMN hierarchy_level INTEGER DEFAULT 0;
  -- 0=leaf, 1=parent, 2=grandparent
ALTER TABLE chunks ADD COLUMN section_path TEXT;  -- e.g., "1.2.3" for nested sections
```

#### 6.2 Parent Retrieval

```python
# rag/storage/vector_store/postgres.py
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
- `rag/storage/vector_store/postgres.py`
- `rag/agent/rag_agent.py`

#### 7.1 Filter Implementation

```python
# rag/storage/vector_store/postgres.py
async def semantic_search(
    self,
    query_embedding: list[float],
    match_count: int | None = None,
    filters: dict | None = None  # Add this
) -> list[SearchResult]:
    # Build filter stage
    filter_stage = self._build_filter_stage(filters) if filters else None

    where_clause, params = self._build_filter_clause(filters) if filters else ("", [])
    params_base = [query_embedding, match_count]
    offset = len(params_base)
    # Inject filter params after base params
    all_params = params_base + params
    sql = f"""
        SELECT c.*, d.title, d.source,
               1 - (c.embedding <=> $1::vector) AS similarity
        FROM chunks c JOIN documents d ON c.document_id = d.id
        {where_clause}
        ORDER BY c.embedding <=> $1::vector
        LIMIT $2
    """

def _build_filter_clause(self, filters: dict) -> tuple[str, list]:
    """Convert filter dict to PostgreSQL WHERE clause and params."""
    conditions: list[str] = []
    params: list = []

    if "source_pattern" in filters:
        params.append(f"%{filters['source_pattern']}%")
        conditions.append(f"d.source ILIKE ${len(params)}")

    if "created_after" in filters:
        params.append(filters["created_after"])
        conditions.append(f"c.created_at >= ${len(params)}")

    if "document_type" in filters:
        params.append(filters["document_type"])
        conditions.append(f"c.metadata->>'file_type' = ${len(params)}")

    if "title_contains" in filters:
        params.append(f"%{filters['title_contains']}%")
        conditions.append(f"d.title ILIKE ${len(params)}")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    return where, params
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
- `rag/storage/vector_store/postgres.py`
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

```sql
-- PostgreSQL chunks table with multiple embedding columns
ALTER TABLE chunks ADD COLUMN embedding_summary vector(1536);  -- OpenAI summary embedding
ALTER TABLE chunks ADD COLUMN embedding_hyde vector(768);      -- Hypothetical doc embedding
-- existing `embedding` column stays as primary (nomic 768-dim)
```

#### 8.3 Multi-Vector Search

```python
# rag/storage/vector_store/postgres.py
async def multi_vector_search(
    self,
    query_embedding: list[float],
    embedding_type: str = "primary",
    match_count: int = 10
) -> list[SearchResult]:
    """Search using a specific embedding type."""
    # Map embedding type to column name
    col = {
        "primary": "embedding",
        "summary": "embedding_summary",
        "hyde": "embedding_hyde",
    }.get(embedding_type, "embedding")
    rows = await self._pool.fetch(
        f"""SELECT c.*, d.title, d.source,
                   1 - (c.{col} <=> $1::vector) AS similarity
            FROM chunks c JOIN documents d ON c.document_id = d.id
            ORDER BY c.{col} <=> $1::vector
            LIMIT $2""",
        query_embedding, match_count
    )
    return [self._row_to_search_result(r) for r in rows]
```

## 10. Langfuse Tracing & Observability

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

## 11. Implementation Roadmap

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
2. Add `get_chunks_by_document()` to PostgreSQL store
3. Update `retrieve_as_context()` to include surrounding chunks
4. Add context size to settings

**Expected improvement**: Better answer quality

### Phase 4: Metadata Filtering (Low Impact, Low Effort)
1. Add `filters` parameter to search methods
2. Build filter clause in PostgreSQL
3. Update agent tool parameters

**Expected improvement**: Domain-specific retrieval

### Phase 5: Hierarchical Chunking (High Impact, High Effort)
1. Create `HierarchicalChunker`
2. Extend PostgreSQL schema
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
| Context Expansion | `retrieval/context_expanders.py` (new) | `postgres.py`, `retriever.py` |
| Parent-Child | `postgres.py`, `pipeline.py` | `chunkers/`, `retriever.py` |
| Metadata Filtering | `postgres.py` | `rag_agent.py` |
| Multi-Vector | `embedder.py`, `postgres.py` | `pipeline.py` |
| Knowledge Graph | `knowledge_graph/graphiti_store.py` (new) | `retriever.py`, `pipeline.py`, `rag_agent.py` |
| Langfuse Tracing | `observability/langfuse_integration.py` | `rag_agent.py`, `settings.py` |
| Streamlit Web UI | `agent/streamlit_app.py` | `rag_agent.py` |
| Mem0 Memory | `memory/mem0_store.py` (new) | `rag_agent.py`, `settings.py` |

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

## 12. Testing

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

# PostgreSQL connection & index tests (requires PostgreSQL/Neon)
python -m pytest rag/tests/test_postgres_store.py -v

# RAG agent integration tests (requires PostgreSQL + Ollama)
python -m pytest rag/tests/test_rag_agent.py -v
python -m pytest rag/tests/test_rag_agent.py -v --log-cli-level=INFO --tb=short # log.info
```

### Debug Agent Flow

python -m pytest rag/tests/test_agent_flow.py::TestAgentFlow::test_agent_flow_verbose -v -s --log-cli-level=INFO 2>&1 > sample_run.txt

#### Execution Flow Summary

```
test_agent_flow_verbose (test_agent_flow.py)
    |
    +--> set_verbose_debug(True)
    |
    +--> stream_agent_interaction(user_input, message_history, deps)
              |                                                  |
              |                            StateDeps[RAGState] --+
              |
              +--> _stream_agent() (agent_main.py:332)
                        |
                        +--> agent.iter(query, deps=deps, ...) --> yields nodes
                        |              |
                        |              +-- NOTE: deps passed but NOT USED by tools
                        |                  Tools create their own store/retriever
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

#### Note on `deps` (StateDeps) - Performance Optimization

`deps` shares a pre-initialized store/retriever across tool calls for better performance:

| Location | What Happens |
|----------|--------------|
| `test_agent_flow.py` | Creates `await RAGState.create()` with shared store/retriever |
| `stream_agent_interaction()` | Receives `deps`, passes to `_stream_agent()` |
| `_stream_agent()` | Passes `deps` to `agent.iter(..., deps=deps)` |
| `search_knowledge_base()` | Uses `ctx.deps.retriever` if available (no connection overhead) |

**Performance benefit:** Without shared deps, each tool call creates a new `PostgresHybridStore` connection. With shared deps, the connection is reused.

```python
# Good: Shared store (fast - connection reused)
state = await RAGState.create()  # Initialize once
deps = StateDeps(state)
# ... multiple tool calls reuse state.retriever ...
await state.close()  # Clean up once

# Without shared deps (slower - new connection per tool call)
state = RAGState()  # Empty state
deps = StateDeps(state)
# ... each tool call creates new PostgresHybridStore() ...
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
| `test_postgres_store.py` | PostgreSQL connection, vector/text indexes | PostgreSQL/Neon |
| `test_rag_agent.py` | Retriever queries, agent integration | PostgreSQL + Ollama |
| `test_agent_flow.py` | Agent flow execution, debug prints | PostgreSQL + Ollama |

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
- `test_postgres_store.py`: 18 tests pass
- `test_rag_agent.py`: All tests pass (requires indexes + Ollama running)

---

## 13. Performance Tuning

### Current Performance Profile

Run with profiling enabled to see timing breakdown:
```bash
python -m pytest rag/tests/test_agent_flow.py::TestAgentFlow::test_agent_flow_verbose -v -s --log-cli-level=INFO
```

**Typical timing breakdown (with local Ollama llama3.1:8b):**

| Phase | Time | Description |
|-------|------|-------------|
| ModelRequestNode (decide) | ~3-5s | LLM deciding to call search tool |
| CallToolsNode (search) | ~2-3s | PostgreSQL search + embedding generation |
| **ModelRequestNode (response)** | **~10-17s** | **LLM generating final response (BOTTLENECK)** |
| Total | ~15-25s | End-to-end query time |

### Bottleneck: Local LLM Response Generation

The main bottleneck is the local LLM generating responses. With `llama3.1:8b` on partial GPU:
```
ollama ps
NAME           SIZE      PROCESSOR
llama3.1:8b    6.3 GB    28%/72% CPU/GPU  <- Most inference on slow CPU!
```

### Performance Improvement Options

#### 1. Use a Smaller/Quantized Model (Recommended)

Smaller models fit entirely in GPU VRAM = 100% GPU inference = much faster.

```bash
# Pull smaller models
ollama pull llama3.2:3b          # ~2GB, fast
ollama pull qwen2.5:3b           # ~2GB, fast
ollama pull phi3:mini            # ~2GB, very fast
ollama pull mistral:7b-instruct-q4_0  # ~4GB, good quality

# Or quantized version of current model
ollama pull llama3.1:8b-instruct-q4_0  # ~4.5GB, fits better
```

Update `.env`:
```bash
LLM_MODEL=llama3.2:3b
```

**Expected improvement:** 3-5x faster response generation

#### 2. Ensure Full GPU Utilization

Check GPU usage:
```bash
ollama ps  # Should show ~100% GPU, 0% CPU for best performance
```

If model doesn't fit in VRAM:
```bash
# Restart Ollama to free memory
taskkill /IM ollama.exe /F   # Windows
# killall ollama             # Linux/Mac
ollama serve
```

Set GPU layers (if needed):
```bash
set OLLAMA_NUM_GPU=99  # Windows
# export OLLAMA_NUM_GPU=99  # Linux/Mac
ollama serve
```

#### 3. Reduce Search Context (Less for LLM to Process)

Fewer search results = smaller context = faster LLM response.

In `rag/agent/rag_agent.py`, change default `match_count`:
```python
async def search_knowledge_base(
    ctx: PydanticRunContext,
    query: str,
    match_count: int | None = 3,  # Reduced from 5
    search_type: str | None = "hybrid",
) -> str:
```

**Trade-off:** Fewer results may miss relevant information.

#### 4. Use Cloud LLM (Fastest, but costs money)

For production or when speed is critical:

```bash
# In .env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini        # Fast and cheap
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
```

**Expected improvement:** ~10x faster (sub-second responses)

#### 5. Lazy Initialization (Already Implemented)

The `RAGState` uses lazy initialization to avoid event loop issues and reuse connections:

```python
# Good: Connection reused across queries
state = RAGState()  # Empty, lazy-initialized
deps = StateDeps(state)
# First query: initializes store/retriever in correct event loop
# Subsequent queries: reuses the same connection
```

This avoids creating new PostgreSQL connections per query.

### Profiling Commands

```bash
# Full profiling output
python -m pytest rag/tests/test_agent_flow.py -v -s --log-cli-level=INFO

# Quick check - just timing lines
python -m pytest rag/tests/test_agent_flow.py -v -s --log-cli-level=INFO 2>&1 | grep PROFILE

# Save full output to file
python -m pytest rag/tests/test_agent_flow.py -v -s --log-cli-level=INFO > profile_output.txt 2>&1
```

### Performance Improvement Checklist

- [ ] Check `ollama ps` - is model using 100% GPU?
- [ ] Try smaller model: `llama3.2:3b` or `qwen2.5:3b`
- [ ] Try quantized model: `llama3.1:8b-instruct-q4_0`
- [ ] Reduce `match_count` from 5 to 3
- [ ] Consider cloud LLM for production use
- [ ] Run profiling to verify improvements

---

## 14. Caching

The RAG system implements two levels of caching to improve response times for repeated queries:

### Cache Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Query                              │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Result Cache (Retriever)                  │
│  Key: (query, search_type, match_count)                     │
│  TTL: 5 minutes                                             │
│  Max size: 100 entries                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  CACHE HIT  → Return cached SearchResult list       │    │
│  │  CACHE MISS → Continue to embedding...              │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────┬───────────────────────────────┘
                              │ (miss)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Embedding Cache (EmbeddingGenerator)         │
│  Key: (query_text, model_name)                              │
│  No TTL (embeddings are deterministic)                      │
│  Max size: 1000 entries                                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  CACHE HIT  → Return cached embedding vector        │    │
│  │  CACHE MISS → Call embedding API (~500-700ms)       │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   PostgreSQL Search                          │
│  Uses embedding to find similar documents                   │
│  Results cached in Result Cache for next time               │
└─────────────────────────────────────────────────────────────┘
```

### Cache Types

| Cache | Location | Key | TTL | Max Size | Purpose |
|-------|----------|-----|-----|----------|---------|
| **Embedding Cache** | `rag/ingestion/embedder.py` | `(text, model)` | None | 1000 | Cache query embeddings |
| **Result Cache** | `rag/retrieval/retriever.py` | `(query, type, count)` | 5 min | 100 | Cache search results |

### Performance Impact

| Query Type | Cold Cache | Warm Cache | Improvement |
|------------|------------|------------|-------------|
| Embedding generation | ~600-800ms | ~0ms | **100%** |
| Full search (same params) | ~700-1000ms | ~0ms | **100%** |
| Full search (diff params) | ~700-1000ms | ~100-200ms | **80%** (embedding cached) |

### Usage

Caching is **enabled by default**. No configuration needed.

#### Check Cache Statistics

```python
from rag.ingestion.embedder import EmbeddingGenerator
from rag.retrieval.retriever import Retriever

# Get embedding cache stats
print(EmbeddingGenerator.get_cache_stats())
# {'size': 5, 'max_size': 1000, 'hits': 3, 'misses': 2, 'hit_rate': '60.0%'}

# Get result cache stats
print(Retriever.get_cache_stats())
# {'size': 3, 'max_size': 100, 'ttl_seconds': 300, 'hits': 2, 'misses': 1, 'hit_rate': '66.7%'}
```

#### Clear Caches

```python
from rag.ingestion.embedder import EmbeddingGenerator
from rag.retrieval.retriever import Retriever

# Clear embedding cache
EmbeddingGenerator.clear_cache()

# Clear result cache
Retriever.clear_cache()
```

#### Disable Caching for Specific Queries

```python
from rag.retrieval.retriever import Retriever

retriever = Retriever(store=store)

# Bypass cache for this specific query
results = await retriever.retrieve(
    query="What is RAG?",
    use_cache=False  # Skip result cache
)
```

### Implementation Details

#### Embedding Cache (`embedder.py`)

Uses `@alru_cache` from `async-lru` for simple async caching:

```python
from async_lru import alru_cache

@alru_cache(maxsize=1000)
async def _cached_embed(text: str, model: str) -> tuple[float, ...]:
    """Cached embedding generation."""
    client = _get_client()
    response = await client.embeddings.create(model=model, input=text)
    return tuple(response.data[0].embedding)
```

The `@alru_cache` decorator handles LRU eviction automatically. Stats available via `_cached_embed.cache_info()`.

#### ResultCache Class (`retriever.py`)

```python
class ResultCache:
    """LRU cache for search results with TTL."""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self._cache: OrderedDict[str, tuple[float, list[SearchResult]]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds

    def get(self, query: str, search_type: str, match_count: int) -> list[SearchResult] | None:
        # Check TTL before returning
        ...

    def set(self, query: str, search_type: str, match_count: int, results: list[SearchResult]) -> None:
        # Evict oldest if over limit
        ...
```

### Cache Design Decisions

1. **Embedding cache uses `@alru_cache`**: Simple async LRU cache from `async-lru` package
2. **Result cache is custom**: Needs TTL support (not available in `@alru_cache`)
3. **No TTL for embeddings**: Embeddings are deterministic - same text always produces same embedding
4. **5-minute TTL for results**: Balances freshness with performance (data may change)
5. **Global caches**: Shared across all instances for maximum reuse

### When to Clear Caches

- **After re-ingesting documents**: Result cache may have stale results
- **After changing embedding model**: Embedding cache will have wrong vectors
- **For debugging**: To ensure fresh queries during development

```python
# Clear both caches after re-ingestion
from rag.ingestion.embedder import EmbeddingGenerator
from rag.retrieval.retriever import Retriever

EmbeddingGenerator.clear_cache()
Retriever.clear_cache()
```

---

## 8. Test Queries by Document Type

This section provides sample queries for testing the RAG system against different document types in the `rag/documents/` folder. These queries are designed to validate that ingestion and retrieval work correctly for each file format.

### Document Inventory

| File Type | Files | Content Focus |
|-----------|-------|---------------|
| **Markdown** | company-overview.md, team-handbook.md, mission-and-goals.md, implementation-playbook.md | Company info, employee policies, goals, best practices |
| **PDF** | client-review-globalfinance.pdf, q4-2024-business-review.pdf, technical-architecture-guide.pdf | Client case studies, financials, technical architecture |
| **DOCX** | meeting-notes-2025-01-08.docx, meeting-notes-2025-01-15.docx | Meeting discussions and decisions |
| **Audio** | Recording1.mp3, Recording2.mp3, Recording3.mp3, Recording4.mp3 | Transcribed meeting/discussion content |

### Markdown File Queries (company-overview.md, team-handbook.md, mission-and-goals.md)

| # | Query | Expected Source | Expected Answer Contains |
|---|-------|-----------------|-------------------------|
| 1 | What does NeuralFlow AI do? | company-overview.md | AI/ML solutions, enterprise, automation |
| 2 | How many employees work at the company? | company-overview.md | 47 employees |
| 3 | Where is NeuralFlow AI headquartered? | company-overview.md | San Francisco |
| 4 | What is the PTO policy? | team-handbook.md | Unlimited PTO, 15-day minimum |
| 5 | What is the learning budget for employees? | team-handbook.md | $2,500 per year |
| 6 | What are the company's core products? | company-overview.md | DocFlow AI, ConversePro, AnalyticsMind |

### PDF File Queries (client-review-globalfinance.pdf, q4-2024-business-review.pdf, technical-architecture-guide.pdf)

| # | Query | Expected Source | Expected Answer Contains |
|---|-------|-----------------|-------------------------|
| 7 | How much did GlobalFinance save by implementing NeuralFlow? | client-review-globalfinance.pdf | $2.4 million savings |
| 8 | What was the processing time reduction for GlobalFinance? | client-review-globalfinance.pdf | 94% reduction |
| 9 | What was Q4 2024 revenue? | q4-2024-business-review.pdf | $2.8 million |
| 10 | What was the quarter-over-quarter growth rate? | q4-2024-business-review.pdf | 47% QoQ growth |
| 11 | What is the 2025 ARR target? | mission-and-goals.md / q4-2024 | $12 million ARR |
| 12 | What database is used for vector storage? | technical-architecture-guide.pdf | PostgreSQL with pgvector |
| 13 | Describe the technical architecture | technical-architecture-guide.pdf | Microservices, RAG system, APIs |

### DOCX File Queries (meeting-notes-*.docx)

| # | Query | Expected Source | Expected Answer Contains |
|---|-------|-----------------|-------------------------|
| 14 | What was discussed in the January 8th meeting? | meeting-notes-2025-01-08.docx | Meeting topics, decisions |
| 15 | What decisions were made in the January 15th meeting? | meeting-notes-2025-01-15.docx | Action items, decisions |
| 16 | What are the recent meeting action items? | meeting-notes-*.docx | Tasks, assignments, deadlines |

### Audio File Queries (Recording*.mp3)

> **Note**: These queries require FFmpeg and `openai-whisper` to be installed. See [Audio Transcription Prerequisites](#supported-file-formats) for setup instructions. Without these dependencies, audio files are stored with error placeholders.

| # | Query | Expected Source | Expected Answer Contains |
|---|-------|-----------------|-------------------------|
| 17 | What was discussed in the audio recordings? | Recording*.mp3 | Transcribed discussion topics |
| 18 | Are there any action items from the recordings? | Recording*.mp3 | Tasks mentioned in transcription |

### Cross-Document Queries (Tests Hybrid Search)

| # | Query | Expected Sources | Purpose |
|---|-------|------------------|---------|
| 19 | What technologies and tools does the company use? | technical-architecture-guide.pdf, company-overview.md | Multi-source retrieval |
| 20 | Summarize all employee benefits | team-handbook.md, possibly others | Comprehensive policy query |

### Running Test Queries

#### Programmatic Testing

```python
import asyncio
from rag.retrieval.retriever import Retriever
from rag.storage.vector_store.postgres import PostgresHybridStore

TEST_QUERIES = [
    "What does NeuralFlow AI do?",
    "How many employees work at the company?",
    "What is the PTO policy?",
    "What is the learning budget for employees?",
    "How much did GlobalFinance save?",
    "What was Q4 2024 revenue?",
    "What is the 2025 ARR target?",
    "What was discussed in the January meetings?",
    "What technologies does the company use?",
]

async def run_test_queries():
    store = PostgresHybridStore()
    retriever = Retriever(store=store)

    for query in TEST_QUERIES:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)

        results = await retriever.retrieve(query, match_count=3)

        for i, r in enumerate(results, 1):
            print(f"\n[{i}] {r.document_title} (score: {r.similarity:.3f})")
            print(f"    {r.content[:150]}...")

    await store.close()

asyncio.run(run_test_queries())
```

#### Using pytest

```bash
# Run the RAG agent integration tests
python -m pytest rag/tests/test_rag_agent.py -v

# With logging to see retrieved content
python -m pytest rag/tests/test_rag_agent.py -v --log-cli-level=INFO
```

### Expected Behavior

1. **Markdown queries**: Should return high-relevance matches from .md files
2. **PDF queries**: Should extract and match content from PDF documents
3. **DOCX queries**: Should find content from Word documents
4. **Audio queries**: Should match transcribed content from Whisper ASR output
5. **Cross-document queries**: Should return results from multiple sources, demonstrating hybrid search

### Debugging Failed Queries

If a query doesn't return expected results:

1. **Verify ingestion**: Run `python -m rag.main --ingest --documents rag/documents --verbose`
2. **Check document count**: Query PostgreSQL to verify chunk count
3. **Test search types separately**:
   ```python
   # Try semantic-only search
   results = await retriever.retrieve(query, search_type="semantic")

   # Try text-only search
   results = await retriever.retrieve(query, search_type="text")

   # Compare with hybrid (default)
   results = await retriever.retrieve(query, search_type="hybrid")
   ```
4. **Inspect embeddings**: Ensure embedding dimension matches index configuration (768 for nomic-embed-text)

---

## 15. Mem0 Memory Layer

Mem0 is a **persistent memory layer** for AI applications that remembers user-specific context across sessions. While RAG retrieves from a document knowledge base, Mem0 remembers things about **users and conversations**.

### Why Mem0 + RAG?

| Feature | RAG Alone | RAG + Mem0 |
|---------|-----------|------------|
| **Knowledge Source** | Static documents | Documents + user memories |
| **Personalization** | Same for all users | Adapts to each user |
| **Context** | Retrieved docs only | Docs + user preferences/history |
| **Session Continuity** | Stateless | Remembers across sessions |

### How It Helps

```
┌─────────────────────────────────────────────────────────────┐
│                      User Query                              │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌───────────────────┐
│    Mem0       │   │      RAG        │   │    LLM Agent      │
│  Recall user  │   │  Retrieve docs  │   │  Generate answer  │
│  preferences  │   │  from knowledge │   │  using both       │
│  & context    │   │  base           │   │  contexts         │
└───────────────┘   └─────────────────┘   └───────────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                   ┌─────────────────────┐
                   │   Store new facts   │
                   │   learned about     │
                   │   user in Mem0      │
                   └─────────────────────┘
```

### Practical Examples

**Example 1: User Context**
```
User: "I'm a senior engineer in the ML team"
→ Mem0 stores: {user_id: "john", fact: "senior engineer, ML team"}

Later...
User: "What training budget do I get?"
→ Mem0 recalls: "senior engineer, ML team"
→ RAG retrieves: General training policy
→ Agent: Combines both for personalized answer about senior-level budget
```

**Example 2: Conversation Continuity**
```
Session 1:
  User: "What's the PTO policy?"
  Agent: "Unlimited PTO with 15-day minimum..."
  → Mem0 stores: "User asked about PTO policy"

Session 2 (days later):
  User: "What else should I know about benefits?"
  → Mem0 recalls: "Previously discussed PTO policy"
  → Agent: Focuses on OTHER benefits, avoiding repetition
```

**Example 3: Preferences**
```
User: "Give me shorter answers please"
→ Mem0 stores: {user_id: "john", preference: "concise responses"}

All future queries:
→ Mem0 recalls: "prefers concise responses"
→ Agent: Adjusts response length automatically
```

### Key Benefits

| Benefit | Description |
|---------|-------------|
| **91% Faster** | Retrieves only relevant memories vs. full history |
| **90% Fewer Tokens** | No need to send entire conversation history |
| **Personalization** | Improves with each interaction |
| **Multi-Level Memory** | User, Session, and Agent-level retention |
| **Graph Memory** | Connects entities (people, places, events) |

### Architecture with Current RAG System

```
┌─────────────────────────────────────────────────────────────┐
│                     rag/memory/                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  mem0_store.py                                       │    │
│  │  ├── Mem0Store                                       │    │
│  │  │   ├── add_memory(text, user_id, metadata)        │    │
│  │  │   ├── search_memories(query, user_id, limit)     │    │
│  │  │   ├── get_all_memories(user_id)                  │    │
│  │  │   └── delete_memory(memory_id)                   │    │
│  │  └── create_mem0_store() -> Mem0Store               │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   rag/agent/rag_agent.py                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  RAGState                                            │    │
│  │  ├── _store: PostgresHybridStore                       │    │
│  │  ├── _retriever: Retriever                          │    │
│  │  └── _mem0: Mem0Store  ← NEW                        │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  @agent.tool                                                │
│  async def search_knowledge_base(...)                       │
│      # 1. Recall user memories from Mem0                    │
│      # 2. Retrieve documents from RAG                       │
│      # 3. Combine contexts for LLM                          │
│                                                             │
│  @agent.tool                                                │
│  async def remember_user_context(...)  ← NEW                │
│      # Store facts about user in Mem0                       │
└─────────────────────────────────────────────────────────────┘
```

### Installation

```bash
pip install mem0ai
```

### PostgreSQL Setup

Mem0 uses **PostgreSQL/pgvector** as its vector store (same database as RAG). The `mem0_memories` table is created automatically on first use.

#### Step 1: Enable Mem0 in `.env`

```bash
# Mem0 Configuration
MEM0_ENABLED=true

# Uses existing PostgreSQL connection (DATABASE_URL)
# Uses existing Ollama models (LLM_MODEL, EMBEDDING_MODEL)
```

#### Step 2: Verify Setup

```bash
# Test Mem0 standalone
python -m rag.memory.mem0_store
```

### Configuration Reference

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| `mem0_enabled` | `MEM0_ENABLED` | `false` | Enable/disable Mem0 |
| `mem0_collection_name` | `MEM0_COLLECTION_NAME` | `mem0_memories` | Table name for memories |

Mem0 automatically uses these existing settings:
- `DATABASE_URL` - PostgreSQL connection string
- `LLM_MODEL` - For fact extraction (default: `llama3.1:8b`)
- `EMBEDDING_MODEL` - For memory vectors (default: `nomic-embed-text:latest`)

### Database Architecture

```
PostgreSQL (neondb)
├── documents        ← RAG source documents
├── chunks           ← RAG chunks with embeddings
└── mem0_memories    ← Mem0 user memories with embeddings
```

### Basic Usage

```python
from rag.memory import Mem0Store, create_mem0_store

# Create store (uses settings from .env)
mem0 = create_mem0_store()

# Store a memory (LLM extracts facts automatically)
mem0.add(
    "User prefers concise answers and works in engineering",
    user_id="john_doe",
    metadata={"source": "user_preference"}
)

# Store raw memory (no LLM processing)
mem0.add(
    "User is on the ML team",
    user_id="john_doe",
    infer=False  # Skip LLM fact extraction
)

# Search memories
results = mem0.search(
    "communication preferences",
    user_id="john_doe",
    limit=5
)

# Get formatted context for LLM
context = mem0.get_context_string(
    "What benefits do I get?",
    user_id="john_doe"
)

# Get all memories for user
all_memories = mem0.get_all(user_id="john_doe")

# Delete all user memories
mem0.delete_all(user_id="john_doe")
```

### Integration with RAG Agent

Mem0 is automatically integrated into the RAG agent. To enable personalization:

```python
from rag.agent.rag_agent import agent, RAGState
from rag.memory import create_mem0_store

# 1. Add user memories (one-time setup or during conversation)
mem0 = create_mem0_store()
mem0.add("I'm a senior engineer on the ML team", user_id="john")
mem0.add("I prefer brief, technical answers", user_id="john")

# 2. Query with user_id for personalized responses
state = RAGState(user_id="john")
result = await agent.run("What training budget do I get?", deps=state)
await state.close()

# The agent will:
# - Recall: "senior engineer, ML team" + "prefers brief answers"
# - Retrieve: Training policy from documents
# - Generate: Personalized response for senior engineers
```

### How It Works (Implementation)

The `search_knowledge_base` tool automatically combines Mem0 and RAG:

```python
@agent.tool
async def search_knowledge_base(ctx, query, match_count=5, search_type="hybrid"):
    # Get user_id from RAGState
    user_id = state.user_id if state else None

    # 1. Get RAG results from PostgreSQL
    rag_result = await retriever.retrieve_as_context(query, match_count, search_type)

    # 2. Get Mem0 user context (if enabled and user_id provided)
    user_context = ""
    if mem0.is_enabled() and user_id:
        user_context = mem0.get_context_string(query, user_id, limit=3)

    # 3. Combine contexts for LLM
    if user_context:
        return f"{user_context}\n\n{rag_result}"
    return rag_result
```

### Memory Types

| Type | Scope | Use Case | Example |
|------|-------|----------|---------|
| **User Memory** | Persists across all sessions | Preferences, role, department | "User is senior engineer" |
| **Session Memory** | Single conversation | Current topic context | "Discussing Q4 financials" |
| **Agent Memory** | Shared across users | Learned facts | "Company has 47 employees" |

### Graph Memory (Advanced)

Mem0's Graph Memory connects entities and relationships:

```python
# Configure with graph memory
config = {
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password"
        }
    },
    # ... llm and embedder config ...
}

# Memories now include entity relationships
memory.add(
    "John works with Sarah on the ML team. They report to Mike.",
    user_id="system"
)

# Query understands relationships
memory.search("Who does John work with?")
# Returns: Sarah, Mike, ML team connections
```

### Performance Comparison

| Metric | Without Mem0 | With Mem0 |
|--------|--------------|-----------|
| Context tokens per query | ~4000 (full history) | ~500 (relevant only) |
| Response latency | Increases with history | Constant (~100ms extra) |
| Personalization | None | Improves over time |
| Storage | Conversation in session | Persistent memories |

### When to Use Mem0

**Good Fit:**
- Multi-session applications (users return over time)
- Personalized assistants (adapt to user preferences)
- Role-based access (remember user's department/role)
- Conversation continuity (avoid repetition)

**Not Needed:**
- Single-query applications
- Anonymous users
- Stateless API endpoints
- When all context fits in prompt

### Files to Modify for Integration

| File | Changes |
|------|---------|
| `rag/memory/mem0_store.py` | New - Mem0 wrapper class |
| `rag/config/settings.py` | Add mem0_enabled setting |
| `rag/agent/rag_agent.py` | Integrate memory into RAGState and tools |
| `requirements.txt` | Add mem0ai dependency |

### References

- [Mem0 Documentation](https://docs.mem0.ai/)
- [Mem0 GitHub](https://github.com/mem0ai/mem0)
- [Mem0 with Ollama Guide](https://docs.mem0.ai/open-source/llms/ollama)



### Do we still need query cache?

We still need the query cache - they serve different purposes:

| Cache              | Purpose                                                 |
|--------------------|---------------------------------------------------------|
| Query/Result Cache | Avoids repeated PostgreSQL searches for identical queries |
| Embedding Cache    | Avoids repeated embedding API calls                     |
| Mem0               | Stores user-specific memories/preferences (not a cache) |

However, with Mem0 the result cache key should include user_id since the same query may return different context for different users.

---

## 16. RAG-Anything Modal Processors

### Overview

[RAG-Anything](https://github.com/HKUDS/RAG-Anything) is an all-in-one multimodal RAG framework built on LightRAG. It provides specialized modal processors for handling images, tables, and equations in documents.

### Installation

```bash
# Basic installation
pip install raganything

# With all optional features
pip install 'raganything[all]'
```

### Installation Troubleshooting

#### Python 3.13 + numpy Issue

On Python 3.13, `pip install 'raganything[all]'` may fail with:

```
error: metadata-generation-failed
numpy<2.0.0,>=1.24.0 ... Unknown compiler(s): [['icl'], ['cl'], ['cc'], ['gcc']]
```

**Cause**: `lightrag-hku` requires `numpy<2.0.0`, but numpy 1.x has no pre-built wheels for Python 3.13 and requires Visual Studio Build Tools to compile from source.

**Solution**: Install with `--no-deps` to skip strict version constraints:

```bash
# Install numpy 2.x (has pre-built wheels)
pip install numpy>=2.0.0

# Install raganything and lightrag without dependency resolution
pip install raganything --no-deps
pip install lightrag-hku --no-deps

# Verify installation
python -c "from raganything import RAGAnything; print('OK')"
python -c "from raganything.modalprocessors import ImageModalProcessor; print('OK')"
```

#### Missing pypinyin Warning

```
WARNING: pypinyin is not installed. Chinese pinyin sorting will use simple string sorting.
```

This is harmless unless you need Chinese text sorting. To fix:

```bash
pip install pypinyin
```

### Modal Processors

RAG-Anything provides three specialized modal processors:

| Processor | Purpose | Input Fields |
|-----------|---------|--------------|
| `ImageModalProcessor` | Process images with vision models | `img_path`, `image_caption`, `image_footnote` |
| `TableModalProcessor` | Extract structured info from tables | `table_body`, `table_caption`, `table_footnote` |
| `EquationModalProcessor` | Process mathematical equations | `text` (LaTeX), `text_format` |

#### ImageModalProcessor

```python
from raganything.modalprocessors import ImageModalProcessor

processor = ImageModalProcessor(
    lightrag=lightrag_instance,
    modal_caption_func=vision_model_func
)

image_content = {
    "img_path": "/absolute/path/to/image.jpg",
    "image_caption": ["Figure 1: System Architecture"],
    "image_footnote": ["Source: Original design"]
}

result = await processor.process_multimodal_content(
    modal_content=image_content,
    content_type="image",
    file_path="research_paper.pdf",
    entity_name="Architecture Diagram"
)
```

#### TableModalProcessor

```python
from raganything.modalprocessors import TableModalProcessor

processor = TableModalProcessor(
    lightrag=lightrag_instance,
    modal_caption_func=llm_model_func
)

table_content = {
    "table_body": """
    | Method | Accuracy | F1-Score |
    |--------|----------|----------|
    | Ours   | 95.2%    | 0.94     |
    | Base   | 87.3%    | 0.85     |
    """,
    "table_caption": ["Performance Comparison"],
    "table_footnote": ["Results on test dataset"]
}

result = await processor.process_multimodal_content(
    modal_content=table_content,
    content_type="table",
    file_path="research_paper.pdf",
    entity_name="Performance Results"
)
```

#### EquationModalProcessor

```python
from raganything.modalprocessors import EquationModalProcessor

processor = EquationModalProcessor(
    lightrag=lightrag_instance,
    modal_caption_func=llm_model_func
)

equation_content = {
    "text": r"L(\theta) = -\frac{1}{N}\sum_{i=1}^{N}[y_i \log(p_i) + (1-y_i)\log(1-p_i)]",
    "text_format": "LaTeX"
}

result = await processor.process_multimodal_content(
    modal_content=equation_content,
    content_type="equation",
    file_path="ml_paper.pdf",
    entity_name="Binary Cross-Entropy Loss"
)
```

### Testing Modal Processors

A test script is provided at `rag/ingestion/processors/test_raganything.py` to verify the modal processors work correctly.

#### Running Tests

```bash
# Test with Ollama (default, no API key needed)
python -m rag.ingestion.processors.test_raganything

# Test with Ollama explicitly
python -m rag.ingestion.processors.test_raganything --use-ollama

# Test with OpenAI
python -m rag.ingestion.processors.test_raganything --api-key YOUR_OPENAI_KEY

# Test with custom base URL
python -m rag.ingestion.processors.test_raganything --api-key KEY --base-url URL
```

#### What Gets Tested

| Processor | Test Content | Description |
|-----------|--------------|-------------|
| `TableModalProcessor` | LLM benchmark table | Processes markdown table with performance metrics |
| `EquationModalProcessor` | Cross-entropy loss formula | Processes LaTeX equation |
| `ImageModalProcessor` | Caption-only test | Processes image metadata without actual image file |

#### Expected Output

```
Testing TableModalProcessor
Description: {'table_body': '| Model | Accuracy |...
Entity Info: {'entity_name': 'LLM Performance Table', 'entity_type': 'table', ...}
TableModalProcessor: SUCCESS

Testing EquationModalProcessor
Description: {'text': 'L(\\theta) = -\\frac{1}{N}...
Entity Info: {'entity_name': 'Binary Cross-Entropy Loss', 'entity_type': 'equation', ...}
EquationModalProcessor: SUCCESS

Testing ImageModalProcessor
Description: {'img_path': '', 'image_caption': [...
Entity Info: {'entity_name': 'RAG Architecture Diagram', 'entity_type': 'image', ...}
ImageModalProcessor: SUCCESS

TEST SUMMARY
  TableProcessor: PASS
  EquationProcessor: PASS
  ImageProcessor: PASS

Total: 3/3 passed
```

### Integration with Our Processors

Our framework has equivalent implementations that can be used alongside or instead of RAG-Anything's processors:

| RAG-Anything API | Our Equivalent | Location |
|------------------|----------------|----------|
| `ImageModalProcessor` | `ImageProcessor` | `rag/ingestion/processors/image.py` |
| `TableModalProcessor` | `TableProcessor` | `rag/ingestion/processors/table.py` |
| `EquationModalProcessor` | `EquationProcessor` | `rag/ingestion/processors/equation.py` |

#### Testing Our Processors

```bash
# Test TableProcessor
python -m rag.ingestion.processors.table

# Test EquationProcessor
python -m rag.ingestion.processors.equation

# Test EquationProcessor with custom equation
python -m rag.ingestion.processors.equation "F = ma" "Newton's second law"
```

### References

- [RAG-Anything GitHub](https://github.com/HKUDS/RAG-Anything)
- [RAG-Anything Paper](https://arxiv.org/abs/2510.12323)
- [LightRAG GitHub](https://github.com/HKUDS/LightRAG)
- [Full RAG-Anything API Reference](../RAG_anything.md)

---

## Misc

### Knowledge Graph RAG with Graphiti

> **Note**: This section documents an exploratory Graphiti-based approach. The production knowledge graph implementation uses `PgGraphStore` (PostgreSQL tables `kg_entities` / `kg_relationships`) and `AgeGraphStore` (Apache AGE / Cypher). See `docs/ARCHITECTURE_SUMMARY.md §7` for the live implementation. `rag/knowledge_graph/graphiti_store.py` is retained as an alternative backend.

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
            │ (PostgreSQL)  │                     │  RAG (Graphiti)   │
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
        store: PostgresHybridStore | None = None,
        embedder: EmbeddingGenerator | None = None,
        graph_store: GraphitiStore | None = None,  # Add this
    ):
        self.store = store or PostgresHybridStore()
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

