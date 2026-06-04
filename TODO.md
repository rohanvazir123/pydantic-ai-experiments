# TODO

## In progress

### Rate limiting, timeouts & retries
Settings fields added to `rag/config/settings.py`. Implementation pending per step below.

- [ ] **Step 1 — Embedding timeouts + retries** (`rag/ingestion/embedder.py`)
  - Pass `timeout=openai.Timeout(connect=5, read=embedding_timeout_s)` to `AsyncOpenAI`
  - Add exponential-backoff retry on `RateLimitError`, `APIConnectionError`, `APITimeoutError`
  - Settings: `embedding_timeout_s`, `embedding_retry_attempts`, `embedding_retry_backoff_s`

- [ ] **Step 2 — DB query timeouts** (`rag/storage/vector_store/postgres.py`)
  - Pass `timeout=db_query_timeout_s` to every `conn.fetch()` / `conn.fetchrow()` / `conn.execute()`
  - Catch `asyncpg.exceptions.QueryCanceledError` specifically (not bare `Exception`)
  - Settings: `db_query_timeout_s`, `db_health_timeout_s` (replaces hardcoded 5.0 in `api/app.py`)

- [ ] **Step 3 — LLM call timeout** (`rag/agent/rag_agent.py`, `rag/api/app.py`)
  - Wrap `traced_agent_run` in `asyncio.wait_for(..., timeout=llm_timeout_s)`
  - Return `504 Gateway Timeout` on deadline exceeded (not 500)
  - Settings: `llm_timeout_s`

- [ ] **Step 4 — Inbound API rate limiting** (`rag/api/app.py`)
  - Add `slowapi` middleware; rate-limit `/v1/chat` + `/v1/chat/stream` by IP
  - Return `429 Too Many Requests` with `Retry-After` header
  - Settings: `api_rate_limit_rpm`, `api_rate_limit_burst`

---

## Queued (from session 2026-06-04)

### Enterprise refactoring — `rag/` folder
Full audit complete. Phased plan in conversation. Deferred to after rate-limiting work.

- [ ] **Phase 1 — Production hardening**
  - [ ] Replace bare `except Exception` with specific types across all files
  - [ ] Fix `threading.Lock` in async code (`embedder.py`)
  - [ ] Add `__aenter__`/`__aexit__` to `DocumentIngestionPipeline`
  - [ ] Startup connectivity validation (`settings.py`)

- [ ] **Phase 2 — Code quality**
  - [ ] Complete type annotations (`pipeline.py`, `embedder.py`, `rerankers.py`)
  - [ ] Extract magic numbers to settings (`lists=100`, `k=60`, cache TTL)
  - [ ] Split `DocumentIngestionPipeline` (700 lines → 3 classes)
  - [ ] Delete `retrieval/dead_code/` directory

- [ ] **Phase 3 — Observability**
  - [ ] Structured logging with `extra={}` fields
  - [ ] Correlation IDs propagated from Langfuse trace
  - [ ] Pool utilisation metrics in health endpoint
  - [ ] Downgrade cache hit/miss logs to DEBUG

### Additional features (from session 2026-06-04)
- [ ] Update `rag/docs/DATASTORE_GUIDE.md` with SQL examples and index notes
- [ ] Add `PROMPTS.md` documenting system prompt design and variables
- [ ] Input safety validation — NSFW / prompt injection guard on `/v1/chat`
- [ ] JWT client authentication on FastAPI endpoints
- [ ] Replace Streamlit UI with Next.js + message-queue + async-worker model
- [ ] Human-in-the-loop audit hooks
- [ ] Supabase support — `PostgresHybridStore` is already compatible (standard asyncpg connection string); add setup instructions to `DATASTORE_GUIDE.md` §3.4 and verify `pgvector`/`pg_trgm` pre-enabled on project creation

---

## Done

- [x] Metadata filtering during retrieval — `MetadataFilter` model, all search legs, cache key, agent tool (`2026-06-04`)
- [x] Settings fields for rate-limiting/timeouts/retries — `rag/config/settings.py` (`2026-06-04`)
