# RAG System Memory Design

> Comprehensive reference for designing memory in a production RAG system. Covers cognitive memory types, four implementation tiers, pruning/eviction/compaction, framework assessment, and GDPR compliance.

---

## Table of Contents

- [tsvector + pgvector: the Search Pattern Used Throughout](#tsvector--pgvector-the-search-pattern-used-throughout)
  - [Where tsvector + pgvector applies in memory tiers](#where-tsvector--pgvector-applies-in-memory-tiers)
- [Cognitive Memory Types — The Framework](#cognitive-memory-types--the-framework)
- [Mapping Cognitive Types to Implementation Tiers](#mapping-cognitive-types-to-implementation-tiers)
  - [Context assembly — how all tiers feed into every request](#context-assembly--how-all-tiers-feed-into-every-request)
- [Tier 1 — Short-Term / Working Memory](#tier-1--short-term--working-memory-per-request)
- [Tier 2 — Episodic Memory](#tier-2--episodic-memory-per-session-server-side)
- [Tier 3 — Semantic Memory (User)](#tier-3--semantic-memory-user-cross-session)
- [Tier 4 — Semantic Memory (Knowledge Corpus)](#tier-4--semantic-memory-knowledge-corpus-shared)
- [Tier 5 — Procedural Memory](#tier-5--procedural-memory-system-level)
- [How the Tiers Interact Per Request](#how-the-tiers-interact-per-request)
- [Token Budget Management](#token-budget-management)
- [Mem0 Write Path — When Memories Are Created](#mem0-write-path--when-memories-are-created)
- [Mem0 Read Path — When Memories Are Injected](#mem0-read-path--when-memories-are-injected)
- [Conversation Auto-Summarization](#conversation-auto-summarization)
- [Memory Pruning, Eviction, and Compaction](#memory-pruning-eviction-and-compaction)
- [Framework Assessment — Build vs Buy](#framework-assessment--build-vs-buy)
- [Database Schema](#database-schema)
- [API Endpoints](#api-endpoints)
- [Frontend Memory Management](#frontend-memory-management)
- [GDPR and Right to Erasure](#gdpr-and-right-to-erasure)
- [Quick Reference](#quick-reference)

---

## tsvector + pgvector: the Search Pattern Used Throughout

Before diving into memory tiers, understand the search pattern the system uses everywhere. It is important that memory tiers follow the same pattern for consistency and performance.

### The pattern: hybrid BM25 + cosine with RRF

The system already uses this pattern in two places:
- **`chunks.content_tsv`** — full-text search on document chunks (Tier 4)
- **`kg_entity_index.name_tsv`** — entity name search for the AGE knowledge graph (Tier 4)

Both use the same structure: a `tsvector` column (GIN-indexed for BM25) alongside a `vector(768)` column (HNSW-indexed for cosine ANN), combined at query time with Reciprocal Rank Fusion (RRF, k=60).

```sql
-- The template applied to every searchable memory column
column_tsv  tsvector GENERATED ALWAYS AS (to_tsvector('english', column_text)) STORED
embedding   vector(768)

CREATE INDEX ON table USING GIN (column_tsv);
CREATE INDEX ON table USING hnsw (embedding vector_cosine_ops);
```

```sql
-- Hybrid search query (RRF k=60) — identical structure used for chunks, entities, and memories
WITH
text_ranked AS (
    SELECT id,
           ROW_NUMBER() OVER (ORDER BY ts_rank(column_tsv, websearch_to_tsquery('english', $1)) DESC) AS rn
    FROM table
    WHERE column_tsv @@ websearch_to_tsquery('english', $1)
),
vec_ranked AS (
    SELECT id,
           ROW_NUMBER() OVER (ORDER BY embedding <=> $2::vector ASC) AS rn
    FROM table
    WHERE embedding IS NOT NULL
    LIMIT 60
),
rrf AS (
    SELECT COALESCE(t.id, v.id) AS id,
           (COALESCE(1.0 / (60.0 + t.rn), 0) + COALESCE(1.0 / (60.0 + v.rn), 0)) AS score
    FROM text_ranked t
    FULL OUTER JOIN vec_ranked v ON t.id = v.id
)
SELECT r.id, e.content, r.score
FROM rrf r JOIN table e ON e.id = r.id
ORDER BY r.score DESC LIMIT $3;
```

**Why both?** Each leg captures different signals:

| Signal | When it wins | When it loses |
|--------|-------------|---------------|
| **tsvector / BM25** | Exact keyword match ("GDPR Article 17"), proper nouns, technical terms | Paraphrases, synonyms, conceptual similarity |
| **pgvector / cosine** | Semantic similarity, paraphrases ("data deletion rights" ≈ "right to erasure") | Rare proper nouns, exact IDs, short queries |
| **RRF combined** | Always at least as good as the better of the two legs | Slightly slower than either alone |

The fallback: if `tsvector` matches nothing (e.g., a query of pure stopwords), the search degrades gracefully to vector-only. If `embedding` is NULL, degrades to text-only.

### Where tsvector + pgvector applies in memory tiers

| Tier | Table | tsvector column | embedding column | Notes |
|------|-------|----------------|-----------------|-------|
| 2 (Episodic) | `messages` | `content_tsv` | — (optional) | Enables "search within conversation history" |
| 3 (Semantic/user) | `user_memories` | `content_tsv` | `embedding vector(768)` | Same RRF hybrid as entity_index |
| 4 (Semantic/world) | `chunks` | `content_tsv` | `embedding vector(768)` | Already implemented |
| 4 (Semantic/world) | `kg_entity_index` | `name_tsv` | `embedding vector(768)` | Already implemented |

Tier 1 (working memory) and Tier 5 (procedural) do not need search indexes — they are loaded directly, not retrieved by similarity.

---

## Cognitive Memory Types — The Framework

The cognitive science taxonomy of memory maps directly onto the distinct problems an AI system must solve. Using the right vocabulary prevents conflating problems that need different solutions.

| Cognitive type | What it stores | Human analogy | AI system problem |
|----------------|---------------|---------------|-------------------|
| **Short-term / Working** | Active information being processed right now | "Holding a phone number in mind" | Context window assembly — what fits in the LLM's prompt |
| **Episodic** | Specific events and their context | "I remember asking about GDPR in my meeting last Tuesday" | Conversation history — what was said, when, with what outcome |
| **Semantic (world)** | General facts and knowledge, domain-independent | "GDPR Article 17 is the right to erasure" | The RAG corpus — documents, chunks, knowledge graph |
| **Semantic (user)** | Durable facts about a specific person | "John prefers concise answers; works in legal" | User preferences and profile, persisted across sessions |
| **Procedural** | How to perform tasks; skills and patterns | "How to write a contract summary" | System prompts, tool definitions, query reformulation strategies |

Each type is stored differently, decays at a different rate, and has a different write/read pattern. Conflating them leads to the most common memory design mistakes: trying to put episodic memory into the context window, or using semantic memory for procedural knowledge.

---

## Mapping Cognitive Types to Implementation Tiers

| Cognitive type | Implementation tier | Storage | Lifespan | Write path | Read path |
|----------------|---------------------|---------|----------|-----------|----------|
| Short-term / Working | **Tier 1** | RAM (context window) | One request | Pipeline assembler | LLM directly |
| Episodic | **Tier 2** | PostgreSQL `messages` + `conversations` | 90 days (configurable) | After every turn, non-blocking | On every request, loaded by `session_id` |
| Semantic (user) | **Tier 3** | PostgreSQL + pgvector (`user_memories`) | Indefinite | Background fact extraction via nano model | `PRE_RETRIEVE` hook, cosine search |
| Semantic (world) | **Tier 4** | PostgreSQL + pgvector + Apache AGE | Until deleted | Ingestion pipeline | Retriever |
| Procedural | **Tier 5** | Files + DB (`system_prompts`, `tool_configs`) | Indefinite (versioned) | Human-written / admin-updated | App startup, per-request |

### Context assembly — how all tiers feed into every request

Every LLM call draws from all five tiers simultaneously. **Tier 1 (working memory) is the output of this assembly** — the bounded context window handed to the LLM, composed from the other four tiers. Priority order inside the token budget (highest priority — never trimmed before lower-priority items):

1. **System prompt** — Tier 5 (procedural)
2. **User memory context** — Tier 3 (top-3 relevant facts, hybrid tsvector + cosine search)
3. **Active conversation turns** — Tier 2 (last 8 turns, or summary + last 8 for long threads)
4. **Retrieved chunks** — Tier 4 (top-K, confidence-filtered via CrossEncoder)
5. **Current query** — always present, never trimmed
6. **→ Tier 1** (working memory) — the assembled, token-bounded context passed to the LLM

```python
# Inputs: Tiers 2, 3, 4, 5
# Output: Tier 1 — the assembled context window
tier1_context = assemble(
    system_prompt,                                                   # Tier 5 — procedural
    user_memories=memory_store.hybrid_search(query, user_id, k=3),  # Tier 3 — semantic/user
    history=conversation_store.load_active_window(session_id),       # Tier 2 — episodic
    chunks=retriever.retrieve(query, corpus_ids),                    # Tier 4 — semantic/world
)
if count_tokens(tier1_context) > budget:
    tier1_context = trim_to_budget(tier1_context)  # drops lowest-priority items first (chunks → turns → memories)
    response["context_truncated"] = True           # never silent

llm_response = await agent.run(query, context=tier1_context)  # Tier 1 consumed here
```

Token budget: 8,192 input tokens by default. See [Token Budget Management](#token-budget-management) for the full trim algorithm.

---

## Tier 1 — Short-Term / Working Memory (per-request)

**Cognitive parallel:** what you can hold in mind right now. Limited capacity; everything that doesn't fit must be retrieved from longer-term storage.

**What it is:** The assembled context window for a single LLM inference call — a composition of the other four tiers, assembled fresh per request and never persisted. It has no storage of its own; its content is entirely derived from Tiers 2–5.

**Token budget:** 8,192 input tokens by default. When the assembled context exceeds this, items are trimmed in priority order — see [Token Budget Management](#token-budget-management).

---

## Tier 2 — Episodic Memory (per-session, server-side)

**Cognitive parallel:** remembering that a specific conversation happened, what was said, and what the outcome was.

**Why server-side, not client-side:** the current design passes `message_history` in `ChatRequest`. This breaks multi-device, loses history on tab close, and scales poorly (30-turn conversation → 30× payload). Replace: client sends only `session_id`; server loads history from DB.

```
Before: ChatRequest { query, session_id, message_history: [...40 turns...] }
After:  ChatRequest { query, session_id }
        server: SELECT * FROM messages WHERE conversation_id = ... LIMIT 20
```

**Active window policy:**
- ≤ 20 turns → send full history as `message_history`
- > 20 turns → send `conversations.summary` + last 8 turns
- Summary is generated once when the threshold is crossed; updated when conversation doubles in length

**Decay and retention:** conversations expire after tenant-configured TTL (default 90 days). A nightly background job soft-deletes expired conversations, then hard-deletes after a 7-day grace period (for accidental recovery).

---

## Tier 3 — Semantic Memory (User, cross-session)

**Cognitive parallel:** durable knowledge about a specific person — their preferences, expertise, and context. Not tied to one event (that's episodic); applies across all interactions.

**What gets stored:** extracted facts about the user, not raw messages.

```
✅ Store: "User is a senior software engineer at a fintech company"
✅ Store: "User is working on a GDPR compliance project (inferred June 2026)"
✅ Store: "User prefers bullet-point answers over prose"
✅ Store: "User has payment processing expertise — corrected system on PCI-DSS scope"

❌ Don't store: "User asked what is GDPR?" (factual query, not a user fact)
❌ Don't store: raw message content (privacy, storage cost)
❌ Don't store: facts about the world ("GDPR Article 17 is right to erasure") — that's Tier 4
```

**Storage:** `user_memories` table with pgvector embedding for cosine retrieval. Mem0 handles extraction, deduplication, and contradiction resolution.

**Mem0 deduplication:** if a new memory contradicts an existing one, Mem0 asks the LLM whether to UPDATE, NOOP, or ADD. Example: old memory "User works at Acme Corp" + new fact "User mentioned moving to Beta Inc" → Mem0 updates the old memory rather than creating a duplicate.

**Hard limit:** 200 memories per user. When exceeded, prune by lowest retrieval frequency + oldest creation date. See [Pruning](#memory-pruning-eviction-and-compaction).

---

## Tier 4 — Semantic Memory (Knowledge Corpus, shared)

**Cognitive parallel:** general world knowledge — facts about the domain that are true for everyone, not specific to any user.

**What it is:** the RAG knowledge base — documents, semantic chunks (pgvector), and structured entities/relationships (Apache AGE knowledge graph).

**This tier is already well-designed** in `RAGV2_DESIGN.md §Ingestion Pipeline`. It is the only shared memory tier — all users on a corpus read from the same knowledge base.

**Key operations:**
- Write: ingestion pipeline (chunking → embedding → vector upsert + graph import)
- Read: retriever (hybrid vector + text + graph)
- Delete: by document_id (incremental ingest) or by corpus_id (corpus deletion)
- Compaction: HNSW index rebuild after large deletes (prevents index degradation)

---

## Tier 5 — Procedural Memory (system-level)

**Cognitive parallel:** knowing HOW to do something — motor skills, habits, expertise in procedure. In AI systems: the instructions that govern behavior, not the facts that inform content.

**What it is:** the system's "skills" — prompts, tool definitions, retrieval strategies, and reasoning patterns. Unlike the other tiers, procedural memory is set by operators (admins, developers), not learned from conversations.

**Components:**

| Component | Where stored | When loaded | Who changes it |
|-----------|-------------|------------|----------------|
| System prompt | `system_prompts` table (versioned) or `knowledge/agent/prompts.py` | App startup | Admin / developer |
| Tool definitions | Python source (`@agent.tool`) | App startup | Developer |
| Retrieval strategy | `CorpusConfig.graph_extraction_contract`, `min_confidence_score`, etc. | Per request | Admin via corpus config |
| Citation constraint | System prompt (hardcoded) | Per request | Developer |
| Uncertainty notice | Injected when `low_confidence_context=True` | Per request | Pipeline |
| Query reformulation | Not yet designed; future: nano model rewrites query before retrieval | — | — |

**Versioning:** system prompts should be versioned (store old versions for regression comparison after prompt changes). Evaluation runs capture `git_commit` which pins the prompt version used.

**Procedural memory is currently static** — prompts are written once and updated rarely. A future enhancement is **prompt tuning**: automatically adjusting prompts based on evaluation feedback (e.g., if faithfulness score drops, tighten the citation constraint automatically). This is out of scope for v2.

---

## How the Tiers Interact Per Request

```
POST /api/v1/chat  { query: "...", session_id: "sess_abc" }
        │
        ├─ 1. Load procedural memory (Tier 5) — app startup cache
        │     system_prompt, tool_definitions
        │
        ├─ 2. Load episodic memory (Tier 2) — DB lookup by session_id
        │     conversations + messages (last 8 turns, or summary + last 8)
        │
        ├─ 3. Load user semantic memory (Tier 3) — PRE_RETRIEVE hook
        │     user_memories WHERE embedding <=> embed(query) ORDER BY cosine LIMIT 3
        │
        ├─ 4. Retrieve world knowledge (Tier 4) — hybrid search
        │     pgvector semantic + tsvector text + AGE graph → top-K chunks
        │
        ├─ 5. Assemble working memory (Tier 1) — trim to 8,192 tokens
        │     system_prompt + user_memories + history + chunks + query
        │
        ├─ 6. LLM inference
        │     agent.run(query, message_history=active_window, deps=state)
        │
        ├─ 7. Persist episodic turn (Tier 2) — background, non-blocking
        │     INSERT INTO messages (conversation_id, role, content, citations, ...)
        │     UPDATE conversations SET turn_count++, last_turn_at=NOW()
        │     IF turn_count == 20: trigger auto-summarization
        │
        ├─ 8. Update user semantic memory (Tier 3) — background, non-blocking
        │     nano_model.extract_facts(query, answer) → mem0.add(facts, user_id)
        │
        └─► RAGResponse { answer, citations, session_id, request_id, cost_usd, ... }
```

Steps 7 and 8 are `asyncio.create_task()` — they never block the response path.

---

## Token Budget Management

Total budget: 8,192 input tokens. Allocation targets:

| Tier | Component | Target tokens | Trim behaviour |
|------|-----------|--------------|----------------|
| 5 | System prompt | ~300 | Never trimmed |
| 3 | User memories (top-3) | ~200 | Reduce to top-1 if needed |
| 2 | Conversation history | ~600–1,000 | Drop oldest turns → substitute summary |
| 4 | Retrieved chunks (top-5) | ~1,000 | Drop lowest-confidence first |
| — | Query | ~50 | Never trimmed |
| — | Buffer / formatting | ~300 | — |

**Trim algorithm (in order):**

```python
def trim_to_budget(parts: ContextParts, budget: int) -> ContextParts:
    # Step 1: drop retrieved chunks below min_confidence (already pre-filtered)
    # Step 2: drop lowest-confidence chunks until fits
    while count_tokens(parts) > budget and len(parts.chunks) > 1:
        parts.chunks = sorted(parts.chunks, key=lambda c: c.confidence, reverse=True)
        parts.chunks.pop()

    # Step 3: replace oldest turns with conversation summary
    if count_tokens(parts) > budget and parts.history_turns > 4:
        parts.history = [parts.conversation_summary] + parts.history[-4:]

    # Step 4: reduce user memories to top-1
    if count_tokens(parts) > budget and len(parts.user_memories) > 1:
        parts.user_memories = parts.user_memories[:1]

    # Step 5: emit warning — don't fail silently
    if count_tokens(parts) > budget:
        log.warning("context_truncated: budget=%d, assembled=%d", budget, count_tokens(parts))
        parts.metadata["context_truncated"] = True

    return parts
```

**Rule:** trimming is visible to the caller (`context_truncated: true` in response). Silent truncation is a bug.

---

## Mem0 Write Path — When Memories Are Created

Triggered by the `POST_LLM` hook, as a non-blocking background task.

```python
# knowledge/memory/mem0_store.py
async def extract_and_store(
    query: str,
    answer: str,
    recent_turns: list[tuple[str, str]],   # last 2 turns for context
    user_id: str,
    tenant_id: str,
) -> None:
    """Extract memorable user facts. Fire-and-forget — never blocks response."""

    # Nano model prompt (cheap: ~50 tokens in, ~100 out):
    # Context: [last 2 turns for continuity]
    # Q: {query}   A: {answer}
    # "List facts revealed about the USER specifically — role, expertise,
    #  preferences, ongoing projects, corrections made to the system.
    #  Ignore facts about the subject matter. If none, return []."
    new_facts = await nano_model.extract_user_facts(query, answer, recent_turns)

    for fact in new_facts:
        # Mem0 handles: embed → search existing → LLM decides UPDATE/ADD/NOOP
        await mem0_client.add(
            fact,
            user_id=hash_user_id(user_id, tenant_id),
            metadata={"tenant_id": tenant_id},
        )
```

**When NOT to extract:**
- Query was abstained (no reliable answer → nothing to learn from)
- `pipeline_status != "answered"` (citation gate or judge gate fired)
- `mem0_enabled = False` in tenant config (off by default)
- Pure factual queries with no user-revealing signal ("What is GDPR?")

---

## Mem0 Read Path — When Memories Are Injected

Fires at `PRE_RETRIEVE` hook — before retrieval, so user context shifts embedding relevance before the main retrieval runs.

```python
# Hybrid search on user_memories: tsvector BM25 + pgvector cosine combined with RRF (k=60)
# Same pattern as kg/entity_index.py hybrid_search() — see tsvector + pgvector section
    query=current_query,
    user_id=hash_user_id(user_id, tenant_id),
    limit=3,
)
# Injected as:
# "User context:\n- Senior engineer at fintech company\n- Working on GDPR compliance"
```

**Why PRE_RETRIEVE:** user context shifts the embedding space slightly, improving retrieval precision for users with known domain focus. A compliance engineer's query "What are the restrictions?" retrieves different chunks than an engineer with no profile.

---

## Conversation Auto-Summarization

When a conversation exceeds 20 turns, the full history exceeds the working memory budget.

**Summarization trigger:** `turn_count > 20 AND summary IS NULL`

**Nano model prompt:**
> "Summarize this conversation in 3–5 sentences. Cover: what the user was trying to learn, key facts established, and any conclusions reached. Do not quote specific messages."

**Active window sent to LLM:**
```
[Summary]: "The user asked about GDPR Article 17 and the system established that..."
[Turn 33]: user: "What about cross-border transfers?"
[Turn 34]: assistant: "..."
...
[Turn 40 — current]: user: "Explain adequacy decisions."
```

**Re-summarization:** when `turn_count` grows by another 20 beyond the last summarization, re-summarize everything up to `turn_count - 8`.

---

## Memory Pruning, Eviction, and Compaction

This is the hardest part of memory design. Without it, memory degrades over time: stale facts persist, storage grows unboundedly, and retrieval quality drops as the signal-to-noise ratio falls.

### Working Memory (Tier 1) — no persistence, no pruning needed

Trimming happens at assembly time (see [Token Budget Management](#token-budget-management)). Nothing to prune — it's RAM-only per request.

---

### Episodic Memory (Tier 2) — time-based eviction + compaction

**Time-based eviction:**
```sql
-- Nightly job: mark expired conversations for deletion
UPDATE conversations
SET deleted_at = NOW()
WHERE expires_at < NOW() AND deleted_at IS NULL;

-- After 7-day grace period: hard delete
DELETE FROM conversations
WHERE deleted_at < NOW() - INTERVAL '7 days';
-- messages cascade automatically (ON DELETE CASCADE)
```

**Compaction (summarization as compression):**

Episodic compaction = auto-summarization. Instead of deleting old turns, compress them into a summary and delete the raw rows:

```python
# When summary is generated for a long conversation:
async def compact_conversation(conversation_id: UUID) -> None:
    # 1. Generate summary from turns 1 to (N - 8)
    old_turns = await load_turns(conversation_id, exclude_last=8)
    summary = await nano_model.summarize(old_turns)

    # 2. Store summary, delete raw old turns
    async with db.transaction():
        await db.execute(
            "UPDATE conversations SET summary = $1 WHERE id = $2",
            summary, conversation_id
        )
        await db.execute(
            """DELETE FROM messages
               WHERE conversation_id = $1
               AND created_at < (
                   SELECT created_at FROM messages
                   WHERE conversation_id = $1
                   ORDER BY created_at DESC LIMIT 1 OFFSET 7
               )""",
            conversation_id
        )
    # Result: summary row + last 8 turns. Old turns gone. Episodic knowledge preserved.
```

**Storage targets:** keep max 8 raw turns per active conversation after compaction. Inactive conversations (> 30 days since last turn) are compacted even if < 20 turns.

---

### Semantic User Memory (Tier 3) — LRU eviction + contradiction resolution + compaction

This tier requires the most nuanced management because:
- Memories have different relevance over time
- Some memories become stale (user changes jobs, project ends)
- Redundant memories accumulate if deduplication isn't tight
- The user has a right to prune memories they disagree with

**Hard capacity limit:** 200 memories per user per tenant. Enforced on every `mem0.add()` call:

```python
async def enforce_capacity(user_id: str, tenant_id: str) -> None:
    count = await db.fetchval(
        "SELECT COUNT(*) FROM user_memories WHERE user_id=$1 AND tenant_id=$2",
        user_id, tenant_id
    )
    if count >= 200:
        # Prune: remove memories that haven't been retrieved in 60+ days
        # and were created more than 90 days ago (stale signal)
        await db.execute("""
            DELETE FROM user_memories
            WHERE user_id = $1 AND tenant_id = $2
              AND last_retrieved_at < NOW() - INTERVAL '60 days'
              AND created_at < NOW() - INTERVAL '90 days'
            ORDER BY last_retrieved_at ASC
            LIMIT 20
        """, user_id, tenant_id)
```

**LRU tracking:** add `last_retrieved_at TIMESTAMPTZ` to `user_memories`. Every time a memory is returned by `mem0.search()`, update this field. Memories never accessed drop to the bottom of the eviction queue.

**Contradiction resolution:** built into Mem0. When `mem0.add("User moved to Beta Inc")` conflicts with existing "User works at Acme Corp", Mem0 calls a nano model to decide: UPDATE the existing memory (preserving its ID and creation date) rather than creating a duplicate. This keeps count stable.

**Compaction — merging related memories:**

When multiple memories cover the same topic:
```
"User works in compliance"
"User asked many questions about GDPR"
"User is auditing their data retention policies"
```
→ merge into: "User is a compliance professional working on GDPR-related data governance"

Compaction runs as a weekly background job using a nano model:

```python
async def compact_user_memories(user_id: str, tenant_id: str) -> None:
    memories = await load_all_memories(user_id, tenant_id)
    # Cluster by embedding similarity (cosine >= 0.85 → same topic cluster)
    clusters = cluster_by_similarity(memories, threshold=0.85)
    for cluster in clusters:
        if len(cluster) >= 3:  # only compact if 3+ similar memories
            merged = await nano_model.merge_memories(cluster)
            async with db.transaction():
                await db.execute("DELETE FROM user_memories WHERE id = ANY($1)", [m.id for m in cluster])
                await mem0_client.add(merged, user_id=user_id)
```

**Manual eviction:** users can delete individual memories or all memories via the memory management UI. Deletion is immediate; no soft-delete (user preference must be respected instantly per GDPR).

---

### Knowledge Memory (Tier 4) — incremental delete + HNSW rebuild

**Document delete (incremental ingest):**
```python
# On re-ingest of a changed document:
await vector_store.delete_by_document_id(document_id)   # deletes chunks
await age_store.delete_document_vertices(corpus_id, document_id)  # deletes graph nodes
# Then re-ingest fresh chunks/graph
```

**Corpus reset:**
```python
await vector_store.truncate_corpus(corpus_id)           # DELETE WHERE corpus_id = $1
await age_store.delete_corpus_graph(corpus_id)          # DROP GRAPH
await entity_index.delete_corpus(corpus_id)             # DELETE from shadow table
```

**HNSW index compaction:** after large batch deletes (> 20% of corpus), the HNSW index has dead nodes that increase query latency. Rebuild:
```sql
REINDEX INDEX CONCURRENTLY chunks_embedding_idx;
-- CONCURRENTLY means no downtime; runs alongside live queries
```

Trigger: run after any operation that deletes > 20% of a corpus's chunks. Track via `pg_stat_user_indexes` metric.

**Semantic cache invalidation (L3):** when a document is updated, cached answers that used that document's chunks may be stale. Invalidate:
```sql
-- After re-ingesting a document, delete L3 cache entries that cited its corpus
DELETE FROM semantic_cache WHERE corpus_ids @> ARRAY[$1]::text[];
```

---

## Framework Assessment — Build vs Buy

| Framework | Covers | Doesn't cover | Verdict |
|-----------|--------|--------------|---------|
| **Mem0** | Tier 3: extraction, cosine retrieval, dedup/contradiction resolution; basic capacity limits | Tier 1 trim policy, Tier 2 schema + compaction, Tier 4 all, Tier 5 all | **Use for Tier 3 only.** It does one thing well. |
| **Zep** | Tier 2: conversation storage + auto-summarization + entity extraction from dialog | Tier 1 trim, Tier 3 (has some), Tier 4/5 none | Good Tier 2 alternative if you want entity tracking from conversations. Adds a service dependency. |
| **Letta (MemGPT)** | In-context memory paging — lets the LLM itself decide what to move in/out of context | Requires LLM that supports function-calling for memory ops; high latency; complex | Overkill for a RAG system; best for autonomous agents |
| **LangMem** | Similar to Mem0 within LangChain ecosystem | Same gaps as Mem0; only useful if you're on LangChain | Not relevant for our Pydantic AI stack |

**Recommendation for this system:**

```
Tier 1 (Working)    — implement ourselves (token budget + trim policy)
Tier 2 (Episodic)   — implement ourselves (PostgreSQL; simple and fits our stack)
Tier 3 (Semantic/U) — Mem0 (open-source, pgvector-backed, already in design)
Tier 4 (Semantic/K) — implement ourselves (already designed: pgvector + AGE)
Tier 5 (Procedural) — implement ourselves (prompts.py + versioned DB table)

Pruning/eviction:   — implement ourselves (nightly jobs; Mem0 handles Tier 3 capacity)
```

The only framework worth adopting is **Mem0 for Tier 3**. Everything else is either simpler to build ourselves given our PostgreSQL stack, or solves problems we don't have at this scale.

**When to reconsider Zep for Tier 2:** if auto-extracting structured entities from conversations (e.g., "user mentioned they have a contract with Google") becomes a priority — Zep does this well. For now, Mem0's fact extraction at the turn level is sufficient.

---

## Database Schema

```sql
-- ── Tier 2: Episodic memory ─────────────────────────────────────────────────

CREATE TABLE conversations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      TEXT NOT NULL UNIQUE,  -- crypto.randomUUID() from frontend
    tenant_id       TEXT NOT NULL,
    user_id         TEXT NOT NULL,         -- SHA-256(jwt_sub + tenant_salt)
    corpus_ids      TEXT[] NOT NULL,
    title           TEXT,                  -- first 60 chars of first user message
    summary         TEXT,                  -- auto-generated after 20 turns; NULL until then
    turn_count      INT NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_turn_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMPTZ,           -- NULL = never; set by retention policy
    deleted_at      TIMESTAMPTZ            -- soft delete; hard delete after 7-day grace
);
CREATE INDEX ON conversations (user_id, last_turn_at DESC);
CREATE INDEX ON conversations (tenant_id, last_turn_at DESC);
CREATE INDEX ON conversations (expires_at) WHERE expires_at IS NOT NULL;

CREATE TABLE messages (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id   UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role              TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content           TEXT NOT NULL,
    -- tsvector for full-text search within conversation history (no embedding needed -
    -- scope is already user+conversation so cosine ANN adds no precision benefit)
    content_tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    -- Assistant-only fields (NULL on user rows)
    citations         JSONB,
    pipeline_status   TEXT,
    confidence        FLOAT,
    model_tier        TEXT,
    prompt_tokens     INT,
    completion_tokens INT,
    cost_usd          FLOAT,
    cache_hit         TEXT,
    request_id        UUID,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX ON messages (conversation_id, created_at);
CREATE INDEX ON messages USING GIN (content_tsv);  -- "find when I asked about X" feature

-- ── Tier 3: Semantic user memory ─────────────────────────────────────────────

CREATE TABLE user_memories (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             TEXT NOT NULL,         -- SHA-256(jwt_sub + tenant_salt)
    tenant_id           TEXT NOT NULL,
    content             TEXT NOT NULL,         -- extracted fact sentence
    -- tsvector for keyword-exact memory retrieval alongside pgvector cosine (same RRF pattern as entity_index)
    content_tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding           vector(768),           -- for cosine ANN (HNSW); combined with content_tsv via RRF
    source_message_id   UUID,                  -- message that triggered extraction
    last_retrieved_at   TIMESTAMPTZ,           -- updated on every search hit (for LRU eviction)
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX ON user_memories (user_id, tenant_id);
CREATE INDEX ON user_memories USING GIN (content_tsv);   -- BM25 leg of hybrid search
CREATE INDEX ON user_memories USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
-- Hybrid search query for user_memories (same RRF k=60 pattern as kg/entity_index.py):
-- 1. text_ranked: ts_rank on content_tsv @@ websearch_to_tsquery(query)
-- 2. vec_ranked:  embedding <=> embed(query) ANN LIMIT 60
-- 3. rrf:         COALESCE(1/(60+t.rn), 0) + COALESCE(1/(60+v.rn), 0)
-- Falls back to vector-only if tsvector has no hits (e.g. pure stopword query).
-- Falls back to text-only if embedding is NULL (shouldn't happen; embed on every add).

-- ── Tier 5: Procedural memory ────────────────────────────────────────────────

CREATE TABLE system_prompts (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,            -- e.g. "rag_agent_v3", "judge_v2"
    content     TEXT NOT NULL,
    version     INT NOT NULL DEFAULT 1,
    active      BOOLEAN NOT NULL DEFAULT FALSE,
    corpus_id   TEXT,                     -- NULL = global; set for corpus-specific overrides
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by  TEXT NOT NULL             -- admin user who set it
);
CREATE UNIQUE INDEX ON system_prompts (name, version);
CREATE INDEX ON system_prompts (name, active) WHERE active = TRUE;
```

---

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/v1/conversations` | `reader` | List conversations for current user (paginated, newest first) |
| `GET` | `/v1/conversations/{id}` | `reader` | Get conversation + messages |
| `DELETE` | `/v1/conversations/{id}` | `reader` | Delete conversation (GDPR) |
| `GET` | `/v1/conversations/{id}/summary` | `reader` | Get or trigger auto-summary |
| `GET` | `/v1/memories` | `reader` | List all memories for current user |
| `POST` | `/v1/memories` | `reader` | Manually add a memory |
| `DELETE` | `/v1/memories/{id}` | `reader` | Delete one memory |
| `DELETE` | `/v1/memories` | `reader` | Delete ALL memories (right to erasure) |
| `GET` | `/v1/admin/users/{id}/memories` | `admin` | Admin: view any user's memories |
| `POST` | `/v1/admin/system-prompts` | `admin` | Create / update a versioned system prompt |
| `GET` | `/v1/admin/system-prompts` | `admin` | List all system prompt versions |
| `POST` | `/v1/admin/system-prompts/{id}/activate` | `admin` | Set a prompt version as active |

---

## Frontend Memory Management

### Conversation history (in `/chat` sidebar)

```
┌── ConversationSidebar ──────────────────────┐
│  [+ New Chat]                               │
│                                             │
│  Today                                      │
│  > GDPR compliance questions       [🗑]     │
│    2h ago · 34 turns  [summarized]          │
│                                             │
│  Yesterday                                  │
│  > Contract review — Acme deal     [🗑]     │
│    Yesterday · 12 turns                     │
│                                             │
│  June 5                            [load]   │
│    PTO policy lookup · expired soon         │
└─────────────────────────────────────────────┘
```

- Conversations loaded from `GET /v1/conversations` (server-side, not Zustand)
- Summarized badge when `conversations.summary IS NOT NULL`
- "Expired soon" badge when `expires_at < NOW() + 7 days`
- [🗑] triggers `DELETE /v1/conversations/{id}` + removes from list

### Memory manager (`/memories` page)

```
┌── My Memories ─────────────────────────────────────────────────────┐
│  The system uses these to personalize responses.      [Delete all] │
│                                                                     │
│  📌 Senior software engineer at a fintech company                  │
│     From: "GDPR questions" · June 3                      [🗑]      │
│                                                                     │
│  📌 Working on GDPR compliance project                             │
│     From: "Contract review" · June 5                     [🗑]      │
│                                                                     │
│  📌 Prefers concise bullet-point answers                           │
│     Added manually · June 6                              [🗑]      │
│                                                                     │
│  [+ Add memory]                                                     │
└────────────────────────────────────────────────────────────────────┘
```

- Each memory links to the source conversation
- Individual delete is immediate (GDPR requirement)
- [Delete all] calls `DELETE /v1/memories` with confirmation dialog
- [+ Add memory] lets power users manually teach the system facts about themselves

---

## GDPR and Right to Erasure

**User-level erasure** flows:

1. `DELETE /v1/memories` → hard-delete all `user_memories` rows (no soft-delete; immediate)
2. `DELETE /v1/conversations` (bulk) → soft-delete `conversations`; cascade deletes `messages`
3. In `audit_events`: replace `user_id` with `SHA-256("ERASED" + tenant_salt)` — row preserved for compliance, user not identifiable
4. In `user_feedback` and `implicit_signals`: same anonymization

**`user_id` storage:**
- Never stored as plaintext
- Always `SHA-256(jwt_sub + tenant_salt)`
- Salt is per-tenant — cross-tenant user linkage is impossible even if hashes leak

**Data retention defaults:**

| Table | Default | Configurable |
|-------|---------|-------------|
| `conversations` | 90 days | Per tenant |
| `messages` | 90 days (cascade) | Per tenant |
| `user_memories` | Until deleted | User-controlled |
| `audit_events` | 2 years | Compliance |
| `user_feedback` | 1 year | — |

---

## Quick Reference

```
Cognitive type → Tier → Storage → Key operation

Short-term     → Tier 1 → RAM            → assemble_working_memory() per request
Episodic       → Tier 2 → PostgreSQL     → INSERT message after turn (background)
                                            SELECT last 8 turns by session_id
                                            Compact: summarize when > 20 turns
                                            Evict: DELETE after expiry TTL
Semantic/user  → Tier 3 → PostgreSQL     → mem0.add(fact) after turn (background)
                          + pgvector       mem0.search(query) at PRE_RETRIEVE
                                            Evict: LRU + hard cap 200/user
                                            Compact: merge similar memories weekly
Semantic/world → Tier 4 → PostgreSQL     → retriever.retrieve() per request
                          + pgvector       Prune: delete by document_id on re-ingest
                          + Apache AGE     Compact: REINDEX after large deletes
Procedural     → Tier 5 → Files + DB    → load at startup; prompt versioned in DB
                                            No decay; human-updated

Framework verdict:
  Tier 3 only → use Mem0 (extraction, dedup, cosine retrieval, contradiction resolution)
  Everything else → build ourselves (fits PostgreSQL stack, fewer dependencies)
```
