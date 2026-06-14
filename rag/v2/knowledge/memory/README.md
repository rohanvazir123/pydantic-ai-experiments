# knowledge/memory/

## Table of Contents

- [What This Is](#what-this-is)
- [Files](#files)
- [Memory Tiers](#memory-tiers)

---

## What This Is

All five memory tiers for the RAG system. Handles episodic conversation storage, user semantic memory (Mem0), working memory assembly, and background pruning jobs.

For the full memory architecture including cognitive type mapping, tsvector + pgvector hybrid search, and the framework assessment (Mem0 vs Zep vs Letta), see `basics/rag/memory/MEMORY_DESIGN.md`.

---

## Files

| File | Purpose |
|------|---------|
| `mem0_store.py` | **Tier 3** — Mem0-backed user semantic memory: `extract_and_store()` (background), `hybrid_search()` (tsvector + pgvector RRF) |
| `conversation_store.py` | **Tier 2** — episodic: conversation + message CRUD; `load_active_window()` (last 8 turns, or summary + last 8) |
| `summarizer.py` | **Tier 2** — auto-summarize conversations when `turn_count > 20` using nano model |
| `working_memory.py` | **Tier 1** — context assembly from Tiers 2–5; `trim_to_budget()` with priority ordering |
| `pruning.py` | Background jobs: Tier 2 TTL eviction, Tier 3 LRU eviction + weekly compaction |

---

## Memory Tiers

| Tier | Cognitive type | Storage | Lifespan |
|------|---------------|---------|----------|
| 1 | Short-term / Working | RAM | Per request |
| 2 | Episodic | PostgreSQL `conversations` + `messages` | 90 days |
| 3 | Semantic (user) | PostgreSQL + pgvector `user_memories` | Indefinite |
| 4 | Semantic (world) | PostgreSQL + pgvector + AGE | Until deleted |
| 5 | Procedural | Files + DB `system_prompts` | Indefinite, versioned |

**Important:** `message_history` is NOT passed in `ChatRequest`. The server loads conversation history from DB using `session_id`. This enables multi-device support and prevents state loss on browser refresh.
