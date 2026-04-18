# FAQ - Pydantic AI RAG System

Code references: line numbers point to files under `rag/` in this repo.

---
## Table of Contents

- [Q1. What is RAG and why is it preferred over fine-tuning for knowledge-intensive tasks?](#q1)
- [Q2. What are the main failure modes of a naive RAG pipeline?](#q2)
- [Q3. What is the difference between standard RAG and agentic RAG?](#q3)
- [Q4. How does chunking strategy affect retrieval quality?](#q4)
- [Q5. What is the "lost in the middle" problem and how does chunk ordering help?](#q5)
- [Q6. Why store the full document alongside chunks?](#q6)
- [Q7. Explain hybrid search. What problem does each leg solve?](#q7)
- [Q8. Walk through RRF. What is the formula and what does k=60 do?](#q8)
- [Q9. When does text search win over semantic search?](#q9)
- [Q10. When does semantic search win?](#q10)
- [Q11. Semantic Hit Rate@5 = 0.90 vs hybrid = 0.80. How do you explain this?](#q11)
- [Q12. If you had to drop one leg, which would you keep?](#q12)
- [Q13. Why PostgreSQL over a dedicated vector DB?](#q13)
- [Q14. What is IVFFlat and how does it trade accuracy for speed?](#q14)
- [Q15. What does `register_vector` do and why in `init=`?](#q15)
- [Q16. Why `executemany` for batch inserts?](#q16)
- [Q17. `ON DELETE CASCADE` — what does it do and why is it critical?](#q17)
- [Q18. Why UUID primary keys over auto-increment?](#q18)
- [Q19. `GENERATED ALWAYS AS (...) STORED` — what does this mean?](#q19)
- [Q19b. How do I install `psql` and connect to the Neon database?](#q19b)
- [Q20. What is a `tsvector` and how does it differ from the original text?](#q20)
- [Q21. What does stemming do and when can it cause false positives?](#q21)
- [Q22. Why `plainto_tsquery` instead of `to_tsquery` for user input?](#q22)
- [Q23. Why is a GIN index better than B-tree for tsvector?](#q23)
- [Q24. "30 days PTO" — the number 30 is dropped. Why? How would you handle numeric search?](#q24)
- [Q25. What happens when `plainto_tsquery` produces an empty query?](#q25)
- [Q26. What does Docling's `HybridChunker` do that sliding-window cannot?](#q26)
- [Q27. What is `contextualize()` and why does it improve embedding quality?](#q27)
- [Q28. Describe the fallback chunking path exactly.](#q28)
- [Q29. Why is `DocumentConverter` cached via `_get_converter()`?](#q29)
- [Q30. What is `merge_peers=True`?](#q30)
- [Q31. Why cache `DocumentConverter` as an instance attribute, not a module-level singleton?](#q31)
- [Q32. What does `nomic-embed-text` produce and why 768 dimensions?](#q32)
- [Q33. Cosine similarity vs Euclidean distance — why cosine?](#q33)
- [Q34. The embedder has an in-memory cache — what is the cache keyed on and what are its limits?](#q34)
- [Q35. Switching from nomic-embed-text (768-dim) to text-embedding-3-small (1536-dim) — what changes?](#q35)
- [Q36. Symmetric vs asymmetric embedding models — which for RAG?](#q36)
- [Q37. Explain HyDE. Why might it outperform raw query embedding?](#q37)
- [Q38. What are the risks of HyDE?](#q38)
- [Q39. When would you enable HyDE?](#q39)
- [Q40. How is HyDE implemented in the retriever?](#q40)
- [Q41. What problem does a cross-encoder solve that bi-encoder retrieval cannot?](#q41)
- [Q42. LLM reranker vs CrossEncoder — trade-offs?](#q42)
- [Q43. Why `asyncio.gather` for LLM reranker scoring?](#q43)
- [Q44. At what corpus size or query volume would you enable the reranker?](#q44)
- [Q45. Retrieval recall vs reranking precision — how do they compose?](#q45)
- [Q46. What makes this system "agentic"?](#q46)
- [Q47. How does Pydantic AI's tool system work?](#q47)
- [Q47a. How does the LLM know which tool to call — does Pydantic AI register tool names with it?](#q47a)
- [Q48. What is `RAGState` and why are its attributes `PrivateAttr`?](#q48)
- [Q49. Why `ContextVar` for Langfuse trace context?](#q49)
- [Q50. Why is per-user state important in a multi-user chat app?](#q50)
- [Q51. How does the agent handle tool call failures?](#q51)
- [Q52. What problem does Mem0 solve that conversation history cannot?](#q52)
- [Q53. How is Mem0 stored in this project?](#q53)
- [Q54. `add()` vs `get_context_string()` — difference?](#q54)
- [Q55. Why is Mem0 disabled by default?](#q55)
- [Q56. How would you prevent Mem0 from storing sensitive information?](#q56)
- [Q57. Why must all I/O be async? What happens with a blocking call?](#q57)
- [Q58. What is an asyncpg connection pool and why use it?](#q58)
- [Q59. Maximum latency improvement from `asyncio.gather` on semantic + text search?](#q59)
- [Q60. Why `init=register_vector` rather than registering after pool creation?](#q60)
- [Q61. If `asyncio.gather` has two coroutines and one raises an exception — what happens?](#q61)
- [Q62. Hit Rate@K vs Precision@K — when do they diverge?](#q62)
- [Q63. What does MRR measure that Hit Rate doesn't?](#q63)
- [Q64. Walk through the NDCG formula.](#q64)
- [Q65. Is 10 queries a sufficient gold dataset?](#q65)
- [Q66. Why do "company mission and vision" and "DocFlow AI" miss consistently?](#q66)
- [Q67. Recall@5 shows values above 1.0 — is that a bug?](#q67)
- [Q68. Why are unit tests and integration tests in the same file?](#q68)
- [Q69. How would you use these metrics to decide whether to enable HyDE or the reranker?](#q69)
- [Q69a. What were the measured retrieval metrics on the NeuralFlow AI corpus?](#q69a)
- [Q69b. Where in the code are retrieval metrics collected?](#q69b)
- [Q70. What does Langfuse trace in this project?](#q70)
- [Q71. Why `ContextVar` rather than function arguments?](#q71)
- [Q72. Using Langfuse traces to debug a wrong answer.](#q72)
- [Q73. Trace vs span vs generation in Langfuse?](#q73)
- [Q74. Two-table schema (documents + chunks) — why not one table?](#q74)
- [Q75. Walk through ingestion of a raw PDF to a searchable chunk.](#q75)
- [Q76. Scale to 10M documents — what breaks first?](#q76)
- [Q77. Implementing true incremental ingestion with deduplication.](#q77)
- [Q78. Is multi-tenancy supported? What would it take to make this prototype production-ready for multiple tenants?](#q78)
- [Q79. Risk of changing the embedding model after ingestion.](#q79)
- [Q80. Sub-100ms latency — what to sacrifice first?](#q80)
- [Q81. Why `pydantic-settings` instead of `os.environ`?](#q81)
- [Q82. What does `ruff` check for vs `flake8 + black`?](#q82)
- [Q83. Why Pydantic models for `ChunkData` and `SearchResult` instead of plain dataclasses?](#q83)
- [Q84. Why `from collections.abc import Callable` rather than `callable`?](#q84)
- [Q85. How does `IngestionConfig` → `ChunkingConfig` separation keep concerns clean?](#q85)
- [Q91. Walk through full ingestion step by step.](#q91)
- [Q92. How does `DocumentConverter` differ from PyPDF2 / pdfplumber?](#q92)
- [Q93. What internal representation does `DoclingDocument` provide and how does `HybridChunker` use it?](#q93)
- [Q94. Explain `contextualize()` — what exactly gets prepended?](#q94)
- [Q95. What is `merge_peers=True` — give an example.](#q95)
- [Q96. What happens to a table in a PDF during chunking?](#q96)
- [Q97. Tokenizer mismatch: `all-MiniLM-L6-v2` for chunking, `nomic-embed-text` for embedding.](#q97)
- [Q98. Describe the fallback chunking path exactly.](#q98)
- [Q99. Why cache `DocumentConverter`?](#q99)
- [Q100. MD5 for content hashing — how it works and limitations.](#q100)
- [Q101. Incremental ingestion — walk through all four cases.](#q101)
- [Q102. Why `_result_cache.clear()` after ingestion?](#q102)
- [Q103. YAML frontmatter — where stored, how used?](#q103)
- [Q104. Top three bottlenecks at 10,000 docs/day and fixes.](#q104)
- [Q105. Parallelizing ingestion while sharing `DocumentConverter` and the asyncpg pool.](#q105)
- [Q106. Zero-downtime re-index when `clean_before_ingest=True`.](#q106)
- [Q107. Scanned PDFs with no text layer.](#q107)
- [Q108. Why return both markdown string and `DoclingDocument`?](#q108)
- [Q109. Audio files — how are they different from PDF chunks?](#q109)
- [Q110. Impact of raw text fallback when PDF conversion fails.](#q110)
- [Q86. RRF scores of 0.01–0.03 — why isn't this low confidence?](#q86)
- [Q87. After re-ingestion, previously passing tests now fail. Possible causes.](#q87)
- [Q88. Query "PTO" — what happens in tsvector and why might it miss "paid time off"?](#q88)
- [Q89. LLM reranker with partial failure (rate limiting).](#q89)
- [Q90. Changing `chunk_overlap` from 100 to 0 — improve some metrics, hurt others?](#q90)
- [Q111. What are the main scale bottlenecks in this system at 1M documents?](#q111)
- [Q112. What are the ingestion latency bottlenecks and how would you profile them?](#q112)
- [Q113. What are the retrieval latency bottlenecks and how would you reduce them to sub-100ms?](#q113)
- [Q114. What models can be swapped in to improve retrieval precision?](#q114)
- [Q115. How would you benchmark and choose between embedding models for this corpus?](#q115)
- [Q116. At what scale would you move away from PostgreSQL/pgvector to a dedicated vector database?](#q116)
- [Q116a. Why aren't we using `pg_textsearch` (Timescale's BM25 extension) instead of `tsvector`/`ts_rank`?](#q116a)
- [Q116b. What other PostgreSQL text search extensions exist, which does this project use, and what would each add?](#q116b)
- [Q116c. What indexes currently exist on the `chunks` table?](#q116c)
- [Q116d. How does re-indexing happen on the fly?](#q116d)
- [Q116e. How are new documents auto-ingested and re-indexed?](#q116e)
- [Q116f. Which tests are currently failing and what needs to be done to fix them?](#q116f)
- [Q116g. How do I inspect what's actually stored in the `chunks` table?](#q116g)
- [Q116h. Why doesn't the SELECT query show trigram data? How do I see what the trigram index stores?](#q116h)
- [Q117. What does the PostgreSQL data model look like — entity diagram and sample records?](#q117)
- [Q118. How does the Pydantic AI agent loop work in this codebase — agent creation, RunContext, deps, and the tool execution cycle?](#q118)
- [Q120. What are all the changes needed to make this RAG system production-ready?](#q120)
- [Q122. Is this project using semantic chunking or fixed-size chunking with overlaps?](#q122)
- [Q121. What are all the tunables in this RAG system and how should they be set for performance?](#q121)
- [Q123. What HTTP endpoints does the REST API expose?](#q123)
- [Q124. How does `POST /v1/chat` work under the hood?](#q124)
- [Q125. How does streaming work — what is the SSE format?](#q125)
- [Q126. What does `GET /health` check and what HTTP status does it return?](#q126)
- [Q127. How does `POST /v1/ingest` work and what are its limitations?](#q127)
- [Q128. Why SSE over WebSockets for streaming?](#q128)
- [Q129. How is the asyncpg pool lifecycle managed across HTTP requests?](#q129)
- [Q130. How would you add authentication to the REST API?](#q130)
- [Q131. How do I fire off a query over the REST API?](#q131)

---


## RAG Fundamentals

<a id="q1"></a>
**Q1. What is RAG and why is it preferred over fine-tuning for knowledge-intensive tasks?**

RAG (Retrieval-Augmented Generation) combines a retrieval step — finding relevant documents from a knowledge store — with a generation step where an LLM uses those documents as context to answer a question. It is preferred over fine-tuning when: (a) the knowledge changes frequently (fine-tuning is a one-time bake-in), (b) you need source attribution (retrieved chunks can be cited), (c) the knowledge base is too large to fit in the model's weights, or (d) you need to reduce hallucinations by grounding the LLM in verifiable text. Fine-tuning is better for teaching the model a new *style* or *skill*, not for injecting factual knowledge.

**Fine-tuning — detailed explanation**

Fine-tuning takes a pre-trained base model (e.g. `llama-3-8b`) and continues training it on a curated dataset of your own examples. The model's weights are updated via gradient descent on your data. After fine-tuning, the knowledge is baked into the weights permanently — no prompt needed at inference time.

Two main variants:

- **Full fine-tuning**: all weights updated. Requires the same GPU RAM as pre-training — expensive and rare outside large labs.
- **Parameter-efficient fine-tuning (PEFT) / LoRA**: only small adapter matrices are trained while the base model is frozen. LoRA injects low-rank matrices `A` and `B` into attention layers, training <1% of total parameters. This is what most practitioners mean when they say "fine-tuning" today.

**Where fine-tuning shines:**

| Use case | Why fine-tuning, not RAG |
|---|---|
| Style / tone / persona | Teaching a specific voice (legal writing, medical notes, brand tone). RAG cannot change *how* the model writes. |
| Output format | Structured JSON schemas, SQL dialects, ICD-10 codes — tasks with a fixed input→output shape. |
| Domain vocabulary | Medical/legal/financial jargon where the base model consistently uses lay terms or gets terminology wrong. Fine-tuning shifts the token probability distribution. |
| Low-latency inference | A fine-tuned 7B model can outperform a large base model + RAG on a narrow task with zero retrieval overhead. |
| Reducing prompt length | 50-line system prompt instructions can be baked into weights, cutting cost and latency per request. |
| Safety / refusal tuning | Teaching domain-specific refusals beyond what RLHF provides. |

**Why RAG beats fine-tuning for knowledge:**

1. **Freshness** — fine-tuning is a snapshot; RAG just re-ingests the updated document.
2. **Hallucination grounding** — fine-tuned models still hallucinate facts; RAG gives the model the actual text to quote from.
3. **Source attribution** — RAG returns `"source: team-handbook.pdf"`. A fine-tuned model cannot tell you where it learned something.
4. **Scale** — you cannot fine-tune millions of documents into a 7B model without catastrophic forgetting. RAG scales to arbitrary corpus size.
5. **Cost** — a LoRA fine-tune costs $10–$200 and hours of GPU time. RAG ingestion costs pennies.

**The power combo:** fine-tune for style/format/vocabulary + RAG for factual knowledge. Example: a medical assistant fine-tuned to output structured SOAP notes + RAG over the patient's chart for factual grounding.

**This project:** RAG only — the LLM is used as-is. The NeuralFlow AI knowledge base changes frequently enough that baking it into weights would be impractical, and source attribution matters for a Q&A system.

<a id="q2"></a>
**Q2. What are the main failure modes of a naive RAG pipeline?**

- **Recall failure**: the relevant chunk is not retrieved at all — wrong embedding model, poor chunking, or the query phrasing differs too much from the document.
- **Precision failure**: the retrieved chunks are topically related but don't contain the answer — results look relevant but are useless.
- **Lost-in-the-middle**: when multiple chunks are stuffed into the LLM context, the model attends poorly to chunks in the middle of a long context window.
- **Chunk boundary mismatch**: a sentence is split across two chunks; neither chunk individually answers the question.
- **Stale index**: documents updated on disk but not re-ingested; the LLM answers from old data.

<a id="q3"></a>
**Q3. What is the difference between standard RAG and agentic RAG?**

Standard RAG is a hardwired pipeline: embed query → retrieve → stuff context → generate. The retrieval always happens regardless of whether the question needs it. Agentic RAG gives the LLM retrieval as a *tool* it can choose to call, with control over the query string and number of results. This project uses agentic RAG: the Pydantic AI agent has a `search_knowledge_base` tool (`rag_agent.py`) and decides when to call it. The agent can also decline to retrieve if the question is trivially answerable. It is "lightweight agentic" — one retrieval tool, no multi-hop planning loops.

**Standard RAG — how it works:**

```
User query
    │
    ▼
Embed query  →  Vector search  →  Top-K chunks  →  Stuff into prompt  →  LLM  →  Answer
```

Every step is fixed and runs unconditionally. The system has no judgement — it always retrieves, always uses exactly K chunks, always calls the LLM once. There is no loop, no decision point, no ability to follow up.

Failure modes this causes:
- Retrieves chunks even for "What is 2+2?" — wastes latency and tokens
- Uses a fixed query string (the raw user question) even if it's vague or ambiguous
- Cannot realise mid-generation that it needs more information and go back to retrieve

**Agentic RAG — how it works:**

```
User query
    │
    ▼
LLM (agent) ──► decides: do I need to search?
    │                        │
    │          Yes           │  No
    │◄────────────────────   └──► answer directly
    │
    ▼
search_knowledge_base(query="refined query", count=5)
    │
    ▼
LLM reads results ──► decides: is this enough?
    │                              │
    │  No (needs more)             │  Yes
    ▼                              ▼
search_knowledge_base(...)      Generate final answer
    │
    ▼
... (loop until satisfied or max iterations)
```

Key differences:

| Dimension | Standard RAG | Agentic RAG |
|---|---|---|
| **Retrieval decision** | Always retrieves | LLM decides whether to retrieve |
| **Query formulation** | Raw user question | LLM rewrites query for better retrieval |
| **Number of retrievals** | Exactly 1 | 0, 1, or many — LLM decides |
| **Multi-hop** | No | Yes — can search → read → search again |
| **Tool choice** | Only retrieval | Can have multiple tools (web search, calculator, DB lookup) |
| **Latency** | Predictable (always 1 retrieval) | Variable (0–N retrievals) |
| **Cost** | Predictable | Variable — more LLM calls |

**Multi-hop example:** User asks *"Who is the manager of the team that owns the billing service?"*
- Standard RAG: embeds the full question, retrieves chunks, hopes the answer is in one chunk.
- Agentic RAG: searches "billing service owner" → finds team name → searches "team X manager" → finds the person. Two hops, two retrievals.

**How Pydantic AI implements the tool loop:**

The agent runs in a loop internally. After each LLM response, Pydantic AI checks if the model emitted a tool call. If yes, it executes the tool (`search_knowledge_base`), appends the result to the message history, and calls the LLM again. This continues until the LLM produces a plain text response with no tool calls — that becomes the final answer. The loop is bounded by `max_result_retries` to prevent infinite loops.

```python
# rag/agent/rag_agent.py
@agent.tool
async def search_knowledge_base(ctx: RunContext[RAGState], query: str, count: int = 5) -> str:
    results = await ctx.deps.retriever.retrieve(query, match_count=count)
    # returns formatted chunk text → LLM reads it and decides what to do next
```

**This project's flavour — "lightweight agentic":**

One tool, no multi-hop planning, no separate planner LLM. The agent can choose *not* to search (for greetings, math questions) and can search with a rewritten query, but it does not chain multiple searches in a reasoning loop in practice. This keeps latency predictable while still getting the benefits of dynamic query formulation and skip-retrieval for trivial questions.

<a id="q4"></a>
**Q4. How does chunking strategy affect retrieval quality?**

Smaller chunks → higher precision (each chunk is tightly scoped) but lower recall (context that spans multiple chunks is split). Larger chunks → more context per result but noisier embeddings (the embedding averages over more text, diluting the signal). For this project, `max_tokens=512` is the hard ceiling set by the embedding model's window. The HybridChunker respects structural boundaries (sections, paragraphs) rather than splitting at an arbitrary character count, which improves coherence without sacrificing precision.

<a id="q5"></a>
**Q5. What is the "lost in the middle" problem and how does chunk ordering help?**

LLMs attend more strongly to tokens near the beginning and end of the context window, and less to tokens in the middle. When 5 chunks are formatted into context, the chunk at position 3 is most likely to be ignored. The mitigation in this codebase is to return ranked results (highest similarity first) so the most relevant chunk is always at position 1, not buried in the middle.

**Why it happens:** Transformer attention is not uniform across the context window. Research (Liu et al., 2023 — "Lost in the Middle") showed that LLMs pay the most attention to the **beginning** (primacy effect) and **end** (recency effect) of the context. Tokens in the middle receive significantly less attention weight.

**Concrete example:** Say you retrieve 5 chunks and stuff them into the prompt:

```
[CHUNK 1] — about PTO policy (rank #1, most relevant)
[CHUNK 2] — about sick leave
[CHUNK 3] — about the actual answer to the question  ← buried in the middle
[CHUNK 4] — about health benefits
[CHUNK 5] — about holidays
```

The user asks: *"How many vacation days do employees get?"* The exact answer is in chunk 3, but the model attends poorly to it and may hallucinate — even though the correct text is in its context.

The problem gets worse when you retrieve many chunks (k=10, 20) and when your retriever is imperfect (the real answer lands at rank #4 or #5 instead of #1).

**Mitigations:**

| Technique | How it helps |
|---|---|
| Rank-order the context (this project) | Most relevant chunk at position 1 → top of context where attention is highest |
| Reduce k | Fewer chunks → less "middle" — but risks missing the answer if it's at rank #6 |
| Reranker | Cross-encoder re-scores chunks more accurately → truly relevant chunk lands at rank #1 |
| "Sandwich" ordering | Most relevant chunks at top AND bottom; less relevant in the middle |
| Smaller context | Keep total context short so there's less "middle" to lose things in |

**This project:** The retriever returns results sorted by RRF score (highest first). The agent formats them in that order, so if retrieval is correct the answer is always near the top. The reranker (when enabled) improves this further by making rank ordering more accurate.

<a id="q6"></a>
**Q6. Why store the full document alongside chunks?**

The `documents` table holds the full text and metadata, while `chunks` holds the searchable fragments. This allows: (a) re-chunking without re-ingesting the source file (just re-process the stored content), (b) displaying the source document to users, (c) computing the content hash for incremental ingestion without re-reading the file, and (d) cascading deletes — `ON DELETE CASCADE` removes all chunks when the parent document is deleted.

---

## Hybrid Search & RRF

<a id="q7"></a>
**Q7. Explain hybrid search. What problem does each leg solve?**

Semantic (vector) search embeds the query and finds chunks whose embeddings are close in vector space. It handles vocabulary mismatch — "compensation" matches "salary" — but struggles with exact terms, acronyms, and proper nouns. Text search (tsvector) matches exact lexemes and is ideal for keywords like "PTO", "llama3", "NeuralFlow" but fails when query and document use different words for the same concept. Hybrid search runs both in parallel (`asyncio.gather`) and merges the ranked lists with RRF, rewarding chunks that appear high in both lists.

<a id="q8"></a>
**Q8. Walk through RRF. What is the formula and what does k=60 do?**

For each chunk, RRF assigns a score from each ranked list:

```
rrf_score(rank) = 1 / (k + rank)
```

The final score is the sum across all lists. k=60 is a smoothing constant that prevents a rank-1 result from dominating completely (1/61 ≈ 0.016 vs 1/1 = 1.0 without smoothing). It was empirically shown in the original RRF paper to work well across diverse datasets. The effect: a chunk ranked #1 in semantic and #5 in text gets a combined score of 1/61 + 1/65 ≈ 0.032, which beats a chunk ranked #1 in only one list.

<a id="q9"></a>
**Q9. When does text search win over semantic search?**

When the query contains exact tokens that appear verbatim in the document. Examples from this corpus: "PTO" (an acronym that embeddings might not distinguish from "PT"), "NeuralFlow" (a proper noun), "llama3" (a model name). The test results confirm this: text search has lower overall Hit Rate@5 (0.40) but for queries like "What is DocFlow AI" it finds the audio transcription that mentions "DocFlow" by exact token match.

<a id="q10"></a>
**Q10. When does semantic search win?**

When query and document use different vocabulary for the same concept. Example: querying "company culture and values" matches a document that uses "core principles" and "work environment" — no shared keywords, but the embedding vectors are close. Semantic Hit Rate@5 = 0.90 on this corpus vs text = 0.40, showing it is the stronger leg for conceptual queries.

<a id="q11"></a>
**Q11. Semantic Hit Rate@5 = 0.90 vs hybrid = 0.80. How do you explain this?**

*Note: this gap existed before audio transcription was fixed. After re-ingesting with Whisper, both hybrid and semantic reach 0.90 — see Q69a for current results.*

RRF merges the two ranked lists. If a chunk ranks #1 in semantic but is not in the text results at all (Hit Rate@5=0.80 vs 0.90 means 1 query that semantic hits but hybrid misses), the RRF score may push another chunk above it. Specifically, a chunk that ranks moderately in *both* lists gets a higher combined RRF score than a chunk that ranks #1 in only one list. The "company mission and vision" query previously missed in hybrid — `mission-and-goals.md` ranks high semantically but doesn't contain strong keywords, so the text leg contributes nothing and a different, keyword-rich chunk edges ahead after RRF. After Whisper re-ingestion added audio transcript content, RRF rankings shifted and this query now hits at rank 1 in hybrid. Fix if it regresses: increase `match_count` (fetch more candidates before RRF) or tune the k constant.

<a id="q12"></a>
**Q12. If you had to drop one leg, which would you keep?**

Semantic search. It handles the majority of query types (conceptual, paraphrased, vocabulary-mismatch). Text search is critical for exact terms and acronyms, but those cases can be partially mitigated with a better embedding model. The converse is not true — you cannot fix vocabulary mismatch with keyword search.

---

## PostgreSQL / pgvector

<a id="q13"></a>
**Q13. Why PostgreSQL over a dedicated vector DB?**

This system already needs PostgreSQL for relational data (documents, chunks, metadata, Mem0 memory). Adding a separate vector DB means two infrastructure components to manage, two connection pools, and a JOIN across network boundaries to correlate chunks with document metadata. PostgreSQL + pgvector handles both in a single query with a JOIN. The trade-off is that pgvector's IVFFlat index is less scalable than purpose-built ANN indexes (HNSW in Pinecone/Weaviate) at hundreds of millions of vectors, but for RAG workloads in the tens-of-thousands range it is entirely adequate.

<a id="q14"></a>
**Q14. What is IVFFlat and how does it trade accuracy for speed?**

IVFFlat (Inverted File Flat) divides the vector space into `lists` Voronoi cells. At index time, each vector is assigned to its nearest centroid. At query time, only `probes` cells are searched rather than the full table. This reduces the search space from O(n) to O(n/lists × probes) but may miss true nearest neighbours that fall in unprobed cells (approximate, not exact). `lists = sqrt(n_rows)` is the standard recommendation. Increasing `probes` raises recall but also latency.

<a id="q15"></a>
**Q15. What does `register_vector` do and why in `init=`?**

asyncpg doesn't know how to serialize/deserialize the `vector` type from pgvector by default. `register_vector` installs custom codecs for `vector` ↔ Python `list[float]`. It must run in the `init` callback because that callback fires once for *each new connection* the pool creates. If you call it once after pool creation, it only registers on the single connection you happen to have at that moment; subsequent connections created by the pool won't have the codec.

<a id="q16"></a>
**Q16. Why `executemany` for batch inserts?**

`executemany` sends a single prepared statement to PostgreSQL and batches the parameter rows, which is significantly faster than N separate `INSERT` statements (N round-trips vs 1). For a document with 20 chunks, this reduces network overhead by 95%. The alternative `COPY` would be even faster for bulk loads but is more complex to use with asyncpg and embeddings.

<a id="q17"></a>
**Q17. `ON DELETE CASCADE` — what does it do and why is it critical?**

When a row in `documents` is deleted, PostgreSQL automatically deletes all rows in `chunks` where `document_id` matches. Without it, deleting a document during re-ingestion would leave orphaned chunks in the DB — chunks with no parent document, wasting storage and polluting search results with unreachable content. The pipeline's `delete_document_and_chunks()` method relies on this: it only needs to delete the document row and the database handles chunk cleanup.

<a id="q18"></a>
**Q18. Why UUID primary keys over auto-increment?**

Auto-increment integers are sequential and predictable (an attacker who gets chunk ID 100 knows IDs 1–99 exist). UUIDs are random, unpredictable, and globally unique — safe to expose in APIs. They also work correctly in distributed/multi-node settings where two nodes generating IDs simultaneously would collide with integer sequences. `gen_random_uuid()` runs inside PostgreSQL so no application-side UUID generation is needed.

<a id="q19"></a>
**Q19. `GENERATED ALWAYS AS (...) STORED` — what does this mean?**

It is a PostgreSQL *generated column*. The value of `content_tsv` is automatically computed by PostgreSQL as `to_tsvector('english', content)` whenever a row is `INSERT`ed or `UPDATE`d. `STORED` means the computed value is written to disk (not recomputed at query time). You never write to this column manually — PostgreSQL enforces this (`GENERATED ALWAYS` prevents explicit writes). On `UPDATE` to `content`, the column is recalculated automatically.

<a id="q19b"></a>
**Q19b. How do I install `psql` and connect to the Neon database?**

`psql` is the PostgreSQL command-line client. It is not bundled with Windows — it must be installed separately.

**Installation (Windows)**

Option 1 — Chocolatey:
```bash
choco install postgresql -y
```

Option 2 — Winget:
```bash
winget install PostgreSQL.PostgreSQL
```

Option 3 — Manual: download the "Command Line Tools" zip from postgresql.org/download/windows and extract `psql.exe` plus its `lib/` folder.

After installation, add the bin directory to your PATH permanently (run in PowerShell):
```powershell
[System.Environment]::SetEnvironmentVariable("PATH", [System.Environment]::GetEnvironmentVariable("PATH", "User") + ";C:\Program Files\PostgreSQL\18\bin", "User")
```
Open a new terminal for the change to take effect. Verify with `psql --version`.

**Connecting to Neon**

```bash
psql "postgresql://<user>:<password>@<host>/neondb?sslmode=require"
```

Use the connection string from your `.env` file (`DATABASE_URL`). The version mismatch warning (`psql 18.x, server 17.x`) is harmless — the client is newer than the server.

**Useful one-liners**

```bash
# Check PostgreSQL version
psql "$DATABASE_URL" -c "SELECT version();"

# List available extensions
psql "$DATABASE_URL" -c "SELECT name, installed_version FROM pg_available_extensions WHERE name IN ('vector', 'pg_trgm', 'pg_search');"

# Check table row counts
psql "$DATABASE_URL" -c "SELECT 'documents' AS tbl, COUNT(*) FROM documents UNION ALL SELECT 'chunks', COUNT(*) FROM chunks;"
```

**Alternative: Neon MCP tools**

If you are running inside Claude Code, the Neon MCP server is available and can run SQL queries against your project directly without `psql` or any local installation.

---

## Full-Text Search

<a id="q20"></a>
**Q20. What is a `tsvector` and how does it differ from the original text?**

A `tsvector` is not the original text — it is a sorted, de-duplicated list of *lexemes* with position tags. Three transformations happen: (1) stop words are removed ("the", "is", "a"), (2) remaining words are stemmed to their root form ("employees" → `employe`, "entitled" → `entitl`), (3) each lexeme is tagged with its position(s) in the original text (for phrase queries). Example: `to_tsvector('english', 'The employees are entitled to PTO')` → `'employe':2 'entitl':4 'pto':6`.

<a id="q21"></a>
**Q21. What does stemming do and when can it cause false positives?**

Stemming reduces word variants to a common root so queries match all inflections. "run", "running", "runs" all become `run`. False positive example: "university" and "universe" both stem to `univers` in some stemmers, so a query for "universe" could match a document about a university. In this codebase, "PTO" and "PT" would both become `pt` — a query for "PT" (physical therapy) could match PTO documents.

<a id="q22"></a>
**Q22. Why `plainto_tsquery` instead of `to_tsquery` for user input?**

`to_tsquery` requires the user to supply valid tsquery syntax (`'pto & policy'`). If a user types `'PTO policy?'` the `?` causes a parse error. `plainto_tsquery` takes raw prose, tokenizes it, and ANDs the non-stop-word lexemes. It never throws a syntax error on user input, making it safe for direct use without sanitisation.

<a id="q23"></a>
**Q23. Why is a GIN index better than B-tree for tsvector?**

A B-tree index works on ordered scalar values (numbers, strings with natural ordering). A `tsvector` is a set of lexemes — there is no natural ordering of the whole vector. A GIN (Generalized Inverted Index) is an inverted index: for each lexeme, it stores the list of rows containing that lexeme. The `@@` operator can look up each lexeme in the tsquery directly in the index rather than scanning every row.

<a id="q24"></a>
**Q24. "30 days PTO" — the number 30 is dropped. Why? How would you handle numeric search?**

Numbers are stop words under the `'english'` configuration. `to_tsvector('english', '30 days PTO')` → `'day':2 'pto':3`. If you need number search, use `'simple'` configuration (no stemming, no stop words) for a second tsvector column, or store structured numeric fields separately and search them with standard SQL comparisons.

<a id="q25"></a>
**Q25. What happens when `plainto_tsquery` produces an empty query?**

If all query words are stop words (e.g. "what is the"), `plainto_tsquery` returns an empty `tsquery`. The `@@` operator against an empty tsquery returns `false` for every row, so the text search leg returns 0 results. In the hybrid search, this means only the semantic leg contributes. The codebase handles this gracefully because both searches run in parallel and the RRF merger works fine with one empty result list — it just returns the semantic results ordered by their semantic rank.

---

## Chunking & DoclingHybridChunker

<a id="q26"></a>
**Q26. What does Docling's `HybridChunker` do that sliding-window cannot?**

Sliding-window splits at fixed character counts with no awareness of document structure. It can cut mid-sentence, mid-table, or mid-code-block. `HybridChunker` operates on the structured `DoclingDocument` produced by `DocumentConverter`, which knows about section headings, paragraph boundaries, table cells, lists, and code blocks. It splits at structural boundaries first (end of a section, paragraph break) and only uses token limits as a hard ceiling. The result is chunks that are semantically coherent units.

<a id="q27"></a>
**Q27. What is `contextualize()` and why does it improve embedding quality?**

`contextualize(chunk)` prepends the heading hierarchy to the chunk's body text. For example, a chunk under "## Benefits > ### PTO" about time-off details becomes: `"Benefits > PTO\n\nEmployees receive 20 days of PTO per year..."`. Without this, the chunk reads "Employees receive 20 days per year..." with no indication of what "days" refers to. The embedding of the contextualized chunk is more specific and matches queries like "PTO policy" better because the topic is explicit in the text being embedded.

**Where is it called?** `rag/ingestion/chunkers/docling.py:191`, inside `DoclingHybridChunker.chunk_document()`. The loop iterates over structural chunks produced by Docling's `HybridChunker.chunk()`, then calls `self.chunker.contextualize(chunk=chunk)` on each — `self.chunker` is the Docling `HybridChunker` instance. The returned contextualized text is what gets embedded and stored, not the raw chunk text. Call chain: `pipeline.py` → `chunker.chunk_document()` → `docling.py:191` → Docling's `HybridChunker.contextualize()`.

<a id="q28"></a>
**Q28. Describe the fallback chunking path exactly.**

Triggered when `docling_doc=None` (plain text, `.txt` files, or conversion failure). The `_simple_fallback_chunk` method (`chunkers/docling.py:228`) uses a sliding window: start at position 0, set `end = start + chunk_size`. It then walks backwards from `end` up to `max(start + min_chunk_size, end - 200)` looking for a sentence boundary (`.`, `!`, `?`, `\n`). If found, it cuts there; otherwise cuts at `end`. The next window starts at `end - overlap` (overlap = 100 chars by default). Token count is computed with the same HuggingFace tokenizer. The `chunk_method` metadata field is set to `"simple_fallback"` so you can distinguish these at query time.

<a id="q29"></a>
**Q29. Why is `DocumentConverter` cached via `_get_converter()`?**

`DocumentConverter` loads several ML models on first instantiation — layout detection, table structure recognition, equation parsing — which takes several seconds and significant memory. Caching it means the cost is paid once per pipeline instance, not once per document. For a batch of 13 documents (this corpus), that's 12 avoided re-initializations. The cache is an instance variable (`_doc_converter`) so it's garbage collected when the pipeline is closed.

<a id="q30"></a>
**Q30. What is `merge_peers=True`?**

When HybridChunker splits a document, it sometimes produces adjacent small chunks that are "peers" — they belong to the same structural level (e.g. consecutive short paragraphs under the same heading). `merge_peers=True` joins these small siblings into a single chunk if the combined token count stays under `max_tokens`. This reduces the number of very short chunks (which have poor embedding signal) and ensures each chunk has sufficient context to be meaningful.

<a id="q31"></a>
**Q31. Why cache `DocumentConverter` as an instance attribute, not a module-level singleton?**

A module-level singleton would be shared across all `DocumentIngestionPipeline` instances (e.g. in tests). Different pipelines might be configured differently. More importantly, during tests each test creates and tears down its own pipeline, and a singleton would leak state across tests. Instance-level caching gives lifetime tied to the pipeline object, which is correct.

---

## Embeddings

<a id="q32"></a>
**Q32. What does `nomic-embed-text` produce and why 768 dimensions?**

`nomic-embed-text` is a general-purpose text embedding model optimized for retrieval, producing 768-dimensional dense vectors. 768 is a common embedding size (BERT-base is also 768). Higher dimensions capture more nuance but increase storage (768 × 4 bytes = 3KB per chunk) and slow down vector similarity computation. For this corpus size the trade-off is fine.

<a id="q33"></a>
**Q33. Cosine similarity vs Euclidean distance — why cosine?**

Cosine similarity measures the angle between vectors, ignoring magnitude. Two texts with the same meaning but different lengths produce vectors pointing in the same direction but at different magnitudes (longer text → larger magnitude). Cosine similarity normalises this away. Euclidean distance treats magnitude differences as semantic differences, which is wrong for text embeddings. pgvector uses `<=>` for cosine distance (`1 - cosine_similarity`).

<a id="q34"></a>
**Q34. The embedder has an in-memory cache — what is the cache keyed on and what are its limits?**

The cache key is the query string (exact text match). This is appropriate for the retriever's query embedding (the same user question typed twice). Limits: (1) it is in-process memory — lost on restart; (2) no eviction policy visible in the code, so it grows unbounded; (3) it only helps for repeated identical queries, not paraphrased queries. In a long-running service, this could cause a memory leak for a large query vocabulary.

<a id="q35"></a>
**Q35. Switching from nomic-embed-text (768-dim) to text-embedding-3-small (1536-dim) — what changes?**

- `EMBEDDING_DIMENSION=1536` in `.env`
- Drop and recreate the `chunks` table (column type changes: `vector(768)` → `vector(1536)`)
- Recreate the IVFFlat index with the new dimension
- Re-ingest all documents (old embeddings are incompatible)
- Update `EMBEDDING_MODEL` and `EMBEDDING_BASE_URL` / `EMBEDDING_PROVIDER`
- The `register_vector` call handles any dimension, so no code change there

<a id="q36"></a>
**Q36. Symmetric vs asymmetric embedding models — which for RAG?**

Symmetric models produce embeddings where query and document live in the same space — comparing a short query to a short sentence. Asymmetric models (like `nomic-embed-text` with `search_query:` / `search_document:` prefixes, or `e5-` models) are trained on (query, passage) pairs where queries are short and documents are long. Asymmetric is more appropriate for RAG because queries and chunks are structurally different — you want the model to understand "query intent" vs "document content". `nomic-embed-text` supports this via instruction prefixes.

---

## HyDE

<a id="q37"></a>
**Q37. Explain HyDE. Why might it outperform raw query embedding?**

HyDE (Hypothetical Document Embeddings, Gao et al. 2022): instead of embedding the raw query ("What is the PTO policy?"), the LLM generates a *hypothetical answer* ("NeuralFlow AI provides 20 days of PTO per year with a 15-day minimum..."), and *that text* is embedded. The intuition: the hypothetical answer is structurally similar to the actual document chunk — same vocabulary, same style. The embedding of the hypothetical answer therefore sits closer in vector space to real chunks than the embedding of a question. It effectively bridges the query-document vocabulary gap.

<a id="q38"></a>
**Q38. What are the risks of HyDE?**

- **Hallucination propagation**: if the LLM generates a plausible-but-wrong hypothetical ("30 days PTO"), the embedding drifts toward chunks about vacation rather than the specific policy document.
- **Added latency**: one extra LLM call before retrieval.
- **Cost**: one LLM API call per query.
- **Worse for factual queries**: when the LLM has no relevant prior knowledge, the hypothetical can be completely off.

<a id="q39"></a>
**Q39. When would you enable HyDE?**

When queries are highly conceptual or domain-specific and the vocabulary gap between queries and documents is large. Good candidates: legal documents (users ask in plain English, documents use legal terminology), medical records, technical patents. Not worth enabling for this corpus where queries are already close to the document language.

<a id="q40"></a>
**Q40. How is HyDE implemented in the retriever?**

In `retriever.py`, if `settings.hyde_enabled` is True, before the search step the retriever calls `_get_hyde()` (lazy init) to get a `HyDEGenerator` instance. It calls `hyde.generate(query)` which makes an LLM API call to get a hypothetical document, then embeds that hypothetical text instead of the raw query. The rest of the pipeline (search, rerank, cache) is unchanged.

---

## Reranking

<a id="q41"></a>
**Q41. What problem does a cross-encoder solve that bi-encoder retrieval cannot?**

Bi-encoders embed query and document independently — they cannot compare them token-by-token. A cross-encoder takes the (query, document) pair concatenated as input and scores relevance jointly with full cross-attention. This is much more accurate because the model sees both texts simultaneously and can model fine-grained interactions ("the document mentions '20 days' in the context of vacation, which matches the query about PTO"). The cost is O(n) cross-encoder calls where n = candidate count.

<a id="q42"></a>
**Q42. LLM reranker vs CrossEncoder — trade-offs?**

| | LLM reranker | CrossEncoder |
|---|---|---|
| Model | Remote LLM API (same as generation) | Local `sentence-transformers` model |
| Quality | High (powerful model) | High (specialized for reranking) |
| Latency | ~1s per chunk via API | ~100ms per chunk, local |
| Cost | API cost per chunk scored | Hardware cost only |
| Privacy | Chunks sent to API | Data stays local |

LLM reranker is the default because it requires no additional model deployment. CrossEncoder is better for latency-sensitive or privacy-sensitive deployments.

<a id="q43"></a>
**Q43. Why `asyncio.gather` for LLM reranker scoring?**

Each chunk scoring is an independent LLM API call. `asyncio.gather` fires all calls concurrently, so n chunks take approximately the time of one call (network-bound), not n × one call (sequential). Without it, reranking 10 candidates at 500ms each = 5 seconds; with `asyncio.gather` ≈ 500ms.

<a id="q44"></a>
**Q44. At what corpus size or query volume would you enable the reranker?**

Enable when: (a) the corpus is large enough that top-K retrieval frequently returns marginally relevant results (usually >50K chunks), or (b) precision matters more than latency (e.g. agent needs exactly the right chunk to answer a specific factual question). For this 13-document, ~150-chunk corpus the retrieval precision is already high and reranking adds latency for marginal gain.

<a id="q45"></a>
**Q45. Retrieval recall vs reranking precision — how do they compose?**

First stage (retrieval): maximize recall — use `match_count * reranker_overfetch_factor` to fetch more candidates than needed. If recall is 80% at K=5 but 95% at K=20, over-fetch to K=20. Second stage (reranker): maximize precision — take the top-K of the 20 reranked candidates. The two-stage pipeline lets you optimize each stage independently. If the first stage misses the relevant chunk (recall failure), no reranker can recover it — this is why recall in retrieval is the first thing to fix.

---

## Agentic RAG & Pydantic AI

<a id="q46"></a>
**Q46. What makes this system "agentic"?**

The LLM (via Pydantic AI) decides *whether* to call retrieval, *what query* to use, and *how many results* to fetch. In a standard RAG pipeline these decisions are hardwired. The agent also has access to a separate `Mem0` memory store and can combine retrieved chunks with user history. It is lightweight agentic — one retrieval tool, one memory tool, no multi-step planning — but the control flow is driven by the model.

<a id="q47"></a>
**Q47. How does Pydantic AI's tool system work?**

The `@agent.tool` decorator registers a Python async function as a tool available to the model. Pydantic AI serializes the function signature (name, parameters, docstring) into the tool schema and includes it in the system prompt / tool list sent to the LLM. When the LLM outputs a tool call (in its structured response), Pydantic AI deserializes the arguments, calls the Python function, and feeds the return value back to the LLM as a tool result. Type annotations are used for the schema — changing a parameter type changes what the LLM knows about the tool.

<a id="q47a"></a>
**Q47a. How does the LLM know which tool to call — does Pydantic AI register tool names with it?**

Yes — this is the core of how tool-calling LLMs work. Pydantic AI introspects the decorated function and builds a JSON Schema describing the tool, then sends it to the LLM as part of the API request. The LLM never "learns" about tools through training on your code — it receives them fresh on every request.

**Step 1 — Pydantic AI introspects the function:**

```python
@agent.tool
async def search_knowledge_base(
    ctx: PydanticRunContext,
    query: str,
    match_count: int | None = 5,
    search_type: str | None = "hybrid",
) -> str:
    """Search the knowledge base for relevant information."""
    ...
```

Pydantic AI reads: the function name (`search_knowledge_base`), every parameter (excluding `ctx`), their types, their defaults, and the docstring. It converts these into a JSON Schema object:

```json
{
  "name": "search_knowledge_base",
  "description": "Search the knowledge base for relevant information.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": { "type": "string" },
      "match_count": { "type": "integer", "default": 5 },
      "search_type": { "type": "string", "default": "hybrid" }
    },
    "required": ["query"]
  }
}
```

**Step 2 — This schema is sent to the LLM API on every request:**

For OpenAI-compatible APIs (which this project uses via `OpenAIChatModel`), the schema is passed in the `tools` field of the chat completion request:

```json
POST /v1/chat/completions
{
  "model": "llama3.1:8b",
  "messages": [...],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "search_knowledge_base",
        "description": "Search the knowledge base...",
        "parameters": { ... }
      }
    }
  ],
  "tool_choice": "auto"
}
```

**Step 3 — The LLM decides whether to call the tool:**

The LLM has been fine-tuned (via RLHF/instruction tuning) to understand the `tools` field. Based on the user message, the system prompt, and the tool descriptions, it decides:
- If the question needs information from the knowledge base → emit a tool call response
- If the question is trivial (greeting, math) → emit a plain text response directly

The system prompt in `prompts.py` provides additional guidance:
```
ONLY search when users explicitly ask for information that would be in the knowledge base
For greetings (hi, hello) -> Just respond conversationally, no search needed
```

**Step 4 — The LLM emits a structured tool call:**

Instead of plain text, the LLM responds with:
```json
{
  "role": "assistant",
  "tool_calls": [{
    "id": "call_abc123",
    "type": "function",
    "function": {
      "name": "search_knowledge_base",
      "arguments": "{\"query\": \"PTO policy\", \"match_count\": 5, \"search_type\": \"hybrid\"}"
    }
  }]
}
```

Note the LLM fills in the argument values — it chose `"PTO policy"` as the query string, not the user's original phrasing.

**Step 5 — Pydantic AI executes the tool and loops:**

Pydantic AI catches this response, validates the arguments against the JSON Schema, calls the Python function, and appends the result to the message history:

```json
{
  "role": "tool",
  "tool_call_id": "call_abc123",
  "content": "Source: team-handbook | Title: Employee Benefits\n\nEmployees receive 20 days of PTO per year..."
}
```

The updated message history is sent back to the LLM for a second call. Now the LLM has the retrieved context and generates the final answer as plain text — no more tool calls.

**Why the tool name and docstring matter so much:**

The LLM's decision to call `search_knowledge_base` (and with what arguments) is entirely driven by the tool's `name` and `description`. If the function were named `do_thing` with no docstring, the LLM would rarely call it. Good tool design:
- Name: verb + noun, self-explanatory (`search_knowledge_base` not `skb`)
- Description (docstring): explain *what*, *when to use it*, and what it returns
- Parameter descriptions: type annotations + defaults give the LLM strong hints

<a id="q48"></a>
**Q48. What is `RAGState` and why are its attributes `PrivateAttr`?**

`RAGState` is the dependency injection container passed as `deps` to every tool call. It holds the `user_id`, `store`, `retriever`, and `mem0_store`. These are declared as `PrivateAttr(...)` because `RAGState` extends `BaseModel` — regular fields would be included in Pydantic's schema/validation/serialization, which is wrong for internal service objects. `PrivateAttr` tells Pydantic "this field exists but is not part of the data model."

<a id="q49"></a>
**Q49. Why `ContextVar` for Langfuse trace context?**

In async Python, multiple coroutines run concurrently on the same thread. A class-level attribute like `_current_trace = None` is shared across all concurrent requests — request A's trace would overwrite request B's. `ContextVar` is Python's mechanism for per-coroutine (async task) local storage. Each concurrent `traced_agent_run` invocation gets its own trace reference that is invisible to all other concurrent invocations.

<a id="q50"></a>
**Q50. Why is per-user state important in a multi-user chat app?**

`RAGState(user_id=user_id)` is created once per conversation turn with the specific user's ID. This ID is used to look up Mem0 memories for that user only (`mem0_store.get_context_string(user_id)`). Without per-user state, all users would see the same memory context — a major privacy and correctness failure.

<a id="q51"></a>
**Q51. How does the agent handle tool call failures?**

The `search_knowledge_base` tool has a `try/except` that returns a formatted error string ("Error searching knowledge base: ...") rather than raising an exception. The LLM receives this error string as the tool result and is expected to gracefully inform the user that retrieval failed. The agent itself won't crash — Pydantic AI propagates the tool result back to the model regardless of whether it indicates success or failure.

---

## Memory (Mem0)

<a id="q52"></a>
**Q52. What problem does Mem0 solve that conversation history cannot?**

Conversation history is ephemeral — it's the message list for the current session. Mem0 persists semantic facts across sessions: "The user prefers detailed explanations", "User is in the engineering team", "User asked about PTO last week." When a user starts a new conversation, Mem0 provides relevant context from past interactions so the agent doesn't start from zero.

<a id="q53"></a>
**Q53. How is Mem0 stored in this project?**

Mem0 uses the same PostgreSQL database configured via `DATABASE_URL`. It creates its own tables (managed by the mem0 library, separate from `documents` and `chunks`). Memories are stored as text with embeddings, supporting vector similarity search to retrieve the most relevant past memories for a given query.

<a id="q54"></a>
**Q54. `add()` vs `get_context_string()` — difference?**

`add(user_id, messages)` takes the current conversation messages, extracts salient facts (via the LLM), and stores them in PostgreSQL for that user. `get_context_string(user_id)` retrieves the most relevant stored memories for that user and formats them as a single string ready to be injected into the system prompt. The agent calls `get_context_string` at the start of each turn and `add` at the end.

<a id="q55"></a>
**Q55. Why is Mem0 disabled by default?**

Mem0 requires an extra LLM call to extract memories from each conversation, adding latency and cost. It also requires the `mem0ai` package which has its own dependencies. For simple single-turn queries (most RAG use cases), it provides no benefit. It is valuable for multi-session, personalized assistants.

<a id="q56"></a>
**Q56. How would you prevent Mem0 from storing sensitive information?**

Options: (a) post-process extracted memories through a PII detection model before storage, (b) configure the memory extraction prompt to explicitly exclude personal details ("do not store names, ages, financial information"), (c) add a content filter in the `add()` wrapper that scans for SSN/credit card patterns before calling the underlying mem0 library.

---

## Async Python & Performance

<a id="q57"></a>
**Q57. Why must all I/O be async? What happens with a blocking call?**

Python's asyncio event loop is single-threaded. A blocking call (e.g. `time.sleep(1)`, `requests.get(url)`) blocks the *entire thread*, meaning no other coroutine can run while it's blocked. With `await asyncio.sleep(1)`, the event loop switches to other coroutines during the wait. A blocking DB call or HTTP call in an otherwise async service would serialize all requests, eliminating the concurrency benefit.

<a id="q58"></a>
**Q58. What is an asyncpg connection pool and why use it?**

A pool maintains a set of pre-established PostgreSQL connections ready to be borrowed. Creating a new TCP connection + TLS handshake + PostgreSQL authentication takes 50–200ms. With a pool, a request borrows an existing connection (~0ms), runs the query, and returns it. `asyncpg.create_pool(min_size=1, max_size=10)` keeps 1–10 connections alive, allowing up to 10 concurrent queries without queuing.

<a id="q59"></a>
**Q59. Maximum latency improvement from `asyncio.gather` on semantic + text search?**

If semantic search takes T_s and text search takes T_t, sequential execution takes T_s + T_t. `asyncio.gather` runs them concurrently, so total time ≈ max(T_s, T_t). Maximum improvement ≈ 50% when both take equal time. In practice, semantic search (vector cosine computation) is slower than text search (GIN index lookup), so the improvement is typically 30–40% — close to the dominant leg's latency.

<a id="q60"></a>
**Q60. Why `init=register_vector` rather than registering after pool creation?**

See Q15. When `init=register_vector` is passed to `asyncpg.create_pool`, asyncpg calls it with each newly created connection before adding it to the pool. If you instead call `await conn.fetch(...)` to register after the pool exists, you only register on the one connection in your hand. The pool creates additional connections lazily as load increases — those connections would not have the codec. The `init` callback guarantees every pooled connection is properly configured.

<a id="q61"></a>
**Q61. If `asyncio.gather` has two coroutines and one raises an exception — what happens?**

By default, `asyncio.gather` re-raises the first exception and cancels the other tasks (in Python 3.11+ with `return_exceptions=False`). In `postgres.py`, each search is wrapped in its own `try/except` that catches errors and logs them, returning an empty list. So both searches always return a list (possibly empty) and `gather` always completes. The RRF merger then works correctly on two lists, one possibly empty.

---

## Evaluation & Retrieval Metrics

<a id="q62"></a>
**Q62. Hit Rate@K vs Precision@K — when do they diverge?**

Hit Rate@K is binary: 1.0 if *any* relevant doc is in top-K, 0.0 if none. Precision@K is the fraction of returned results that are relevant. They diverge when: a query has multiple relevant documents and some are retrieved. Example — 1 relevant result out of 5: Hit Rate@5 = 1.0, Precision@5 = 0.2. You care about Precision when you're stuffing all K results into the LLM context (you don't want 4 out of 5 to be noise). You care about Hit Rate when you're reranking — as long as the relevant doc is in the candidate set, the reranker can surface it.

<a id="q63"></a>
**Q63. What does MRR measure that Hit Rate doesn't?**

MRR = mean of `1 / rank_of_first_relevant_result`. Example: Hit Rate@5 = 1.0 for both queries, but if query A's first relevant doc is rank 1 (MRR contribution = 1.0) and query B's is rank 5 (MRR contribution = 0.2), the average MRR = 0.6. Hit Rate would show both as 1.0 — misleading. MRR is better when you only show the user the top result, or when the LLM is most influenced by the first chunk in context.

<a id="q64"></a>
**Q64. Walk through the NDCG formula.**

```
DCG@K  = Σ rel_i / log2(i+2)   for i = 0..K-1
IDCG@K = Σ 1    / log2(i+2)   for i = 0..min(|relevant|, K)-1
NDCG@K = DCG@K / IDCG@K
```

`rel_i` ∈ {0, 1} (binary relevance). The denominator `log2(i+2)` is the position discount:
- Position 1: weight = 1/log2(2) = 1.0
- Position 2: weight = 1/log2(3) ≈ 0.63
- Position 5: weight = 1/log2(6) ≈ 0.39

IDCG is the DCG of the ideal ranking (all relevant docs at the top). Dividing by IDCG normalises to [0,1]. NDCG = 1.0 means all relevant docs appear before all irrelevant ones.

<a id="q65"></a>
**Q65. Is 10 queries a sufficient gold dataset?**

No — 10 queries gives high variance estimates. Changing one query outcome flips metrics by 10%. A production evaluation dataset should have 100–500 queries. To build it: (1) sample real user queries from logs, (2) manually annotate relevant documents for each, (3) use LLM-as-judge to scale annotation. The current gold dataset is appropriate for CI regression testing (did a code change break retrieval?) but not for publication-quality evaluation.

<a id="q66"></a>
**Q66. Why do "company mission and vision" and "DocFlow AI" miss consistently?**

*Current status (post-Whisper fix):* "company mission and vision" now **passes** (Hit ✓, RR=1.00). "DocFlow AI" still **fails** (Hit ✗, RR=0.00) — see below.

"Company mission and vision" — previously, `mission-and-goals.md` ranked well semantically but weakly in text search (generic language, no strong keywords), so after RRF a keyword-rich chunk from `company-overview.md` edged ahead. After re-ingesting with Whisper, the additional audio transcript content shifted RRF rankings enough that `mission-and-goals.md` now surfaces at rank 1. This was a side-effect fix, not a targeted one — the underlying fragility (thin keyword signal) remains and could regress on corpus changes.

"DocFlow AI" — content lives in `Recording2.mp3`. Whisper is now installed and the file was re-ingested, producing 1 chunk of transcript. However, the query still misses. The likely cause is that the Whisper transcription of `Recording2.mp3` does not use the exact phrase "DocFlow AI" prominently enough to surface in top-5 results for that query — the transcript may refer to it differently or the chunk is outscored by other documents that discuss document processing more generally. Next steps: inspect the actual transcript content in the DB (`SELECT content FROM chunks WHERE document_id IN (SELECT id FROM documents WHERE source ILIKE '%Recording2%')`), then either re-phrase the gold query to match the transcript, or expand `relevant_sources` to include additional source stems.

<a id="q67"></a>
**Q67. Recall@5 shows values above 1.0 — is that a bug?**

Not a bug in the metric code, but a limitation of the gold dataset definition. Recall@K = `relevant_found / total_relevant`. `total_relevant` is set to `len(entry["relevant_sources"])` — the number of *documents* in the relevant_sources list, not the number of *chunks* retrieved. When a relevant document has multiple chunks in top-K (e.g. 3 chunks from `team-handbook`), `relevant_found` = 3 but `total_relevant` = 1, giving Recall = 3.0.

This is now confirmed in practice: after Whisper re-ingestion, Recall@5 = **2.250** (up from 0.900 before). The audio transcripts added new chunks from documents already in the index, so several queries now retrieve multiple relevant chunks per relevant document. The fix is to count distinct relevant *documents* found in top-K rather than chunks — but the current values are not harmful to the CI gate since the threshold is 0.40 and any value above 0 satisfies it. Treat Recall in this codebase as a coarse "coverage" signal, not a precise fraction.

<a id="q68"></a>
**Q68. Why are unit tests and integration tests in the same file?**

The metric functions (`hit_rate`, `ndcg_at_k`, etc.) are directly imported by the integration tests. Keeping them co-located avoids a split where you'd need to import from a utility module. The `TestMetricFunctions` class has no async fixtures and runs in milliseconds — it acts as a correctness gate for the math before the expensive DB tests run. Separating them would add a module boundary with no organisational benefit.

<a id="q69"></a>
**Q69. How would you use these metrics to decide whether to enable HyDE or the reranker?**

Run the gold dataset with each configuration: baseline (off/off), HyDE only, reranker only, both. Compare Hit Rate@5, MRR@5, NDCG@5, and mean latency. Enable the component if: (a) the metric improvement exceeds a threshold (e.g. +0.05 on MRR), and (b) the latency increase is acceptable for the use case. If HyDE helps MRR but adds 800ms latency for a chatbot, skip it. If the reranker helps NDCG@5 by 0.1 (better ranking quality), enable it.

---

<a id="q69a"></a>
**Q69a. What were the measured retrieval metrics on the NeuralFlow AI corpus?**

Gold dataset: 10 queries against the NeuralFlow AI document corpus (8 docs + 4 audio files, all transcribed via Whisper). Baseline configuration: HyDE disabled, reranker disabled, Ollama local (nomic-embed-text embeddings). Run via `python -m pytest rag/tests/test_retrieval_metrics.py -v --log-cli-level=INFO`.

**Hybrid search results by K (current — post Whisper fix)**

```
=================================================================
  RETRIEVAL METRICS — hybrid search, NeuralFlow AI corpus
=================================================================
  Metric                   K=1       K=3       K=5
-----------------------------------------------------------------
  HIT_RATE@K             0.900     0.900     0.900
  MRR@K                  0.900     0.900     0.900
  PRECISION@K            0.900     0.633     0.560
  RECALL@K               0.750     1.550     2.250
  NDCG@K                 0.900     0.884     0.874
-----------------------------------------------------------------
  Mean latency            308ms
  P95  latency            903ms
=================================================================
```

**Previous results (before Whisper — for reference)**

```
  Metric                   K=1       K=3       K=5
-----------------------------------------------------------------
  HIT_RATE@K             0.700     0.800     0.800
  MRR@K                  0.700     0.733     0.733
  PRECISION@K            0.700     0.267     0.160
  RECALL@K               0.600     0.800     0.900
  NDCG@K                 0.700     0.756     0.756
-----------------------------------------------------------------
  Mean latency            312ms  /  P95: 748ms
```

**Per-query breakdown (K=5, hybrid — current)**

```
Query                                                Hit    RR     Lat
------------------------------------------------------------------------
What does NeuralFlow AI do?                           ✓   1.00   1447ms  ← cold-start embed
What is the PTO policy?                               ✓   1.00    238ms
What is the learning budget for employees?            ✓   1.00    235ms
What technologies and architecture does the ...       ✓   1.00    203ms
What is the company mission and vision?               ✓   1.00    167ms  ← fixed (was ✗ pre-Whisper)
GlobalFinance Corp loan processing success story      ✓   1.00    146ms  ← fixed (was RR=0.50)
How many employees work at NeuralFlow AI?             ✓   1.00    165ms
What is DocFlow AI and how does it process ...        ✗   0.00    146ms  ← still missing (see Q66)
Q4 2024 business results and performance review       ✓   1.00    186ms
implementation approach and playbook                  ✓   1.00    149ms
```

**Search type comparison (Hit Rate@5 — current)**

| Search type | Hit Rate@5 | MRR@5 | NDCG@5 | Notes |
|---|---|---|---|---|
| Hybrid (RRF) | 0.90 | 0.900 | 0.874 | Tied with semantic; one miss ("DocFlow AI") |
| Semantic only | 0.90 | 0.900 | — | Same single miss |
| Text only | 0.40 | — | — | Down from 0.60 pre-Whisper; audio transcripts added content that text search struggles with |

Hybrid and semantic are now equal. Text search dropped from 0.60 to 0.40 after Whisper re-ingestion — the audio transcripts are conversational prose with few strong keywords, which hurts text search more than it helps it.

**Minimum passing thresholds (CI gate)**

```python
THRESHOLDS_K5 = {
    "hit_rate":  0.60,   # current: 0.900 (+0.300 headroom)
    "mrr":       0.40,   # current: 0.900 (+0.500 headroom)
    "precision": 0.15,   # current: 0.560 (+0.410 headroom)
    "recall":    0.40,   # current: 2.250 (>1.0 — chunk-level inflation, see Q67)
    "ndcg":      0.40,   # current: 0.874 (+0.474 headroom)
}
```

All thresholds now have substantial headroom. Precision@5 had only 0.010 headroom before Whisper; it jumped to 0.410 because audio transcript chunks are highly relevant to their queries and dominate top results.

**Metric-by-metric analysis**

*Hit Rate@K — 0.90 → 0.90 → 0.90*

Flat across all K values — the 9 hits all land at rank 1, and the one miss (DocFlow AI) is absent from the index at any rank. No improvement from widening the candidate window, confirming the miss is a data/relevance problem not a ranking depth problem. The ceiling with this gold dataset is 0.90 until the DocFlow AI query is resolved (either fix the transcript content match or update the gold query to match what the transcript actually says).

*MRR@K — 0.900 → 0.900 → 0.900*

Perfect flat profile. Every hit lands at rank 1, giving RR=1.00 per query. The pre-Whisper GlobalFinance outlier (RR=0.50, relevant doc at rank 2) is now rank 1 — audio content shifted RRF rankings enough to push the correct chunk to the top. This is a meaningful improvement: the LLM receives the right chunk first on 9/10 queries.

*Precision@K — 0.900 → 0.633 → 0.560*

Still declines with K (expected — denominator grows faster than relevant results), but substantially higher than before. The K=1 jump from 0.700 to 0.900 directly reflects 2 additional queries hitting at rank 1 (mission/vision and GlobalFinance). The K=5 value of 0.560 means on average 2.8 out of 5 returned results are relevant — well above the 0.15 threshold and indicating the corpus is dense with relevant content.

*Recall@K — 0.750 → 1.550 → 2.250*

Values above 1.0 are now confirmed in production (Q67). This is chunk-level recall inflation: multiple chunks from the same relevant document appear in top-K, each counted as a separate relevant result but divided by `total_relevant = 1`. The jump from pre-Whisper (0.90 at K=5) to current (2.250 at K=5) is entirely explained by audio transcript chunks joining the index — for queries like "PTO policy" or "learning budget", both a handbook text chunk and an audio chunk about the same topic now appear in top-5. Not a metric to trust at face value; use Hit Rate and MRR as primary signals.

*NDCG@K — 0.900 → 0.884 → 0.874*

Slight downward slope with K, which is normal: as K increases, slightly less-relevant chunks fill positions 4–5 and gently reduce DCG. The 0.874 at K=5 is excellent — relevant documents are consistently at the top of the ranked list. The pre-Whisper plateau (0.756 flat from K=3 to K=5) has been replaced by a gradual decline, which is the healthier pattern — it means positions 4–5 now contain partially-relevant content rather than completely irrelevant noise.

*Latency — 308ms mean, 903ms P95*

Mean is virtually unchanged (308ms vs 312ms). P95 jumped from 748ms to 903ms — the first query after Whisper ingestion hit a cold-start embedding delay of 1447ms (Ollama loading the model into memory for the first time). In steady-state operation (Ollama model already loaded) all queries run at ~150–350ms. The 10s P95 threshold has ample headroom; tighten to 2,000ms for a realistic CI gate.

**Remaining miss: DocFlow AI**

"What is DocFlow AI and how does it process documents?" still fails at RR=0.00 despite `Recording2.mp3` now being transcribed. The transcript exists in the DB but the query doesn't match. To diagnose: inspect the actual chunk content with:

```sql
SELECT content FROM chunks
WHERE document_id = (SELECT id FROM documents WHERE source ILIKE '%Recording2%');
```

If the transcript uses different terminology (e.g. "document processing platform" rather than "DocFlow AI"), either update the gold query to match the transcript vocabulary, or expand `relevant_sources` for this entry to include additional source stems.

**What to act on, ranked by impact**

| Priority | Action | Expected gain |
|---|---|---|
| 1 | Inspect Recording2 transcript, fix gold query or relevant_sources | Hit Rate@5: 0.90 → 1.00 |
| 2 | Expand gold dataset to 50+ queries | Reduces metric variance from ±0.10 to ±0.03 |
| 3 | Add keyword-heavy queries to gold dataset | Text search dropped to 0.40; need queries that expose where hybrid beats semantic |
| 4 | Tighten P95 latency threshold from 10s → 2s | More realistic CI gate |
| 5 | Fix Recall metric to count distinct documents | Remove >1.0 inflation so Recall becomes a meaningful signal |
| 6 | Enable reranker and re-measure | All queries now hit; reranker value is in improving ranking within hits |

---

<a id="q69b"></a>
**Q69b. Where in the code are retrieval metrics collected?**

All collection lives in `rag/tests/test_retrieval_metrics.py`. Here is the full pipeline:

**1. Gold dataset — lines 34–75**

Static list of 10 queries and the document filename stems considered relevant. No DB involved.

```python
GOLD_DATASET: list[dict] = [
    {"query": "What does NeuralFlow AI do?",
     "relevant_sources": ["company-overview", "mission-and-goals"]},
    ...
]
```

**2. Raw retrieval — `_run_gold_dataset` (lines 286–310)**

Loops over every gold query, calls the real `retriever.retrieve()` against PostgreSQL, times each call, and converts results to a binary relevance list.

```python
t0 = time.perf_counter()
results = await retriever.retrieve(query=entry["query"], match_count=k, search_type=search_type)
latencies.append((time.perf_counter() - t0) * 1000)
rel_list = build_relevance_list(results, entry["relevant_sources"])
```

**3. Relevance judgement — `is_relevant` / `build_relevance_list` (lines 94–102)**

Substring match — if a `relevant_sources` stem (e.g. `"team-handbook"`) appears anywhere in the result's `document_source` path, the result is marked relevant (`1`), otherwise `0`. No LLM judge; no human annotation at query time.

```python
def is_relevant(document_source: str, relevant_sources: list[str]) -> bool:
    src_lower = document_source.lower()
    return any(stem.lower() in src_lower for stem in relevant_sources)
```

**4. Metric computation — pure functions (lines 105–165)**

Each metric is a standalone function that operates only on the binary relevance list — no I/O, fully unit-testable in isolation:

| Function | Lines | Formula |
|---|---|---|
| `hit_rate(rel)` | 105–107 | `1.0` if any `1` in list, else `0.0` |
| `reciprocal_rank(rel)` | 110–115 | `1 / position` of first `1` |
| `precision_at_k(rel, k)` | 118–122 | `sum(rel[:k]) / k` |
| `recall_at_k(rel, k, total)` | 125–129 | `sum(rel[:k]) / total_relevant` |
| `ndcg_at_k(rel, k)` | 132–144 | `DCG@K / IDCG@K` |

All five are aggregated in `compute_all_metrics` (lines 147–165), which takes the full list of per-query relevance lists and returns mean values across all queries.

**5. Latency percentile — `percentile` (lines 168–175)**

Linear interpolation between adjacent sorted values. Called as `percentile(latencies, 95)` for P95.

**6. Logging — `_log_metrics_table` + `_log_per_query_detail` (lines 312–347)**

Prints the metrics table and per-query breakdown (query text, ✓/✗, RR, latency) to test stdout via `logger.info`. Display only — does not affect assertions.

**7. CI assertions — individual test methods (lines 354–474)**

Each test method calls `_run_gold_dataset`, computes metrics, and asserts against `THRESHOLDS_K5`. `test_semantic_vs_text_hit_rate` (line 437) and `test_hybrid_beats_semantic_alone` (line 451) produce the 0.90 / 0.80 / 0.60 search-type comparison.

**Data flow**

```
GOLD_DATASET (static)
       │
       ▼
retriever.retrieve()          ← rag/retrieval/retriever.py
       │  (real PostgreSQL + Ollama)
       ▼
build_relevance_list()        ← substring match on document_source
       │
       ▼
compute_all_metrics()         ← pure math, no I/O
       │
       ▼
_log_metrics_table()          ← prints table to test stdout
assert score >= threshold     ← CI gate
```

The retriever (`rag/retrieval/retriever.py`) is where the actual hybrid search executes. `_run_gold_dataset` is purely a timing harness that loops over queries and measures wall-clock time around each `retrieve()` call.

---

## Observability & Langfuse

<a id="q70"></a>
**Q70. What does Langfuse trace in this project?**

Each `traced_agent_run` call creates a Langfuse trace covering the full agent turn. Within it: the initial user message, the `search_knowledge_base` tool call (inputs + output), the Mem0 memory lookup, and the final LLM generation. This gives a per-turn view of what was retrieved, what context was provided, and what the model generated — essential for debugging wrong answers.

<a id="q71"></a>
**Q71. Why `ContextVar` rather than function arguments?**

Passing a trace object through every function argument would require touching every function signature in the call chain (agent → tool → retriever → store). `ContextVar` stores the trace implicitly, available to any coroutine in the same async task without parameter threading. This is the standard Python pattern for request-scoped context (similar to `flask.g` in sync Flask).

<a id="q72"></a>
**Q72. Using Langfuse traces to debug a wrong answer.**

1. Find the trace for the query.
2. Check the `search_knowledge_base` tool call — what query was sent to the retriever?
3. Check the retrieved chunks — are the relevant documents present? Are they ranked first?
4. Check the Mem0 context — is any stale/wrong memory being injected?
5. Check the final generation — is the LLM ignoring the correct chunk? (Lost in the middle?)
This narrows the bug to one of: retrieval failure, ranking failure, memory contamination, or LLM generation failure.

<a id="q73"></a>
**Q73. Trace vs span vs generation in Langfuse?**

A **trace** is the top-level unit — one user request end-to-end. A **span** is a named sub-operation within a trace (e.g. "retrieve", "rerank") with start/end times. A **generation** is a special span that captures an LLM API call — it records the prompt, completion, model name, token usage, and cost. Langfuse aggregates generations for cost tracking.

---

## System Design

<a id="q74"></a>
**Q74. Two-table schema (documents + chunks) — why not one table?**

Storing full document content in every chunk row would be massive redundancy (a 10-page PDF split into 20 chunks → the full PDF stored 20 times). The two-table design stores the full document once in `documents` and references it from `chunks` via FK. It also enables document-level operations (update, delete, list) without touching chunks, and supports the `ON DELETE CASCADE` pattern for clean teardown.

<a id="q75"></a>
**Q75. Walk through ingestion of a raw PDF to a searchable chunk.**

1. `_find_document_files()` discovers `technical-architecture-guide.pdf`.
2. `_compute_file_hash()` computes MD5; in incremental mode, compare against stored hash.
3. `_read_document()` calls `_get_converter()` → `DocumentConverter.convert(pdf_path)` → Docling ML pipeline (layout detection, text extraction, table structure) → returns `(markdown_text, DoclingDocument)`.
4. `_extract_title()` scans first 10 lines for `# ` heading.
5. `_extract_document_metadata()` records file path, size, word count, hash, ingestion date.
6. `chunker.chunk_document(content, title, source, metadata, docling_doc)` → `HybridChunker.chunk(dl_doc)` produces structural chunks → `contextualize(chunk)` prepends heading path → returns `list[ChunkData]` with token counts.
7. `embedder.embed_chunks(chunks)` calls `POST /v1/embeddings` (Ollama) → attaches 768-dim vectors.
8. `store.save_document(...)` → `INSERT INTO documents ...` → returns UUID.
9. `store.add(embedded_chunks, document_id)` → `executemany INSERT INTO chunks ...` including embedding vector.
10. GIN index on `content_tsv` is updated automatically via generated column.
11. `_result_cache.clear()` invalidates the retriever's in-memory cache.

<a id="q76"></a>
**Q76. Scale to 10M documents — what breaks first?**

1. **IVFFlat vector index** — at 10M rows with 768-dim vectors, IVFFlat recall degrades unless `lists` is tuned to ~3162 (sqrt of 10M) and `probes` increased. Switch to HNSW (supported in pgvector ≥0.5) which maintains recall at scale.
2. **Ingestion throughput** — `DocumentConverter` is single-threaded and CPU-bound. A single process cannot ingest fast enough. Need a message queue (Kafka/SQS) + worker pool.
3. **PostgreSQL write throughput** — 10M documents × ~20 chunks × 3KB embeddings ≈ 600GB. Need table partitioning, read replicas, and potentially a separate vector store.

<a id="q77"></a>
**Q77. Implementing true incremental ingestion with deduplication.**

The pipeline already does this (`clean_before_ingest=False`): compute MD5 hash of the file, compare against `metadata.content_hash` stored in the `documents` table. If equal → skip. If different → `delete_document_and_chunks(source)` then re-ingest. If the source doesn't exist → ingest as new. Deleted files are handled by comparing `current_sources` (files on disk) against `get_all_document_sources()` (files in DB) and deleting any that are in DB but not on disk.

<a id="q78"></a>
**Q78. Is multi-tenancy supported? What would it take to make this prototype production-ready for multiple tenants?**

**Current state: No multi-tenancy support.**

The system has a single shared `documents` and `chunks` table with no tenant isolation. Every user queries the same corpus. If customer A's documents are ingested alongside customer B's, B can retrieve A's content and vice versa — there is no access boundary.

**Three isolation strategies — trade-offs:**

| Strategy | Isolation | Complexity | Cost | Best for |
|---|---|---|---|---|
| Row-level security (RLS) | Strong (policy-enforced) | Medium | Low (one DB) | Many small tenants (SaaS) |
| Schema-per-tenant | Strong (namespace) | High (DDL per tenant) | Medium | Tens of tenants |
| Database-per-tenant | Complete | Very high | High | Few large enterprise customers |

---

**Option 1 — Row-Level Security (recommended for SaaS)**

Add `tenant_id` to both tables and let PostgreSQL enforce access at the row level. No application-level filtering needed — the DB rejects queries that cross tenant boundaries even if the app has a bug.

Schema changes:
```sql
ALTER TABLE documents ADD COLUMN tenant_id TEXT NOT NULL;
ALTER TABLE chunks    ADD COLUMN tenant_id TEXT NOT NULL;

-- Index for fast per-tenant scans
CREATE INDEX documents_tenant_idx ON documents(tenant_id);
CREATE INDEX chunks_tenant_idx    ON chunks(tenant_id);

-- RLS policies
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks    ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation_documents ON documents
    USING (tenant_id = current_setting('app.tenant_id'));

CREATE POLICY tenant_isolation_chunks ON chunks
    USING (tenant_id = current_setting('app.tenant_id'));
```

Per-request, set the tenant context on the connection before any query:
```python
async with pool.acquire() as conn:
    await conn.execute("SET LOCAL app.tenant_id = $1", tenant_id)
    # All subsequent queries on this connection automatically scoped to tenant
    results = await conn.fetch("SELECT * FROM chunks ...")
```

Changes needed in this codebase:
- `PostgresHybridStore.add()` — pass `tenant_id` into every INSERT
- `PostgresHybridStore.semantic_search()` / `text_search()` — set `app.tenant_id` on the connection before querying (RLS handles the rest automatically)
- `RAGState` — carry `tenant_id` alongside `user_id`, pass it into the store
- Ingestion pipeline — accept `tenant_id` as a parameter and write it to every document/chunk row
- IVFFlat index — may need `lists` retuning since the effective index size per tenant is smaller than the full table

**RLS caveat with connection pools:** asyncpg reuses connections. `SET LOCAL` resets at transaction end, but `SET` persists for the connection lifetime. Always use `SET LOCAL` (transaction-scoped) or reset after use to prevent tenant leakage across pooled connections.

---

**Option 2 — Schema-per-tenant**

Each tenant gets their own PostgreSQL schema with identical table structure:

```sql
CREATE SCHEMA tenant_acme;
CREATE TABLE tenant_acme.documents ( ... );  -- same DDL
CREATE TABLE tenant_acme.chunks    ( ... );
```

The `PostgresHybridStore` is initialised with a `schema` parameter:
```python
store = PostgresHybridStore(schema="tenant_acme")
# All queries use f"{self.schema}.chunks" instead of "chunks"
```

Pros: true namespace isolation, no RLS policy bugs, easy to dump/restore a single tenant.
Cons: schema migration (adding a column) must run against every tenant schema — needs a migration runner that iterates all tenant schemas.

---

**Option 3 — Database/branch per tenant (Neon)**

Each tenant gets their own Neon branch or project. Maximum isolation — one tenant's load cannot affect another's query latency. `DATABASE_URL` is tenant-specific and stored in a tenant registry.

```python
# Tenant registry (e.g. stored in a separate "control plane" DB)
tenant_db_urls = {
    "acme":  "postgresql://user:pass@acme.neon.tech/rag",
    "globex": "postgresql://user:pass@globex.neon.tech/rag",
}
store = PostgresHybridStore(database_url=tenant_db_urls[tenant_id])
```

Neon's branching is particularly well-suited: create a branch per tenant from a template branch that already has the schema and indexes set up.

Cons: connection pool per tenant (memory cost), cross-tenant analytics require federated queries.

---

**Full production readiness checklist beyond just multi-tenancy:**

| Area | What's missing | What to add |
|---|---|---|
| **Auth** | No authentication | JWT validation middleware; `tenant_id` extracted from the token claim |
| **Tenant provisioning** | Manual | API to create tenant → run schema migration → register in tenant registry |
| **Data isolation** | None | RLS (recommended) or schema-per-tenant |
| **Rate limiting** | None | Per-tenant query rate limits (token bucket in Redis) |
| **Ingestion ACL** | None | Only tenant admins can ingest documents for their tenant |
| **Audit logging** | None | Log every query with `tenant_id`, `user_id`, timestamp to an audit table |
| **Embedding model lock** | None | Store `embedding_model` + `embedding_dimension` per tenant; block re-ingestion with wrong model |
| **Soft deletes** | None | `deleted_at` column instead of hard DELETE, for audit trail |
| **Billing hooks** | None | Count chunks/queries per tenant for usage-based billing |
| **Backup/restore** | None | Per-tenant pg_dump or Neon branch snapshot |
| **Zero-downtime re-index** | None | Shadow table swap (Q106) per tenant |
| **Search result ACL** | None | Document-level permissions within a tenant (not just tenant-level) |

**Recommended migration path for this codebase (RLS approach):**

1. Add `tenant_id TEXT NOT NULL DEFAULT 'default'` to both tables (non-breaking — existing data gets `'default'` tenant)
2. Add indexes on `tenant_id`
3. Enable RLS and create policies
4. Update `PostgresHybridStore` to `SET LOCAL app.tenant_id` at connection checkout
5. Update ingestion pipeline to accept and write `tenant_id`
6. Update `RAGState` to carry `tenant_id` from the auth layer
7. Add JWT middleware to the Streamlit/API layer to extract `tenant_id` from the request token

<a id="q79"></a>
**Q79. Risk of changing the embedding model after ingestion.**

All existing chunk embeddings are in the old model's vector space. New query embeddings are in the new model's space. Vector similarity between old and new spaces is meaningless — cosine similarity of incomparable vectors would return arbitrary scores. Result: total retrieval failure. Fix: re-ingest all documents with the new model before switching query embedding. Zero-downtime approach: dual-write to a new index during migration, switch queries over once the new index is complete.

<a id="q80"></a>
**Q80. Sub-100ms latency — what to sacrifice first?**

Drop HyDE first (saves one LLM call, ~500ms). Then disable reranking (saves n API calls). Then consider switching from hybrid to semantic-only (saves the text search + RRF merge, ~50ms). Finally, switch from remote Ollama embedding to a locally loaded model with caching. The embedding call is the dominant latency outside the DB query itself.

---

## Code Quality

<a id="q81"></a>
**Q81. Why `pydantic-settings` instead of `os.environ`?**

`pydantic-settings` provides: (1) type validation — `EMBEDDING_DIMENSION=abc` raises a `ValidationError` immediately rather than failing at runtime with a cryptic type error; (2) automatic `.env` file loading; (3) default values with documentation in the model definition; (4) credential masking in `__repr__` (API keys shown as `***`). Raw `os.environ` gives you a dict of strings with no validation, defaults, or type coercion.

<a id="q82"></a>
**Q82. What does `ruff` check for vs `flake8 + black`?**

`ruff` is a Rust-based linter that replaces both `flake8` (style + lint rules) and `black` (formatting) in a single tool. It is 10–100× faster than the Python equivalents and checks for: unused imports, undefined names, import ordering (isort), type annotation style, security issues (bandit-equivalent rules), and more. `ruff format` handles formatting (black-compatible). The key benefit over flake8 + black is a single configuration file and a single command.

<a id="q83"></a>
**Q83. Why Pydantic models for `ChunkData` and `SearchResult` instead of plain dataclasses?**

Pydantic provides runtime type validation — if a search result is returned with `similarity` as a string instead of a float, Pydantic raises a `ValidationError` immediately rather than a downstream `AttributeError`. Pydantic models also have automatic `__repr__`, JSON serialisation, and schema generation. For data flowing between system boundaries (DB → Python → LLM context), the validation guarantees are worth the overhead.

<a id="q84"></a>
**Q84. Why `from collections.abc import Callable` rather than `callable`?**

`callable` is a built-in function, not a type. `Callable[[int], str]` is a type annotation saying "a function that takes an int and returns a str". In Python ≤3.8, `typing.Callable` was the way; in 3.9+, `collections.abc.Callable` is preferred (the `typing` versions are being deprecated). The CLAUDE.md convention exists because a previous bug was introduced by using lowercase `callable` as a type annotation — it evaluated to `True`/`False` rather than the type spec.

<a id="q85"></a>
**Q85. How does `IngestionConfig` → `ChunkingConfig` separation keep concerns clean?**

`IngestionConfig` is the pipeline-level config — it owns parameters relevant to the pipeline as a whole (chunk_size, chunk_overlap, max_chunk_size, max_tokens). `ChunkingConfig` is the chunker's own config — it's what the `DoclingHybridChunker` constructor accepts. The pipeline translates one into the other. This means the chunker is usable independently of the pipeline (e.g. in tests, in the notebook) without constructing a full `IngestionConfig`. It also means the chunker's interface can evolve without changing the pipeline's public API.

---

## Ingestion Pipeline Deep Dive

<a id="q91"></a>
**Q91. Walk through full ingestion step by step.**

See Q75 — detailed answer there. Summary path: `_find_document_files()` → `_compute_file_hash()` → `_read_document()` (Docling → markdown + DoclingDocument) → `_extract_title()` → `_extract_document_metadata()` → `chunker.chunk_document()` (HybridChunker → contextualize → ChunkData list) → `embedder.embed_chunks()` (POST /v1/embeddings) → `store.save_document()` (INSERT documents) → `store.add()` (executemany INSERT chunks) → `_result_cache.clear()`.

<a id="q92"></a>
**Q92. How does `DocumentConverter` differ from PyPDF2 / pdfplumber?**

PyPDF2 and pdfplumber extract raw text streams from PDF content streams — they are layout-unaware. A two-column PDF produces interleaved text from both columns. Tables become unformatted text. Docling's `DocumentConverter` runs a full ML pipeline: (1) layout detection (identifies text blocks, tables, figures, headers/footers per page using a vision model), (2) reading order determination (correct multi-column flow), (3) table structure recognition (identifies rows/cols in table images), (4) formula detection. The output is a structured `DoclingDocument` with typed elements: `TextItem`, `TableItem`, `PictureItem`, `SectionHeaderItem` etc., preserving semantic structure.

<a id="q93"></a>
**Q93. What internal representation does `DoclingDocument` provide and how does `HybridChunker` use it?**

`DoclingDocument` is a hierarchical document object with: a `body` containing a tree of typed items (`SectionHeaderItem`, `TextItem`, `TableItem`, `ListItem`, etc.), each tagged with its heading path (e.g. item is under "## Architecture > ### Storage"). `HybridChunker` traverses this tree, grouping items into chunks such that: (a) a `SectionHeaderItem` starts a new chunk boundary, (b) `TextItem`s within the same section are merged until `max_tokens` is exceeded, (c) a `TableItem` is kept as a single chunk (never split mid-table), (d) `merge_peers=True` merges adjacent small chunks at the same structural level.

<a id="q94"></a>
**Q94. Explain `contextualize()` — what exactly gets prepended?**

`contextualize(chunk)` reads the `heading_path` attribute of the chunk (set by HybridChunker from the parent `SectionHeaderItem` ancestors) and prepends it as a breadcrumb: `"Level1 > Level2 > Level3\n\n"` followed by the chunk's raw text. For a chunk about PTO under `## Benefits > ### Time Off Policy`, the output is:

```
Benefits > Time Off Policy

Employees are entitled to 20 days of PTO per year...
```

The embedding of this contextualized text places it closer in vector space to queries about "benefits PTO policy" than the embedding of the raw text alone.

<a id="q95"></a>
**Q95. What is `merge_peers=True` — give an example.**

Consider a document with three consecutive short paragraphs under "## Goals", each 50 tokens:
- Without `merge_peers`: three separate chunks of 50 tokens each — too short, poor embedding signal.
- With `merge_peers=True`: the three paragraphs are merged into one chunk of ~150 tokens, under the shared heading context. Better semantic coherence and one fewer DB row to search.

You'd turn it off if you need maximum granularity for a corpus with very long sections where merging pushes chunks over `max_tokens`.

<a id="q96"></a>
**Q96. What happens to a table in a PDF during chunking?**

Docling's `DocumentConverter` identifies table regions and applies table structure recognition to parse rows and columns. The table becomes a `TableItem` in `DoclingDocument` with structured data. `HybridChunker` treats a `TableItem` as an atomic unit — it is never split across chunk boundaries. The table is serialized to a text representation (usually a markdown table or CSV-like format) and included as a single chunk. This preserves the relational structure of the table for embedding.

<a id="q97"></a>
**Q97. Tokenizer mismatch: `all-MiniLM-L6-v2` for chunking, `nomic-embed-text` for embedding.**

`all-MiniLM-L6-v2`'s tokenizer is used by HybridChunker to count tokens and enforce the 512-token limit. `nomic-embed-text` uses a different tokenizer (based on GPT-style BPE). The two tokenizers have different vocabularies — a chunk that is 512 tokens by `all-MiniLM` may be 530 tokens by `nomic-embed-text`'s tokenizer, causing silent truncation when the embedding model processes it. Mitigation: use the embedding model's own tokenizer for chunk boundary decisions. In practice, the difference is small (~5%) and rarely causes significant truncation.

<a id="q98"></a>
**Q98. Describe the fallback chunking path exactly.**

In `_simple_fallback_chunk` (`chunkers/docling.py:228`):
1. Start at `pos = 0`.
2. Compute `end = pos + chunk_size` (default 500 chars).
3. Walk backwards from `end` to `max(pos + min_chunk_size, end - 200)` looking for `.`, `!`, `?`, or `\n`.
4. If found, cut there (sentence boundary). Otherwise cut at `end`.
5. Strip and create `ChunkData` with `chunk_method="simple_fallback"`.
6. Advance: `pos = end - overlap` (overlap = 100 chars default).
7. Repeat until `pos >= len(content)`.
8. After all chunks are built, update `total_chunks` in each chunk's metadata.

<a id="q99"></a>
**Q99. Why cache `DocumentConverter`?**

Creating a `DocumentConverter` loads PyTorch ML models (layout detection ~200MB, table structure ~100MB) from disk into memory, initializes GPU/CPU compute contexts, and allocates memory. On CPU this takes 5–15 seconds. With caching (`_get_converter()` returns `self._doc_converter` if already set), this cost is paid once per pipeline instance. For a batch of 13 documents, that's 12 avoided re-loads, saving up to 3 minutes of startup time.

<a id="q100"></a>
**Q100. MD5 for content hashing — how it works and limitations.**

`_compute_file_hash()` reads the file in 8192-byte blocks and feeds them to `hashlib.md5()`, returning the hex digest. Stored in `metadata.content_hash` in the `documents` table. Incremental ingestion compares this hash with the stored one: equal → skip, different → delete + re-ingest.

Limitations: (1) MD5 has known collision vulnerabilities — not cryptographically safe, but fine for file change detection (not security). (2) If a file's bytes change but its content is semantically unchanged (e.g. PDF metadata update, BOM encoding change), the hash changes and triggers unnecessary re-ingestion. (3) Conversely, a semantically meaningful change that happens to produce the same MD5 (collision) would be silently skipped — extremely unlikely in practice.

<a id="q101"></a>
**Q101. Incremental ingestion — walk through all four cases.**

In `ingest_documents()` with `clean_before_ingest=False`:

- **New file**: `get_document_hash(source)` returns `None` (not in DB). Log `[NEW]`. Call `_ingest_single_document()`. Increment `new_count`.
- **Unchanged file**: `get_document_hash(source)` returns a hash that matches `_compute_file_hash()`. Log `[SKIP]`. Increment `skipped_count`. No processing.
- **Modified file**: hash mismatch. Log `[UPDATE]`. Call `delete_document_and_chunks(source)` (deletes document + cascades to chunks). Then call `_ingest_single_document()` to re-ingest. Increment `updated_count`.
- **Deleted file**: after processing all files on disk, call `get_all_document_sources()` from DB and compare against `current_sources` (files found on disk). Any source in DB but not on disk → `delete_document_and_chunks()`. Increment `deleted_count`.

<a id="q102"></a>
**Q102. Why `_result_cache.clear()` after ingestion?**

The retriever has a module-level `_result_cache` (LRU + TTL) that stores query → results mappings. After ingestion, new chunks exist in the DB that were not there when the cache entries were computed. If a user queries "What is the PTO policy?" 1 minute before ingestion and again 1 minute after, the cache would return stale results (missing the newly ingested chunks). Clearing the cache forces re-queries against the updated DB immediately. The cache is module-level and shared across all `Retriever` instances, so a single `.clear()` suffices.

<a id="q103"></a>
**Q103. YAML frontmatter — where stored, how used?**

`_extract_document_metadata()` checks if the content starts with `---` and tries to parse the YAML block between the first `---` and the next `\n---\n`. The parsed key-value pairs are merged into the `metadata` dict, which is stored in the `documents.metadata` JSONB column and also copied into each `ChunkData.metadata`. At query time, metadata is returned in `SearchResult` objects and can be used for filtering (e.g. `WHERE metadata->>'author' = 'Alice'`) or display. Currently it is stored but not used for search filtering — a future enhancement would be metadata-filtered retrieval.

<a id="q104"></a>
**Q104. Top three bottlenecks at 10,000 docs/day and fixes.**

1. **`DocumentConverter` (CPU-bound, sequential)**: single ML inference pipeline processes ~1 doc/sec on CPU. At 10K/day that's ~2.8 hours. Fix: parallelize with a worker pool (`asyncio.to_thread` wrapping the sync converter call), multiple processes, or GPU-enabled instances.

2. **Embedding API calls (network-bound, sequential per document)**: `embed_chunks()` calls the embeddings API once per document's chunks. At 20 chunks/doc × 10K docs = 200K API calls. Fix: batch across documents (accumulate chunks from multiple documents and embed in large batches), use a local embedding model, or async-parallel embed across multiple documents.

3. **PostgreSQL write throughput**: `executemany` is fast per document but 10K documents × 20 chunks × 3KB vectors = 600MB of data per day. Fix: use PostgreSQL `COPY` protocol for bulk load, partition the `chunks` table by ingestion date, and tune `work_mem`/`checkpoint_segments` for write performance.

<a id="q105"></a>
**Q105. Parallelizing ingestion while sharing `DocumentConverter` and the asyncpg pool.**

`DocumentConverter` is not thread-safe (PyTorch models share state). The pattern: run conversion in `asyncio.to_thread` — each conversion call gets its own thread where it calls a fresh (not cached) `DocumentConverter`. Alternatively, use a `multiprocessing.Pool` where each worker process has its own converter instance. The asyncpg pool is already thread-safe and handles concurrent connections. A semaphore limits concurrent conversions to avoid OOM:

```python
sem = asyncio.Semaphore(4)  # 4 concurrent conversions
async def ingest_with_limit(file):
    async with sem:
        return await asyncio.to_thread(convert_and_ingest, file)
await asyncio.gather(*[ingest_with_limit(f) for f in files])
```

<a id="q106"></a>
**Q106. Zero-downtime re-index when `clean_before_ingest=True`.**

The current `clean_collections()` drops all data before re-ingesting — there is a window where the DB is empty and queries return nothing. Zero-downtime approach:
1. Create a new set of tables (`documents_v2`, `chunks_v2`).
2. Ingest all documents into the new tables.
3. Atomically swap table names (PostgreSQL `ALTER TABLE RENAME` is transactional).
4. Drop the old tables.

Or use PostgreSQL table inheritance / partitioning with a read view that spans both versions during migration. Neon's branching feature makes this even cleaner — ingest on a branch, then merge.

<a id="q107"></a>
**Q107. Scanned PDFs with no text layer.**

`DocumentConverter` runs OCR (via Tesseract or a built-in OCR pipeline) when no text layer is detected. This is slower than digital PDF processing. If OCR is not configured or fails, `export_to_markdown()` returns an empty or near-empty string. The pipeline falls through to the raw UTF-8 read fallback, which for a scanned PDF with no text layer returns binary garbage or an empty file. Fix: detect empty conversion output before chunking (`if len(markdown_content.strip()) < 100: raise`) and log a clear error rather than creating empty chunks.

<a id="q108"></a>
**Q108. Why return both markdown string and `DoclingDocument`?**

Re-parsing the markdown string would lose the original structure. `DoclingDocument` is Docling's in-memory structured representation — it has typed elements, heading trees, table data. If you serialise to markdown and re-parse, you get a flat text representation: headings become `# text`, tables become text grids, and the structural hierarchy is lost. `HybridChunker` needs the original `DoclingDocument` to use structural boundaries. The markdown string is stored in `documents.content` for human readability and full-text indexing; the `DoclingDocument` is used only during the chunking step.

<a id="q109"></a>
**Q109. Audio files — how are they different from PDF chunks?**

Audio transcription goes through `AsrPipeline` (Whisper Turbo). The output `DoclingDocument` contains `TextItem`s with timestamps (`[time: 0.0-5.2]` markers embedded in the text) rather than heading-structured text. There is no heading hierarchy, so `contextualize()` has nothing to prepend — the heading path is empty. The result is that audio chunks behave like the simple fallback path for contextualization but use HybridChunker's token-aware splitting. The `[time: X.X-Y.Y]` markers in chunk text allow the retrieval system to surface the exact timestamp in the audio file where a topic was discussed.

<a id="q110"></a>
**Q110. Impact of raw text fallback when PDF conversion fails.**

When Docling fails, `_read_document()` falls back to `open(file_path, encoding='utf-8').read()`. For PDFs this returns binary-encoded garbage (PDF syntax: `%PDF-1.4`, object streams, xref tables). This is passed to `_simple_fallback_chunk()` which creates chunks of garbage text. These chunks get embedded (producing meaningless vectors) and stored. At query time they score low on semantic search but may accidentally score on text search (e.g. if the PDF binary happens to contain the word "PTO" in a content stream). Better fallback: detect the file type before the UTF-8 read and return `("[Error: could not convert PDF]", None)` immediately, so the document is recorded in the DB with an error but no garbage chunks are created.

---

## Tricky / Deep-Dive

<a id="q86"></a>
**Q86. RRF scores of 0.01–0.03 — why isn't this low confidence?**

RRF scores are not probabilities. They are sums of `1/(k+rank)` terms. With k=60, the maximum possible RRF score for a chunk that ranks #1 in both semantic and text is `1/61 + 1/61 ≈ 0.033`. A score of 0.016 (rank #1 in one list only) is high — it's half the maximum. The intuition: RRF scores are relative within a result set, not absolute confidence levels. Compare results by their RRF scores against each other, not against 1.0.

<a id="q87"></a>
**Q87. After re-ingestion, previously passing tests now fail. Possible causes.**

1. **Chunk boundaries changed**: `DocumentConverter` is non-deterministic at the token boundary when `merge_peers` adjustments occur. A chunk that previously contained the key phrase may now be split differently.
2. **Different embedding values**: embedding models have non-deterministic temperature or the model was updated — same text produces slightly different vectors, shifting rankings.
3. **New documents added**: new chunks from additional documents may outscore the previously top-ranked chunk for some queries.
4. **Content hash collision**: a file was modified but `_compute_file_hash` incorrectly returned the old hash (race condition), so the document was not re-ingested with the new content.
5. **Result cache stale**: `_result_cache.clear()` was not called after ingestion in the test setup (check that `ingest_documents()` was awaited fully before running test queries).

<a id="q88"></a>
**Q88. Query "PTO" — what happens in tsvector and why might it miss "paid time off"?**

`plainto_tsquery('english', 'PTO')` → `'pto'`. This lexeme is searched in `content_tsv`. A document chunk that contains "paid time off" but never mentions "PTO" → `to_tsvector('english', '... paid time off ...')` → `'paid':2 'time':3 'off':4`. The lexeme `'pto'` is absent. Text search returns 0 for this chunk. The semantic leg would still match (embedding of "PTO" is close to "paid time off" in vector space). This is exactly the use case for hybrid search — text search misses the vocabulary mismatch, semantic catches it.

<a id="q89"></a>
**Q89. LLM reranker with partial failure (rate limiting).**

`asyncio.gather(*scoring_calls)` with default settings: if one call raises an exception, `gather` re-raises it and other tasks are not cancelled (Python 3.11 with `return_exceptions=False` cancels them; with `return_exceptions=True` returns exceptions as values). The safe approach used should be `return_exceptions=True`, then filter out exceptions from the results and assign a neutral score (0.0) to failed chunks. The reranker should then return the subset of successfully scored chunks rather than erroring entirely. If all calls fail, fall back to the original retrieval order.

<a id="q90"></a>
**Q90. Changing `chunk_overlap` from 100 to 0 — improve some metrics, hurt others?**

With overlap=0: fewer total chunks (no duplicated content at boundaries), cleaner boundaries, no duplicate information in the index. Precision@K may improve (less redundant chunks in results). Recall@K may drop: a sentence that straddles a boundary is now fully in one chunk rather than partially in two — if it's in the "wrong" chunk, the query misses it. MRR could go either way. The improvement is most visible in small corpora where duplicate chunks from overlap pollute results. For large corpora, overlap is important to prevent boundary-straddling losses.

---

## Scale, Latency & Precision Models

<a id="q111"></a>
**Q111. What are the main scale bottlenecks in this system at 1M documents?**

At 1M documents the system hits three hard limits:

1. **IVFFlat index accuracy degrades** — IVFFlat partitions vectors into `lists` clusters and searches only `probes` clusters at query time. With 1M vectors, the default `lists=100` is grossly under-partitioned (pgvector recommends `lists ≈ sqrt(rows)`  → ~1000 for 1M rows). With too few lists, each cluster is huge and `probes` must be increased to maintain recall — but that defeats the speed benefit. Fix: rebuild the index with `lists=1000`, tune `probes` to balance recall vs latency, or migrate to HNSW which scales better.

2. **PostgreSQL table scan for text search** — GIN index on `content_tsv` scales well to millions of rows, but the `ts_rank` scoring function re-scores every matched row. At 1M chunks, a broad query like "company policy" may match 100K rows that all need ranking. Fix: limit via metadata filters (tenant_id, date range) before full-text scoring.

3. **Single PostgreSQL instance write throughput** — 1M documents × 20 chunks × 768-dim float32 vectors = ~60GB of vector data. A single Postgres instance hits I/O limits during bulk ingestion. Fix: Neon's branching for parallel ingest on separate branches, then merge; or partition `chunks` by `document_id` hash across multiple Postgres instances.

<a id="q112"></a>
**Q112. What are the ingestion latency bottlenecks and how would you profile them?**

The ingestion pipeline has four serial stages per document, each with a different bottleneck type:

| Stage | Typical latency | Bottleneck type | Profiling tool |
|---|---|---|---|
| `DocumentConverter.convert()` | 2–15s/doc (CPU) | CPU-bound ML inference (layout detection, table recognition) | `cProfile`, GPU utilisation |
| `chunker.chunk_document()` | 50–200ms/doc | CPU-bound tokenization | `time.perf_counter` around call |
| `embedder.embed_chunks()` | 100–500ms/doc | Network I/O (HTTP to embedding API) | HTTP request tracing, async profiler |
| `store.add()` (DB write) | 20–100ms/doc | Network I/O + disk I/O | `EXPLAIN ANALYZE` on INSERT, asyncpg query timing |

**How to profile:**
```python
import time
t0 = time.perf_counter()
result = converter.convert(path)
print(f"convert: {time.perf_counter() - t0:.2f}s")
```

Or instrument with Langfuse spans — wrap each stage in a `langfuse_client.span(name="convert")` context manager to get per-stage timing in the Langfuse dashboard.

**Biggest win**: `DocumentConverter` is the dominant cost. Parallelising it with a semaphore-bounded `asyncio.to_thread` pool gives near-linear throughput improvement up to the number of CPU cores.

<a id="q113"></a>
**Q113. What are the retrieval latency bottlenecks and how would you reduce them to sub-100ms?**

Current retrieval path latency breakdown (approximate, local Ollama):

| Step | Latency | Notes |
|---|---|---|
| Embed query | 20–50ms | HTTP to local Ollama embedding endpoint |
| Semantic search (IVFFlat) | 5–20ms | PostgreSQL vector scan |
| Text search (GIN) | 2–10ms | PostgreSQL tsvector scan |
| RRF merge (Python) | <1ms | Pure in-memory |
| HyDE LLM call (if enabled) | 500–2000ms | Full LLM generation — dominant cost |
| Reranker (if enabled) | 200–1000ms | N parallel LLM calls or CrossEncoder forward pass |

**To reach sub-100ms:**
1. **Disable HyDE and reranker** — both are off by default. Retrieval without them is already ~30–80ms.
2. **Cache embeddings** — the embedder has an in-memory cache keyed on query text. Repeated queries return instantly.
3. **Cache retrieval results** — `_result_cache` (LRU+TTL) returns cached results for identical queries.
4. **Switch IVFFlat → HNSW** — HNSW has lower query latency at high recall. Trade-off: more memory (~2–3× IVFFlat) and slower index build.
5. **Use a faster embedding model** — smaller models (e.g. `nomic-embed-text` at 768-dim is already fast; `all-MiniLM-L6-v2` at 384-dim is faster but lower quality).
6. **Connection pooling** — asyncpg pool avoids TCP handshake + SSL per query. Already in place.
7. **Co-locate** — run PostgreSQL and the app on the same machine or in the same datacenter to cut network RTT.

<a id="q114"></a>
**Q114. What models can be swapped in to improve retrieval precision?**

Precision can be improved at three stages — embedding, reranking, and generation:

**Embedding models (affects semantic search quality):**

| Model | Dimensions | Strengths | Trade-off |
|---|---|---|---|
| `nomic-embed-text` (current) | 768 | Fast, local via Ollama, good general quality | Not fine-tuned for RAG |
| `text-embedding-3-small` (OpenAI) | 1536 | Strong general retrieval, MTEB top tier | Paid API, network latency |
| `text-embedding-3-large` (OpenAI) | 3072 | Best OpenAI retrieval quality | 2× cost of small, more DB storage |
| `voyage-3` (Voyage AI) | 1024 | Optimised for RAG, strong on long documents | Paid API |
| `voyage-3-lite` | 512 | Fastest Voyage model | Lower quality than voyage-3 |
| `bge-large-en-v1.5` (BAAI) | 1024 | Open source, strong MTEB scores | Larger than nomic, needs more RAM |
| `e5-mistral-7b-instruct` | 4096 | Instruction-tuned, best open-source quality | 7B params — slow without GPU |

Switching model requires: (1) update `EMBEDDING_MODEL` + `EMBEDDING_DIMENSION` in `.env`, (2) drop and recreate the IVFFlat index with the new dimension, (3) re-ingest all documents (old vectors are incompatible).

**Reranking models (affects precision@K after retrieval):**

| Model | Type | Latency | Quality |
|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | CrossEncoder | ~50ms/batch | Good, fast |
| `cross-encoder/ms-marco-electra-base` | CrossEncoder | ~100ms/batch | Better quality |
| `BAAI/bge-reranker-large` | CrossEncoder | ~150ms/batch | Strong open-source reranker |
| `voyage-rerank-2` (Voyage AI) | API-based | ~200ms | Best-in-class precision |
| `cohere-rerank-3` (Cohere) | API-based | ~200ms | Strong, especially multilingual |
| LLM-as-reranker (current option) | Generative | 500–2000ms | Highest quality, highest cost |

CrossEncoder rerankers score each (query, chunk) pair jointly — they see both at once, unlike bi-encoders that embed independently. This gives them much higher precision but they cannot be pre-computed, so they only run on the top-K retrieved candidates (typically K=20→rerank→return top 5).

**Generation models (affects answer quality, not retrieval):**

| Model | Notes |
|---|---|
| `llama3.1:8b` (current, local) | Fast, private, good for simple Q&A |
| `llama3.1:70b` | Much better reasoning, needs strong GPU |
| `gpt-4o` | Best answer quality, paid, low latency |
| `claude-sonnet-4-6` | Strong reasoning, good context handling |
| `gemini-1.5-pro` | 1M context window — can stuff entire corpus |

<a id="q115"></a>
**Q115. How would you benchmark and choose between embedding models for this corpus?**

1. **Build a gold dataset** — the existing `GOLD_DATASET` in `test_retrieval_metrics.py` (10 queries) is a start. Expand to 50–100 queries with known relevant sources, covering edge cases: acronyms (PTO), proper nouns (NeuralFlow), multi-hop (manager of team that owns X).

2. **Run the evaluation harness** for each candidate model:
   - Re-ingest with the new model
   - Run `test_retrieval_metrics.py` → collect Hit Rate@5, MRR@5, NDCG@5, mean latency
   - Record index size (storage cost of higher-dimension vectors)

3. **Key metrics to compare:**

| Metric | What it tells you |
|---|---|
| Hit Rate@5 | Does the right document appear at all in top 5? |
| MRR@5 | Is the right document near the top? |
| NDCG@5 | Full quality of the ranked list |
| P95 latency | Worst-case query speed |
| Index size | Storage cost (dim × n_chunks × 4 bytes) |

4. **Decision rule**: prefer the model with highest NDCG@5 among those whose P95 latency stays under your SLA (e.g. 100ms). If two models tie on NDCG, pick the smaller dimension (cheaper storage, faster search).

<a id="q116"></a>
**Q116. At what scale would you move away from PostgreSQL/pgvector to a dedicated vector database?**

pgvector is appropriate up to ~5–10M vectors with HNSW. Beyond that, or when you need:

- **Sub-10ms P99 latency at high QPS** → dedicated vector DBs (Qdrant, Weaviate) are optimised for this; pgvector shares I/O with OLTP workloads.
- **Filtered vector search at scale** → pgvector applies filters post-retrieval; Qdrant/Weaviate apply filters during HNSW traversal (payload indexing), which is far more efficient.
- **Multi-tenant isolation** — dedicated DBs have namespace/collection isolation built in; pgvector requires `WHERE tenant_id = ?` on every query.
- **Distributed horizontal scaling** — pgvector is single-node; Qdrant/Weaviate/Pinecone are distributed.

The advantage of staying on pgvector: single database for both relational data (documents table) and vectors (chunks table), transactional consistency, no extra infrastructure. This project's corpus (hundreds of documents, tens of thousands of chunks) is well within pgvector's sweet spot.

<a id="q116a"></a>
**Q116a. Why aren't we using `pg_textsearch` (Timescale's BM25 extension) instead of `tsvector`/`ts_rank`?**

Short answer: we weren't aware of it at build time, and for a prototype it wasn't a blocker. But it is a meaningful upgrade for production.

**What `pg_textsearch` is:**

`pg_textsearch` (github.com/timescale/pg_textsearch, v1.0.0, production-ready) is a PostgreSQL extension from Timescale that replaces the built-in `tsvector`/`ts_rank` FTS stack with a **BM25-based** index and ranking engine. Same idea — full-text search inside Postgres — but with a better ranking algorithm, faster top-k retrieval, and a simpler query syntax.

**BM25 vs `ts_rank` (TF-IDF):**

`ts_rank` scores based on raw term frequency (TF) and inverse document frequency (IDF). BM25 improves on this with two corrections:
- **Term frequency saturation** (k1 parameter, default 1.2) — the score boost from seeing a term 20× vs 10× is diminished; stops long-document term spam from dominating
- **Length normalisation** (b parameter, default 0.75) — a match in a short chunk scores higher than the same match in a 5000-word chunk, because finding the term in less context is more informative

For a RAG corpus where chunks vary in size (50-token fallback chunks vs 512-token HybridChunker chunks), BM25 length normalisation is directly relevant — `ts_rank` will unfairly favour longer chunks that contain the term more times by chance.

**API comparison:**

```sql
-- Current: tsvector/ts_rank
CREATE INDEX chunks_content_tsv_idx ON chunks USING GIN(content_tsv);

SELECT content, ts_rank(content_tsv, plainto_tsquery('english', $1)) AS score
FROM chunks
WHERE content_tsv @@ plainto_tsquery('english', $1)
ORDER BY score DESC
LIMIT 20;

-- With pg_textsearch: BM25
CREATE INDEX chunks_bm25_idx ON chunks USING bm25(content)
  WITH (text_config='english');

SELECT content
FROM chunks
ORDER BY content <@> 'PTO policy'   -- <@> operator, returns negative BM25 score
LIMIT 20;
```

The `<@>` operator integrates with `LIMIT` via **Block-Max WAND** — an algorithm that skips document blocks that cannot possibly make the top-k, avoiding scoring most of the index. `ts_rank` scores all matched rows before truncating.

**Feature comparison:**

| Feature | `tsvector` / `ts_rank` (current) | `pg_textsearch` BM25 |
|---|---|---|
| Ranking algorithm | TF-IDF (`ts_rank`) | BM25 (industry standard) |
| Top-k optimisation | None — scores all matches | Block-Max WAND — skips non-competitive docs |
| Index build | Single-threaded | Parallel (multi-worker) |
| Query syntax | `@@ plainto_tsquery(...)` | `<@>` operator |
| Tunable parameters | None | k1, b per index |
| Length normalisation | None | Yes (b parameter) |
| Partitioned table support | Yes | Yes |
| Memory management | OS-managed | Configurable DSA limit |
| Installation | Built-in | Extension (`shared_preload_libraries`) |
| PostgreSQL version | All | 17/18 (pre-built binaries) |

**What would change in this codebase:**

```python
# postgres.py — text_search() method

# Remove:
content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
CREATE INDEX ... USING GIN(content_tsv)
WHERE content_tsv @@ plainto_tsquery('english', $1)
ts_rank(content_tsv, plainto_tsquery('english', $1)) as similarity

# Add:
CREATE INDEX chunks_bm25_idx ON chunks USING bm25(content)
  WITH (text_config='english', k1=1.2, b=0.75);

-- Query becomes:
SELECT id, content, metadata, document_id,
       -(content <@> $1) AS similarity   -- negate: <@> returns negative scores
FROM chunks
ORDER BY content <@> $1
LIMIT $2;
```

`content_tsv` column and its GIN index can be dropped entirely — `pg_textsearch` maintains its own BM25 index internally.

**When to switch:**

- If retrieval precision on keyword queries is noticeably poor (BM25 will improve it)
- If the corpus has high chunk-length variance (BM25 length normalisation helps more)
- If text search latency is a bottleneck at scale (Block-Max WAND is faster for large corpora)
- When running PostgreSQL 17+ (pre-built binaries available; PostgreSQL 16 requires building from source)

**Current limitation that `pg_textsearch` does not fix:**

Both `tsvector` and BM25 drop stop words and use stemming — so "PTO" → `'pto'` and "paid time off" → `'paid' 'time' 'off'` still do not match each other. This vocabulary mismatch is why the semantic leg of hybrid search exists. Switching to BM25 improves ranking quality within text search, but does not solve cross-vocabulary retrieval — that remains the job of the embedding model.

<a id="q116b"></a>
**Q116b. What other PostgreSQL text search extensions exist, which does this project use, and what would each add?**

**Overview — all relevant extensions:**

| Extension | Used? | What it does | Best for |
|---|---|---|---|
| `tsvector` / `tsquery` (built-in) | **Yes** | Stemming, stop-word removal, lexeme indexing via GIN | General keyword search, already in every PostgreSQL install |
| `pg_trgm` | **Yes** | Trigram similarity — splits text into 3-char grams, supports fuzzy `%` and `<->` operators | Typo tolerance, fuzzy matching, `LIKE`/`ILIKE` acceleration |
| `pg_textsearch` (Timescale) | No — not on Neon | BM25 ranking via `bm25` index + `<@>` operator, Block-Max WAND top-k | Better ranking quality than `ts_rank`, faster top-k at scale |
| `pg_search` (ParadeDB) | **Yes** | BM25 via `bm25` index + `@@@` operator, also supports fuzzy, phrase, boost queries | Full Elasticsearch-like search inside PostgreSQL |
| `pgvector` | **Yes** | Dense vector storage + IVFFlat/HNSW ANN search | Semantic/embedding-based retrieval |

---

**`pg_trgm` — fuzzy matching**

`pg_trgm` splits text into overlapping 3-character grams (`"NeuralFlow"` → `"neu"`, `"eur"`, `"ura"`, `"ral"`, `"alf"`, `"lfl"`, `"flo"`, `"low"`). Two strings are similar if they share many trigrams.

```sql
CREATE EXTENSION pg_trgm;
CREATE INDEX chunks_trgm_idx ON chunks USING GIN(content gin_trgm_ops);

-- Fuzzy match: "NeuralFow" still matches "NeuralFlow"
SELECT content, similarity(content, 'NeuralFow') AS sim
FROM chunks
WHERE content % 'NeuralFow'   -- % operator: similarity above pg_trgm.similarity_threshold (default 0.3)
ORDER BY sim DESC
LIMIT 10;

-- Also accelerates LIKE/ILIKE queries:
SELECT content FROM chunks WHERE content ILIKE '%neuralflow%';
```

**What it adds vs `tsvector`:**
- `tsvector` does not handle typos — `'neuralFow'` produces the lexeme `'neuralfow'` which does not match `'neuralflow'`. `pg_trgm` handles this naturally via shared trigrams.
- Useful for user-facing search where typos are expected (product names, proper nouns, codes).

**Why it adds little on top of `tsvector` in this project:**
- The semantic search leg already handles vocabulary variation much better than trigrams.
- The corpus is company documents — misspellings in the source are unlikely. User query typos would be caught by the embedding model (cosine similarity is robust to minor variations).
- Adds another index (~same size as GIN tsvector index), another query path, and more complexity in the hybrid merge.

**When to add it:** if users frequently search for proper nouns or product codes with typos, and semantic search is not catching them (e.g. internal tool names, employee IDs, part numbers).

---

**`pg_search` (ParadeDB) — BM25 + Elasticsearch-like queries inside PostgreSQL**

ParadeDB's `pg_search` is the most feature-complete text search extension. It uses the same BM25 algorithm as `pg_textsearch` but adds:

- **Phrase queries**: `"paid time off"` matches the exact phrase, not just individual words
- **Fuzzy term matching**: `fuzzy_term(field=>'content', value=>'neuralfow', distance=>1)` — edit-distance-based fuzzy within BM25
- **Boosted fields**: boost matches in titles over body text
- **Range filters**: combine BM25 score with numeric/date filters in one index scan
- **Highlighting**: return matched snippets with hit terms highlighted
- **`@@@` operator** with a rich query builder API

```sql
CREATE EXTENSION pg_search;
CREATE INDEX chunks_search_idx ON chunks
  USING bm25(id, content, metadata)
  WITH (text_fields='{"content": {"tokenizer": {"type": "default"}}}');

-- Basic BM25 query
SELECT id, content, paradedb.score(id) AS score
FROM chunks
WHERE chunks @@@ 'PTO policy'
ORDER BY score DESC LIMIT 10;

-- Fuzzy query (handles "NeuralFow" → "NeuralFlow")
SELECT id, content
FROM chunks
WHERE chunks @@@ paradedb.fuzzy_term(field=>'content', value=>'NeuralFow')
ORDER BY paradedb.score(id) DESC LIMIT 10;

-- Phrase match
WHERE chunks @@@ paradedb.phrase(field=>'content', phrases=>ARRAY['paid time off'])
```

**What it adds vs `tsvector` + `pg_textsearch`:**

| Feature | `tsvector` | `pg_textsearch` | `pg_search` |
|---|---|---|---|
| BM25 ranking | No (TF-IDF) | Yes | Yes |
| Phrase queries | No | No | Yes |
| Fuzzy term matching | No | No | Yes |
| Field boosting | No | No | Yes |
| Snippet highlighting | `ts_headline()` | No | Yes |
| Query builder API | `plainto_tsquery` | `<@>` operator | Rich Pydantic-style API |
| Maturity | Stable (built-in) | v1.0.0 | Active, production users |

**Why not used in this project:**
- Prototype was built with built-in `tsvector` — sufficient for the NeuralFlow corpus size
- `pg_search` requires ParadeDB-distributed PostgreSQL or installing the extension + `shared_preload_libraries` — heavier operational overhead than a built-in
- The semantic leg already handles fuzzy/vocabulary matching; phrase queries are less critical when chunks are short (≤512 tokens) and focused

**The upgrade path if text search quality becomes a bottleneck:**

```
Current:  tsvector + ts_rank (TF-IDF, no fuzzy, no phrases)
    ↓
Step 1:   pg_textsearch (drop-in BM25 upgrade, minimal code change, better ranking)
    ↓
Step 2:   pg_search / ParadeDB (BM25 + fuzzy + phrase + boosting, if precision still insufficient)
    ↓
Parallel: pg_trgm (add only if users report typo misses on proper nouns/codes)
```

In all cases, the semantic (pgvector) leg remains unchanged — the text search upgrade only affects one half of the hybrid search pipeline.

---

<a id="q116c"></a>
**Q116c. What indexes currently exist on the `chunks` table?**

The following indexes are active in the Neon database as of the current deployment. Each serves a different search path in the hybrid retrieval pipeline:

| Index | Type | Column(s) | Search path |
|---|---|---|---|
| `chunks_pkey` | btree (unique) | `id` | Primary key lookups |
| `chunks_document_id_idx` | btree | `document_id` | JOIN to `documents`, cascade deletes |
| `chunks_embedding_idx` | ivfflat (cosine) | `embedding` | Semantic search (`semantic_search`) |
| `chunks_content_tsv_idx` | GIN | `content_tsv` | Full-text search (`text_search`) |
| `chunks_content_trgm_idx` | GIN (trigram) | `content` | Fuzzy search (`fuzzy_search`) |
| `chunks_bm25_idx` | bm25 (pg_search) | `id`, `content` | BM25 search (`bm25_search`) |

All four search indexes feed into `hybrid_search` via parallel `asyncio.gather`, then merge through Reciprocal Rank Fusion (RRF).

<a id="q116d"></a>
**Q116d. How does re-indexing happen on the fly?**

All indexes except IVFFlat are maintained automatically by PostgreSQL on every `INSERT` — no manual step needed:

- **btree** indexes update instantly on insert.
- **GIN tsvector** (`content_tsv`) is a generated column — PostgreSQL recomputes and indexes it automatically on every insert/update.
- **GIN trigram** (`content gin_trgm_ops`) — new content trigrams are added to the inverted index on insert.
- **BM25** (`pg_search`) — uses a memtable architecture: new rows land in an in-memory inverted index first, then spill to disk automatically.

**IVFFlat is the exception.** Its Voronoi centroids are fixed at build time. New vectors are assigned to the nearest existing centroid, which works fine for small growth — but if the chunk count grows significantly beyond what the index was built at, recall degrades because the centroid layout no longer reflects the data distribution.

This project handles it automatically: after every `add()` call, `PostgresHybridStore` checks if the total chunk count has reached 3× the count recorded at index build time. If so, it runs:

```sql
REINDEX INDEX CONCURRENTLY chunks_embedding_idx
```

`CONCURRENTLY` means the rebuild happens without locking reads or writes — queries continue uninterrupted during the reindex. After completion, `_ivfflat_index_build_count` is reset to the new count, restarting the 3× window.

<a id="q116e"></a>
**Q116e. How are new documents auto-ingested and re-indexed?**

New documents flow through the same pipeline as the initial ingest — no special re-index step is needed:

1. **Ingest** — `DocumentIngestionPipeline.ingest_document()` converts the file (Docling), chunks it, generates embeddings, and calls `PostgresHybridStore.add()`.
2. **Insert** — `add()` runs `executemany` to batch-insert all chunks into the `chunks` table.
3. **Auto-index** — PostgreSQL automatically updates all five indexes (btree, GIN tsvector, GIN trigram, BM25, IVFFlat) on insert.
4. **IVFFlat growth check** — after the insert, `add()` checks if the 3× threshold has been crossed and triggers `REINDEX CONCURRENTLY` if needed.
5. **Duplicate detection** — `ingest_document()` hashes the file content and skips re-ingestion if the hash matches what's already stored (`get_document_hash()`). Only changed or new documents are processed.

The result: pointing the pipeline at a folder of new or updated documents is all that's required. Retrieval immediately reflects the new content.

<a id="q116f"></a>
**Q116f. Which tests are currently failing and what needs to be done to fix them?**

As of the current test run: **1 test failing, 147 passing, 12 skipped.**

| Test | File | Failure reason | Fix required |
|---|---|---|---|
| `test_agent_run_specific_query` | `test_rag_agent.py` | The chunk containing "NeuralFlow AI has 47 employees" is not present in the indexed documents. The LLM also calls the search tool with suboptimal parameters (`query=''`, `match_count=1`) due to the small model (llama3.2:3b). | Re-ingest the NeuralFlow AI company overview document that contains employee headcount, **or** update the test assertion to match data that is actually in the DB. |

**Root cause detail:**

The test asks `"How many employees does NeuralFlow AI have?"` and asserts the response contains a number. The DB has 194 chunks from 13 documents — all customer case studies — none containing the company's own headcount. The fact was never ingested.

The test is also sensitive to the LLM's tool-calling quality. With `llama3.2:3b`, the model sometimes calls `search_knowledge_base` with an empty query string or `match_count=1`, which limits what retrieval can return even if the data were present. Switching to a larger model (e.g. `llama3.1:8b` or any OpenAI model) would make tool calls more reliable.

**Skipped tests (12):** all integration tests that require live services not running in CI (Ollama server, Langfuse, Mem0 with non-default config). These are expected skips, not failures.

<a id="q116g"></a>
**Q116g. How do I inspect what's actually stored in the `chunks` table?**

Run this in the Neon console (or via `psql`) to see a sample of real rows including the generated tsvector and embedding columns:

```sql
SELECT
    id,
    document_id,
    chunk_index,
    token_count,
    LEFT(content, 100)        AS content_preview,
    content_tsv::text         AS tsv_lexemes,
    LEFT(embedding::text, 80) AS embedding_preview
FROM chunks
LIMIT 3;
```

**Example output:**

| column | example value |
|---|---|
| `id` | `1e18fff3-1866-42dd-9607-9bb3a5976cf4` |
| `document_id` | `f1b01e53-dd5d-4eee-b3bd-853e96fd6e07` |
| `chunk_index` | `18` |
| `token_count` | `22` |
| `content_preview` | `Challenges & Learnings\nWhile Q4 was highly successful, we encountered...` |
| `tsv_lexemes` | `'challeng':1,11 'encount':9 'high':6 'insight':15 'learn':2 'provid':13` |
| `embedding_preview` | `[0.0123, -0.0456, 0.0789, ...]` |

Key things visible in the output:

- **`content_tsv`** is already populated with stemmed lexemes and position tags (e.g. `'challeng':1,11` means the stem of "challenges" appears at positions 1 and 11). This is the generated column — PostgreSQL computed it automatically on insert, no application code involved.
- **Stop words are dropped** — "while", "was", "we", "that" don't appear in `tsv_lexemes`.
- **Stems, not words** — `'challeng'` not `"challenges"`, `'encount'` not `"encountered"`.
- **`embedding`** is a high-dimensional float vector (768 dims for `nomic-embed-text`). Truncated here for readability.

<a id="q116h"></a>
**Q116h. Why doesn't the SELECT query show trigram data? How do I see what the trigram index stores?**

The trigram index (`chunks_content_trgm_idx`) is just an **index**, not a stored column. There is no `content_trgm` column to select — unlike `content_tsv` which is a generated column physically written to disk, trigrams are computed internally by PostgreSQL when building the index and at query time. They are never stored in a retrievable form.

To **see the trigrams for any string**, use `show_trgm()`:

```sql
SELECT show_trgm('NeuralFlow AI');
```

Example output:
```
{"  a","  n","ai ","al ","eur","flo","low","neu","ral","ura"}
```

Each 3-character gram includes padding spaces at word boundaries (`"  n"` = start of "NeuralFlow"). Two strings are considered similar if they share enough of these grams.

To **see the index working at query time**, run a fuzzy match directly:

```sql
SELECT
    LEFT(content, 100)                     AS content_preview,
    word_similarity('NeuralFlow', content)  AS trgm_score
FROM chunks
WHERE word_similarity('NeuralFlow', content) > 0.2
ORDER BY trgm_score DESC
LIMIT 5;
```

This shows which chunks match a fuzzy query and their similarity scores — exactly what `fuzzy_search()` in `postgres.py` runs under the hood. The `0.2` threshold is tunable: lower catches more (with more noise), higher is stricter.

---

## Data Model

<a id="q117"></a>
**Q117. What does the PostgreSQL data model look like — entity diagram and sample records?**

**Entity diagram:**

```
┌─────────────────────────────────────────────────────┐
│                     documents                       │
├──────────────┬──────────────────────────────────────┤
│ id           │ UUID  PK  gen_random_uuid()          │
│ title        │ TEXT  NOT NULL                       │
│ source       │ TEXT  NOT NULL  UNIQUE               │  ← file path / URL
│ content      │ TEXT                                 │  ← full markdown text
│ metadata     │ JSONB  DEFAULT '{}'                  │  ← title, author, hash…
│ created_at   │ TIMESTAMPTZ  DEFAULT NOW()           │
└──────────────┴──────────────────────────────────────┘
        │ 1
        │ ON DELETE CASCADE
        │ N
┌─────────────────────────────────────────────────────┐
│                      chunks                         │
├──────────────┬──────────────────────────────────────┤
│ id           │ UUID  PK  gen_random_uuid()          │
│ document_id  │ UUID  FK → documents.id              │
│ content      │ TEXT  NOT NULL                       │  ← contextualized chunk text
│ embedding    │ vector(768)                          │  ← pgvector float32 array
│ chunk_index  │ INTEGER  NOT NULL                    │  ← 0-based order within doc
│ metadata     │ JSONB  DEFAULT '{}'                  │  ← chunk_method, token_count…
│ token_count  │ INTEGER                              │
│ created_at   │ TIMESTAMPTZ  DEFAULT NOW()           │
│ content_tsv  │ tsvector  GENERATED ALWAYS STORED    │  ← auto-updated from content
└──────────────┴──────────────────────────────────────┘

Indexes:
  chunks_embedding_idx    USING ivfflat (embedding vector_cosine_ops)  lists=100
  chunks_content_tsv_idx  USING GIN (content_tsv)
  chunks_document_id_idx  USING btree (document_id)
  documents_source_idx    USING btree (source)
```

**Sample `documents` row:**

```
id          │ a3f1c2d4-88b1-4e2a-9c3f-1234567890ab
title       │ Employee Handbook
source      │ rag/documents/team-handbook.md
content     │ # Employee Handbook\n\n## Benefits\n\nEmployees receive 20 days
            │ of PTO per year...[full markdown, ~8000 chars]
metadata    │ {
            │   "file_type": "md",
            │   "content_hash": "d41d8cd98f00b204e9800998ecf8427e",
            │   "chunk_count": 18,
            │   "title": "Employee Handbook"
            │ }
created_at  │ 2025-03-15 10:23:41+00
```

**Sample `chunks` row:**

```
id           │ 7b2e9f13-cc4a-4d88-b901-abcdef012345
document_id  │ a3f1c2d4-88b1-4e2a-9c3f-1234567890ab   ← FK to documents row above
content      │ Benefits > Time Off Policy
             │
             │ Employees are entitled to 20 days of paid time off (PTO) per
             │ calendar year. PTO accrues monthly and unused days roll over
             │ up to a maximum of 10 days.
embedding    │ [0.0213, -0.1047, 0.0831, 0.0492, -0.2103, 0.1774, ...] (768 floats)
chunk_index  │ 3
metadata     │ {
             │   "chunk_method": "hybrid",
             │   "has_context": true,
             │   "document_source": "rag/documents/team-handbook.md",
             │   "document_title": "Employee Handbook"
             │ }
token_count  │ 87
created_at   │ 2025-03-15 10:23:42+00
content_tsv  │ 'benefit':1 'calendar':11 'day':8,16 'entitl':5 'maximum':20
             │ 'month':14 'off':9 'paid':7 'polic':3 'pto':10 'roll':17
             │ 'time':8 'unus':15 'year':12
             │                              ← auto-generated, stemmed lexemes
```

**Key observations:**

- `content` in the chunk is the *contextualized* text (`"Benefits > Time Off Policy\n\n..."`) — the heading breadcrumb is baked in, so the embedding captures the topic context.
- `embedding` is a `vector(768)` — 768 × 4 bytes = 3,072 bytes per chunk row purely for the vector.
- `content_tsv` is computed automatically by PostgreSQL on every INSERT/UPDATE — you never write to it directly. The lexemes are stemmed (`'entitl'` for "entitled", `'polic'` for "policy") and stop words (`"are"`, `"to"`, `"of"`) are dropped.
- `metadata` JSONB is flexible — the pipeline writes `chunk_method` and `has_context` here; YAML frontmatter fields from markdown files also land here.
- `chunk_index` preserves the original document order, useful for re-ranking by position or reconstructing document flow.

**How a hybrid search touches these tables:**

```sql
-- Semantic leg: cosine similarity on embedding
SELECT id, content, metadata, document_id,
       1 - (embedding <=> $1::vector) AS score
FROM chunks
ORDER BY embedding <=> $1::vector
LIMIT 20;

-- Text leg: tsvector full-text ranking
SELECT id, content, metadata, document_id,
       ts_rank(content_tsv, plainto_tsquery('english', $2)) AS score
FROM chunks
WHERE content_tsv @@ plainto_tsquery('english', $2)
ORDER BY score DESC
LIMIT 20;

-- Results merged in Python via RRF, then joined back to documents for title/source
```

---

## Pydantic AI Internals

<a id="q118"></a>
**Q118. How does the Pydantic AI agent loop work in this codebase — agent creation, RunContext, deps, and the tool execution cycle?**

**1. Agent creation (`rag_agent.py:227`)**

```python
agent = PydanticAgent(get_llm_model(), system_prompt=MAIN_SYSTEM_PROMPT)
```

`PydanticAgent` is instantiated once at module level — it is stateless and reused across all requests. It holds:
- the LLM model configuration (`OpenAIChatModel` pointing at Ollama/OpenAI)
- the system prompt string
- the registry of tools (populated by `@agent.tool` decorators below it)

No database connections or user state live here — those are in `RAGState`.

---

**2. Registering a tool (`rag_agent.py:230`)**

```python
@agent.tool
async def search_knowledge_base(
    ctx: PydanticRunContext,
    query: str,
    match_count: int | None = 5,
    search_type: str | None = "hybrid",
) -> str:
    """Search the knowledge base for relevant information."""
    ...
```

`@agent.tool` does three things at decoration time (before any request):
- Registers `search_knowledge_base` in the agent's internal tool registry
- Introspects the signature and docstring to build a JSON Schema (see Q47a for details)
- The first parameter `ctx: PydanticRunContext` is **always** the run context — it is stripped from the JSON Schema sent to the LLM (the LLM never sees it or fills it in)

The remaining parameters (`query`, `match_count`, `search_type`) become the tool's callable arguments that the LLM fills in.

---

**3. Starting a run — passing `deps`**

```python
# Simple run — no deps, no shared state
result = await agent.run("What does NeuralFlow AI do?")

# Full run — with RAGState for connection reuse and user personalisation
state = RAGState(user_id="alice")
result = await agent.run("What is the PTO policy?", deps=state)
await state.close()
```

`deps` is arbitrary — any Python object. Pydantic AI doesn't care what it is; it just makes it available inside every tool call via `ctx.deps`. This is the dependency injection mechanism. The `deps_type` annotation on the agent (if set) enables type checking, but it is optional.

In `traced_agent_run` (`rag_agent.py:362`) this is done automatically:

```python
state = RAGState(user_id=user_id)
result = await agent.run(query, deps=state)
await state.close()
```

---

**4. The `RunContext` object inside a tool**

When Pydantic AI calls `search_knowledge_base`, it injects a `RunContext` as the first argument. This object exposes:

| Attribute | Type | What it contains |
|---|---|---|
| `ctx.deps` | `Any` (here: `RAGState \| None`) | Whatever was passed as `deps` to `agent.run()` |
| `ctx.model` | `Model` | The LLM model instance |
| `ctx.usage` | `Usage` | Token counts so far in this run |
| `ctx.prompt` | `str` | The original user prompt |
| `ctx.tool_call_id` | `str` | Unique ID of this specific tool invocation |

In this codebase only `ctx.deps` is used:

```python
deps = ctx.deps
state = deps if isinstance(deps, RAGState) else getattr(deps, "state", None)
```

The `isinstance` guard handles two call patterns:
- `agent.run(query, deps=state)` → `ctx.deps` is a `RAGState` directly
- `agent.run(query, deps=some_wrapper)` → `ctx.deps` is a wrapper with a `.state` attribute

---

**5. The full agent loop — what happens during `agent.run()`**

```
agent.run("What is the PTO policy?", deps=state)
│
├─► Build messages list:
│     [ system_prompt, user: "What is the PTO policy?" ]
│
├─► POST /v1/chat/completions  (with tools=[search_knowledge_base schema])
│
├─► LLM responds with tool call:
│     tool_calls: [{ name: "search_knowledge_base", args: '{"query":"PTO policy"}' }]
│
├─► Pydantic AI sees tool_calls in response → does NOT return to caller yet
│
├─► Validates args against JSON Schema ──► calls search_knowledge_base(ctx, query="PTO policy")
│       │
│       ├─► ctx.deps → RAGState → retriever.retrieve_as_context("PTO policy")
│       ├─► Hits PostgreSQL (semantic + text search, RRF merge)
│       └─► Returns formatted string of top-5 chunks
│
├─► Appends tool result to messages:
│     [ system_prompt, user: "...", assistant: tool_call, tool: "Benefits > PTO\n\nEmployees..." ]
│
├─► POST /v1/chat/completions  (second LLM call, same tools available)
│
├─► LLM responds with plain text (no tool calls):
│     "Employees receive 20 days of PTO per year, accruing monthly..."
│
└─► agent.run() returns AgentResult(output="Employees receive 20 days...")
```

The loop runs until the LLM produces a response with **no tool calls**. If the LLM calls the tool again (e.g. a second search for a follow-up detail), Pydantic AI executes it and loops again. The loop is bounded by `max_result_retries` (default: 1 retry on validation error) and implicitly by the LLM's own decision to stop calling tools.

---

**6. `RAGState` — why lazy initialisation matters**

```python
class RAGState(BaseModel):
    user_id: str | None = None

    _store: PostgresHybridStore | None = PrivateAttr(default=None)
    _retriever: Retriever | None = PrivateAttr(default=None)
    _initialized: bool = PrivateAttr(default=False)
    _init_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    async def get_retriever(self) -> Retriever:
        async with self._init_lock:
            if not self._initialized:
                self._store = PostgresHybridStore()
                await self._store.initialize()   # creates asyncpg pool HERE
                self._retriever = Retriever(store=self._store)
                self._initialized = True
        return self._retriever
```

The asyncpg pool is created **inside** `get_retriever()`, which is called **inside** the tool, which runs in the same event loop as `agent.run()`. This is intentional — asyncpg pools are bound to the event loop that created them. If the pool were created in `RAGState.__init__` (which might run in a different loop, e.g. Streamlit's startup loop), every query would fail with "pool attached to a different loop". The `_init_lock` prevents double-initialisation if two tool calls happen concurrently.

---

**7. Message history for multi-turn conversations**

```python
# First turn
result1 = await agent.run("What is the PTO policy?", deps=state)

# Second turn — pass previous messages for context
result2 = await agent.run(
    "How does it compare to the sick leave policy?",
    message_history=result1.new_messages(),
    deps=state,
)
```

`result.new_messages()` returns the messages from that run (user prompt + tool calls + tool results + assistant response). Passing them as `message_history` on the next call appends them before the new user message, giving the LLM full conversation context. This is how `traced_agent_run` supports multi-turn chat in the Streamlit UI.

---

## Production Readiness

<a id="q120"></a>
**Q120. What are all the changes needed to make this RAG system production-ready?**

The current codebase is a well-structured prototype — async throughout, typed, tested, with observability hooks. But it has significant gaps before it can serve real users reliably. Changes are grouped by category.

---

### 1. Authentication & Authorisation

**Current state:** No auth. Anyone who can reach the Streamlit UI or call `agent.run()` gets full access to everything.

| Change | Where | Detail |
|---|---|---|
| JWT/session auth | Streamlit UI + API layer | Validate bearer token on every request; extract `user_id` and `tenant_id` from claims |
| Role-based access | New `roles` table | `admin` (can ingest), `reader` (query only), `superadmin` (cross-tenant) |
| API key management | New `api_keys` table | Hashed keys for programmatic access; rate-limited per key |
| Secure secret storage | Config | Move all keys out of `.env` into a secrets manager (AWS Secrets Manager, Vault, Doppler) — `.env` is fine locally, not in prod |

---

### 2. Multi-tenancy

**Current state:** Single shared corpus, no isolation (see Q78 for full detail).

| Change | Where | Detail |
|---|---|---|
| Add `tenant_id` column | `documents`, `chunks` tables | `TEXT NOT NULL` with B-tree index |
| Row-Level Security | PostgreSQL | RLS policies + `SET LOCAL app.tenant_id` per connection |
| Tenant provisioning API | New service | Create tenant → register in tenant registry → confirm schema ready |
| Pass `tenant_id` through stack | `RAGState` → `PostgresHybridStore` → every query | Extracted from the auth token, never user-supplied |

---

### 3. API Layer

**Current state:** Entry points are Streamlit UI and direct Python imports. No HTTP API.

| Change | Detail |
|---|---|
| FastAPI app | Wrap `traced_agent_run` in `POST /chat`, ingestion in `POST /ingest`, health in `GET /health` |
| Request/response models | Pydantic models for all endpoints — input validation, OpenAPI docs auto-generated |
| Async request handling | FastAPI + uvicorn already async-compatible with asyncpg pool |
| Streaming responses | `agent.run_stream()` → `StreamingResponse` for real-time token output in the UI |
| Versioned endpoints | `/v1/chat`, `/v1/ingest` — enables non-breaking API evolution |

---

### 4. Ingestion Pipeline

**Current state:** CLI only (`python -m rag.main --ingest`), synchronous processing, no job queue, `clean_before_ingest=True` takes the system down briefly.

| Change | Detail |
|---|---|
| Background job queue | Celery + Redis or arq — ingest jobs are CPU/network-heavy, should not block API requests |
| Job status tracking | `ingestion_jobs` table: `job_id`, `status`, `progress`, `error`; poll via `GET /ingest/{job_id}` |
| Zero-downtime re-index | Shadow table swap (Q106) — ingest into `documents_new`/`chunks_new`, then `ALTER TABLE RENAME` atomically |
| Deduplication | MD5 hash check before ingestion (already exists for incremental); add SHA-256 for security-sensitive dedup |
| Parallel document processing | Semaphore-bounded `asyncio.to_thread` for `DocumentConverter` (Q105) |
| Webhook / event notification | Notify downstream systems when ingestion completes |
| Max file size guard | Reject files above configurable limit before conversion begins |
| Virus scanning | ClamAV scan uploaded files before processing |

---

### 5. Database

**Current state:** Single asyncpg pool, `pool_min=1 / pool_max=10`, IVFFlat index with `lists=100`, no migrations, no connection SSL enforcement.

| Change | Detail |
|---|---|
| Schema migrations | Alembic — version-controlled DDL changes; never `CREATE TABLE IF NOT EXISTS` in application code in prod |
| IVFFlat → HNSW | Better query latency at scale; tune `m` and `ef_construction` for recall/speed trade-off |
| Connection SSL | Enforce `sslmode=require` in `DATABASE_URL`; already supported by asyncpg |
| Pool tuning | `pool_min` = number of worker processes; `pool_max` = based on PostgreSQL `max_connections` limit |
| Read replica | Route `semantic_search` and `text_search` to a read replica; writes go to primary |
| Soft deletes | Add `deleted_at TIMESTAMPTZ` to `documents`; filter `WHERE deleted_at IS NULL`; hard delete via scheduled job |
| Connection timeout / retry | Set `command_timeout` and add retry logic with exponential backoff for transient connection failures |
| pgBouncer | Connection pooler in front of PostgreSQL to handle burst traffic without exhausting `max_connections` |

---

### 6. Embedding Model Management

**Current state:** Model name and dimension are env vars. No record of which model was used per document. Changing the model silently breaks all existing vectors.

| Change | Detail |
|---|---|
| Store model metadata | Add `embedding_model TEXT`, `embedding_dimension INT` to `documents` table at ingestion time |
| Model version guard | On startup, check that `settings.embedding_model` matches the model recorded in the DB; refuse to query if mismatched |
| Model migration tooling | Script to re-embed all chunks with a new model into a shadow table, then swap (zero-downtime model upgrade) |

---

### 7. Observability & Alerting

**Current state:** Langfuse tracing is optional and off by default. No metrics, no structured logging, no alerts.

| Change | Detail |
|---|---|
| Enable Langfuse in prod | `LANGFUSE_ENABLED=true`; trace every agent run, tool call, and ingestion job |
| Structured logging | Replace `logging.info(f"...")` with structured JSON logs (`structlog` or `python-json-logger`); include `tenant_id`, `user_id`, `trace_id` in every log line |
| Prometheus metrics | Expose `/metrics` endpoint: query latency histogram, retrieval hit rate, ingestion throughput, error rate, pool connection count |
| Alerting | Alert on: P95 query latency > 2s, error rate > 1%, embedding API failure rate > 5%, ingestion queue depth > 100 |
| Distributed tracing | Add OpenTelemetry spans around DB queries and embedding API calls for end-to-end request tracing |
| Health checks | `GET /health` returns DB connectivity, embedding API reachability, LLM API reachability — used by load balancer |

---

### 8. Reliability & Error Handling

**Current state:** Tool errors return a string `"Error searching knowledge base: ..."` to the LLM. No retries, no circuit breakers, no graceful degradation.

| Change | Detail |
|---|---|
| Retry with backoff | Wrap embedding API and LLM calls with `tenacity` — retry on 429/503 with exponential backoff + jitter |
| Circuit breaker | If embedding API is down, fast-fail new requests rather than queuing them; fall back to text-search-only mode |
| Graceful degradation | If semantic search fails → fall back to text search only; if reranker fails → return pre-rerank order |
| Timeout enforcement | Set explicit timeouts on all external calls: embedding API (5s), LLM (30s), DB query (10s) |
| Dead letter queue | Failed ingestion jobs go to a DLQ for manual inspection rather than silently dropped |

---

### 9. Security

**Current state:** No input sanitisation beyond the table name validator in `settings.py`. API keys in `.env`.

| Change | Detail |
|---|---|
| Input sanitisation | Validate and truncate user query length (e.g. max 1000 chars) before embedding |
| Prompt injection defence | Strip or escape content that looks like system prompt overrides before passing to LLM |
| `plainto_tsquery` already safe | Already used (Q22) — no SQL injection risk from user queries |
| Secrets rotation | Rotate DB password, API keys on schedule; store in secrets manager with auto-rotation |
| Dependency scanning | Add `pip-audit` or `safety` to CI to catch known CVEs in dependencies |
| HTTPS only | TLS termination at load balancer; redirect HTTP → HTTPS |
| CORS policy | Restrict allowed origins in the FastAPI CORS middleware |

---

### 10. Testing

**Current state:** 118 tests, 12 skipped. No load tests, no contract tests, no chaos tests.

| Change | Detail |
|---|---|
| Expand gold dataset | 10 queries → 100+ with edge cases: acronyms, multi-hop, negation, out-of-corpus queries |
| Load testing | `locust` or `k6` — simulate 100 concurrent users, measure P95 latency and error rate under load |
| Contract tests | Pin embedding API response schema — detect breaking changes before deployment |
| Chaos testing | Kill the DB connection mid-request, kill the embedding API — verify graceful degradation |
| CI pipeline | GitHub Actions: lint (ruff) → unit tests → integration tests (with test DB) → load test (nightly) |

---

### 11. Deployment

**Current state:** Runs locally via `python -m rag.main` or `streamlit run`. No containerisation, no CI/CD.

| Change | Detail |
|---|---|
| Dockerfile | Multi-stage build: `python:3.13-slim` base, install deps, copy source, run as non-root user |
| Docker Compose | `app` + `postgres` + `ollama` services for local dev parity |
| Kubernetes / managed container | Deploy FastAPI app as a `Deployment` with HPA (scale on CPU/request rate); separate `Job` for ingestion |
| CI/CD pipeline | On merge to main: build image → run tests → push to registry → deploy to staging → smoke test → promote to prod |
| Environment promotion | `dev` → `staging` → `prod` with separate Neon branches or databases per environment |
| Graceful shutdown | Handle `SIGTERM`: stop accepting new requests, drain in-flight requests, close asyncpg pool, flush Langfuse |

---

### Priority order for a startup moving from prototype to production:

```
Phase 1 — Make it safe to expose:
  Auth (JWT) → HTTPS → Input sanitisation → Secrets in vault

Phase 2 — Make it multi-user:
  FastAPI layer → Multi-tenancy (RLS) → Structured logging → Health checks

Phase 3 — Make it reliable:
  Retries + circuit breakers → Background ingestion queue → Alembic migrations → Monitoring + alerts

Phase 4 — Make it scalable:
  IVFFlat → HNSW → Read replica → Connection pooler → Load testing → HPA
```

---

## Chunking Strategy

<a id="q122"></a>
**Q122. Is this project using semantic chunking or fixed-size chunking with overlaps?**

Neither purely — the active strategy is **structure-aware chunking** via Docling's `HybridChunker`, with a **fixed-size sliding window fallback** for plain text. A fully-implemented semantic chunker exists in the codebase but is not wired into the pipeline.

---

**Primary path — `DoclingHybridChunker` (structure-aware)**

Used for: PDF, DOCX, and all formats handled by `DocumentConverter`.

`pipeline.py:168` calls `create_chunker(config)` from `docling.py`, which always returns a `DoclingHybridChunker`. This wraps Docling's `HybridChunker` and splits documents at **structural boundaries** extracted from the `DoclingDocument` object — section headings, paragraph breaks, table boundaries — not at fixed character counts and not by measuring embedding similarity between sentences.

"Hybrid" in Docling's terminology means: structural layout signals (from `DoclingDocument`) + token budget enforcement (`max_tokens=512`). A chunk grows to fill a structural section, capped at 512 tokens. Tables are always kept atomic — never split mid-row.

After splitting, `contextualize()` prepends the heading breadcrumb (`"Benefits > Time Off Policy\n\n..."`) to each chunk before embedding. This is unique to the structural approach — semantic and fixed-size chunkers have no heading hierarchy to prepend.

`chunk_method` in chunk metadata is set to `"hybrid"` for these chunks.

---

**Fallback path — `_simple_fallback_chunk` (fixed-size with overlap)**

Triggered for: `.txt` files, plain markdown without structure, or when `DocumentConverter` fails.

This is a sliding window over characters:
- `chunk_size = 1000` chars
- `chunk_overlap = 200` chars (20% overlap)
- Cuts at the nearest sentence boundary (`.`, `!`, `?`, `\n`) within 200 chars of the target end — so it is sentence-aware but not semantically aware

`chunk_method` is set to `"simple_fallback"` for these chunks.

---

**What exists but is NOT active — `semantic.py`**

`rag/ingestion/chunkers/semantic.py` contains two fully-implemented semantic chunkers that are **never instantiated** by the pipeline:

**`SemanticChunker`** — threshold-based:
1. Splits document into sentences (regex on `.`, `!`, `?`)
2. Embeds every sentence with `all-MiniLM-L6-v2` via `SentenceTransformer`
3. Computes cosine similarity between each adjacent sentence pair
4. Starts a new chunk when similarity drops below `similarity_threshold=0.5` AND the current chunk has at least `min_sentences=2`
5. Also splits when `max_sentences=15` is reached regardless of similarity

**`GradientSemanticChunker`** — percentile-based (more adaptive):
1. Same sentence embedding step
2. Computes all adjacent sentence similarities across the document
3. Sets the split threshold at the bottom `percentile_threshold=25`th percentile of all similarity scores — adapts to each document's style rather than using a fixed 0.5 cutoff
4. Also enforces `min_chunk_size=100` / `max_chunk_size=2000` chars

Neither is wired into `pipeline.py`. `create_chunker()` in `docling.py:302` always returns `DoclingHybridChunker`.

---

**Why structure-aware beats semantic chunking for this corpus:**

| | Structure-aware (active) | Semantic (implemented, unused) | Fixed-size (fallback only) |
|---|---|---|---|
| Chunk boundary quality | Excellent — section = natural unit of meaning | Good — topic shifts detected | Poor — arbitrary cuts |
| Table handling | Atomic — `TableItem` never split | May split mid-table | Splits mid-table |
| Heading context | `contextualize()` prepends breadcrumb | No heading hierarchy available | No heading hierarchy |
| Token budget | Enforced via HuggingFace tokenizer | Approximated as `len(text) // 4` | Exact character count |
| Extra cost at ingest | None beyond conversion | Embeds every sentence to find boundaries | None |
| Works on plain text | No → falls back to sliding window | Yes | Yes |
| `chunk_method` metadata | `"hybrid"` | `"semantic"` / `"gradient_semantic"` | `"simple_fallback"` |

For structured company documents (handbooks, policy docs, architecture docs), the section heading is the strongest available signal for chunk boundaries. Semantic chunking pays the cost of embedding every sentence at ingest time and still produces worse boundaries than just following the document's own structure.

**When semantic chunking would be preferable:**
- Corpus is raw, unstructured prose with no headings (e.g. books, emails, transcripts)
- Documents are not in a format Docling can parse (unusual binary formats)
- You want topic-coherent chunks that span multiple short paragraphs under the same heading

---

## Performance Tuning

<a id="q121"></a>
**Q121. What are all the tunables in this RAG system and how should they be set for performance?**

Every tunable is grouped by the stage of the pipeline it affects. For each one: what it does, the current default, the effect of increasing/decreasing it, and a recommended starting point.

---

### 1. Chunking

Chunking is the highest-leverage tunable. It affects every downstream stage — embedding quality, text search recall, storage size, and retrieval latency.

**`max_tokens`** — `settings.py` / `pipeline.py`, default `512`

The hard ceiling on chunk size measured in tokens (using the `all-MiniLM-L6-v2` tokenizer). `HybridChunker` never produces a chunk larger than this.

| Value | Effect |
|---|---|
| Too small (< 128) | Chunks lack context — embeddings are poor signal, precision drops |
| 128–256 | Good for FAQ-style docs with short, self-contained answers |
| 256–512 (current) | Best general-purpose range for most document types |
| > 512 | Truncated by the embedding model's context window — content silently cut |

**Rule:** set `max_tokens` to ≤ the embedding model's max input tokens. `nomic-embed-text` supports 8192 tokens, so 512 is conservative — you could go to 1024 for denser chunks if precision allows.

---

**`chunk_size`** — `pipeline.py`, default `1000` (characters)

Used only by the **simple fallback chunker** (plain text, `.txt`, failed PDF conversions). The sliding window width in characters.

**`chunk_overlap`** — `pipeline.py`, default `200` (characters)

Overlap between consecutive fallback chunks. Prevents answers that straddle a boundary from being missed entirely.

| overlap=0 | Fewer chunks, no duplicate content, risk of boundary-straddling misses |
| overlap=100–200 | Balanced — one sentence of shared context between chunks |
| overlap > 50% of chunk_size | Massive redundancy, index bloat, duplicate results in retrieval |

**Recommendation:** keep overlap at 10–20% of chunk_size. For 1000-char chunks, 100–200 char overlap is appropriate.

---

**`merge_peers`** — `docling.py`, default `True`

Controls whether `HybridChunker` merges small adjacent chunks at the same heading level. Turning it off gives maximum granularity — every paragraph becomes its own chunk.

Turn off if: your corpus has very long sections and you want finer retrieval granularity. Turn on (keep default) if: you want coherent multi-paragraph chunks with better embedding signal.

---

**`TOKENIZER_MODEL`** — `docling.py:105`, hardcoded `"sentence-transformers/all-MiniLM-L6-v2"`

The tokenizer used to measure chunk token counts. Should match the embedding model's tokenizer for accurate boundary decisions (see Q97 for the mismatch risk). Not currently configurable via `.env` — requires a code change.

---

### 2. Embedding Model

The embedding model is the single biggest determinant of semantic search quality.

**`embedding_model`** — `settings.py`, default `"nomic-embed-text:latest"`
**`embedding_dimension`** — `settings.py`, default `768`

Must be changed together. Changing the model after ingestion requires a full re-ingest — all existing vectors become invalid.

| Model | Dim | Speed | Quality | Notes |
|---|---|---|---|---|
| `nomic-embed-text` (current) | 768 | Fast (local) | Good | Local via Ollama, no cost |
| `text-embedding-3-small` | 1536 | Fast (API) | Strong | OpenAI paid, MTEB top tier |
| `text-embedding-3-large` | 3072 | Moderate | Best OpenAI | 2× cost, higher storage |
| `voyage-3` | 1024 | Fast (API) | Excellent for RAG | Purpose-built for retrieval |
| `bge-large-en-v1.5` | 1024 | Moderate (local) | Strong open-source | Runs locally, no API cost |
| `e5-mistral-7b-instruct` | 4096 | Slow (7B model) | Best open-source | GPU required |

**Storage impact:** each chunk stores `embedding_dimension × 4` bytes. At 768 dim and 50,000 chunks: 768 × 4 × 50,000 = **150MB** just for vectors. At 3072 dim: **600MB**.

**When to change:** run `test_retrieval_metrics.py` with different models and compare NDCG@5. Pick the highest-quality model whose P95 latency stays under your SLA.

---

### 3. Vector Index (pgvector)

**`ivfflat.lists`** — `postgres.py:187`, default `100`

Number of Voronoi cells the IVFFlat index partitions vectors into. pgvector recommendation: `lists ≈ rows / 1000` for up to 1M rows, `sqrt(rows)` beyond that.

| Corpus size | Recommended lists |
|---|---|
| < 10,000 chunks | 10–50 |
| 10,000–100,000 chunks | 100 (current default is fine) |
| 100,000–1,000,000 chunks | 300–1000 |
| > 1,000,000 chunks | Migrate to HNSW |

Changing `lists` requires dropping and recreating the index (`CREATE INDEX ... USING ivfflat`).

---

**`ivfflat.probes`** — `postgres.py:275`, default `10` (set per-query via `SET LOCAL`)

Number of cells inspected during a query. More probes → higher recall, higher latency.

| probes | Recall | Latency |
|---|---|---|
| 1 | ~70–80% | Fastest |
| 10 (current) | ~95% | Good balance |
| lists (= full scan) | 100% | Same as no index |

**Rule:** set `probes` to ~1% of `lists` for fast approximate search, up to 10% for near-exact results.

---

**IVFFlat → HNSW migration**

For corpora above ~500K chunks or when P99 latency matters more than build time:

```sql
DROP INDEX chunks_embedding_idx;
CREATE INDEX chunks_embedding_hnsw_idx ON chunks
  USING hnsw (embedding vector_cosine_ops)
  WITH (m=16, ef_construction=64);
-- At query time, set ef_search for recall/speed trade-off:
SET hnsw.ef_search = 40;
```

HNSW is faster at query time and does not require `lists` tuning, but uses 2–3× more memory and takes longer to build.

---

### 4. Hybrid Search & RRF

**`default_text_weight`** — `settings.py`, default `0.3`

Controls the relative weight of text search vs semantic search in the RRF merge. Not directly a weight in the RRF formula — used to tune the number of results fetched from each leg before merging.

**`rrf_k`** — `postgres.py:421`, default `60`

The smoothing constant in the RRF formula `1 / (k + rank)`. Higher k → less penalty for lower-ranked results, flatter score distribution.

| k | Effect |
|---|---|
| 1 | Aggressive — rank 1 scores much higher than rank 2 |
| 60 (current) | Standard — smooth score distribution, widely used default |
| 120 | Very flat — rank 1 and rank 10 are nearly equivalent |

**Leave at 60** unless you have evidence from your evaluation harness that a different value improves NDCG.

---

**`default_match_count`** — `settings.py`, default `10`
**`max_match_count`** — `settings.py`, default `50`

How many chunks are returned from retrieval and stuffed into the LLM context. More chunks → higher recall but more tokens → higher cost, longer latency, and lost-in-the-middle risk.

| match_count | Use case |
|---|---|
| 3–5 | High-precision queries, simple factual Q&A |
| 5–10 (current) | General purpose |
| 10–20 | Complex multi-part questions, research synthesis |
| > 20 | Only with a reranker — without one, precision collapses |

---

### 5. HyDE

**`hyde_enabled`** — `settings.py`, default `False`

Generates a hypothetical answer to the query using the LLM, then embeds the answer (instead of the query) for retrieval. Closes the vocabulary gap between query phrasing and document phrasing.

| Scenario | Use HyDE? |
|---|---|
| Queries are short, vague, or conversational | Yes — a good hypothetical answer is much richer than 3 words |
| Queries are precise technical terms | No — the query embedding is already close to documents |
| Latency budget < 200ms | No — HyDE adds a full LLM call (500–2000ms) |
| LLM is weak/small | No — a bad hypothetical answer hurts retrieval |

---

### 6. Reranker

**`reranker_enabled`** — `settings.py`, default `False`
**`reranker_type`** — `settings.py`, default `"llm"`
**`reranker_overfetch_factor`** — `settings.py`, default `3`

When enabled, retrieves `match_count × reranker_overfetch_factor` candidates, reranks them, and returns the top `match_count`.

**`reranker_type` options and trade-offs:**

| Type | Model | Latency | Quality | Cost |
|---|---|---|---|---|
| `llm` (current) | `llama3.1:8b` | 500–2000ms | High | LLM tokens per chunk |
| `cross_encoder` | `BAAI/bge-reranker-base` | 50–150ms | High | Local inference only |
| `cross_encoder` | `BAAI/bge-reranker-large` | 100–250ms | Higher | Larger model |

**`reranker_overfetch_factor`:** fetch 3× the requested results, rerank, return top N. Higher factor → better reranking coverage but more DB and embedding work.

| factor | Effect |
|---|---|
| 2 | Minimal — reranker sees twice as many candidates |
| 3 (current) | Good balance |
| 5+ | Useful only with a fast cross-encoder; LLM reranker becomes too slow |

**When to enable:** corpus > 10,000 chunks, or when evaluation shows top-ranked result is often wrong. Start with `cross_encoder` + `BAAI/bge-reranker-base` for the best latency/quality trade-off.

---

### 7. Result Cache

**`max_size`** — `retriever.py:104`, default `100` (number of cached queries)
**`ttl_seconds`** — `retriever.py:104`, default `300` (5 minutes)

In-memory LRU cache keyed on `(query, match_count, search_type)`. Identical queries within the TTL return instantly without hitting PostgreSQL or the embedding API.

| ttl_seconds | Use case |
|---|---|
| 60 | Rapidly changing corpus — stale results are a concern |
| 300 (current) | General purpose |
| 3600 | Static corpus — maximise cache hits |
| 0 (disable) | Debugging, evaluation runs (you want fresh results) |

Cache is cleared automatically after ingestion via `_result_cache.clear()`.

---

### 8. Connection Pool

**`db_pool_min_size`** — `settings.py`, default `1`
**`db_pool_max_size`** — `settings.py`, default `10`
**`command_timeout`** — `postgres.py`, default `60` seconds

| Setting | Recommendation |
|---|---|
| `min_size` | Set to number of worker processes — keeps connections warm |
| `max_size` | Set to `(PostgreSQL max_connections - system connections) / number_of_app_instances` |
| `command_timeout` | Lower to 10–15s in production — fast-fail slow queries rather than holding connections |

---

### 9. LLM Model

**`llm_model`** — `settings.py`, default `"llama3.1:8b"`

The LLM used for answer generation (and for the LLM reranker and HyDE if enabled). Does not affect retrieval quality — only the quality and style of the final answer.

| Model | Context window | Quality | Latency | Cost |
|---|---|---|---|---|
| `llama3.1:8b` (current) | 128K | Good | Fast (local) | Free |
| `llama3.1:70b` | 128K | Very good | Slow (local, needs GPU) | Free |
| `gpt-4o-mini` | 128K | Good | Fast | Low ($) |
| `gpt-4o` | 128K | Excellent | Fast | Medium ($$$) |
| `claude-sonnet-4-6` | 200K | Excellent | Fast | Medium ($$$) |
| `gemini-1.5-pro` | 1M | Excellent | Moderate | Medium ($$$) |

For RAG specifically, a smaller model is often sufficient — the LLM is summarising retrieved chunks, not recalling facts from memory. `gpt-4o-mini` or `llama3.1:8b` handles this well. Use a larger model when answers require complex reasoning across many chunks.

---

### 10. Complete Tuning Reference Table

| Tunable | Default | File | Impact area | When to change |
|---|---|---|---|---|
| `max_tokens` | 512 | settings / pipeline | Chunk quality | Increase if chunks are too small; never exceed embedding model limit |
| `chunk_size` | 1000 chars | pipeline | Fallback chunking | Increase for denser documents, decrease for FAQ-style content |
| `chunk_overlap` | 200 chars | pipeline | Fallback recall | 10–20% of chunk_size |
| `merge_peers` | True | docling.py | Chunk granularity | False for maximum granularity |
| `embedding_model` | nomic-embed-text | settings | Semantic quality | When NDCG@5 evaluation shows room for improvement |
| `embedding_dimension` | 768 | settings | Storage / speed | Must match model |
| `ivfflat.lists` | 100 | postgres.py | Index recall/speed | rows/1000 for < 1M chunks |
| `ivfflat.probes` | 10 | postgres.py | Query recall/speed | 1–10% of lists |
| `default_match_count` | 10 | settings | Recall vs cost | Lower for speed, higher for complex queries |
| `rrf_k` | 60 | postgres.py | RRF score distribution | Leave at 60 unless evaluation says otherwise |
| `default_text_weight` | 0.3 | settings | Text vs semantic balance | Increase for keyword-heavy queries |
| `hyde_enabled` | False | settings | Semantic recall | Enable for vague/conversational queries with latency budget |
| `reranker_enabled` | False | settings | Precision@K | Enable for large corpora where top-1 accuracy matters |
| `reranker_type` | llm | settings | Reranker speed/quality | cross_encoder for lower latency |
| `reranker_overfetch_factor` | 3 | settings | Reranker coverage | 3–5 depending on reranker speed |
| `cache max_size` | 100 | retriever.py | Cache hit rate | Increase for high-traffic, repetitive query workloads |
| `cache ttl_seconds` | 300 | retriever.py | Cache freshness | Lower for frequently updated corpora |
| `db_pool_max_size` | 10 | settings | Concurrency | Set based on PostgreSQL max_connections |
| `command_timeout` | 60s | postgres.py | Reliability | Lower to 10–15s in production |
| `llm_model` | llama3.1:8b | settings | Answer quality | Upgrade when answer synthesis is the bottleneck |
| `reranker_model` | bge-reranker-base | settings | Reranking precision | Upgrade to bge-reranker-large for better precision |

---

## REST API

<a id="q123"></a>
**Q123. What HTTP endpoints does the REST API expose?**

`rag/api/app.py` exposes four endpoints under a FastAPI app (OpenAPI docs auto-generated at `/docs`):

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Connectivity checks for DB, embedding API, and LLM API |
| `POST` | `/v1/chat` | Full (non-streaming) answer via `traced_agent_run` |
| `POST` | `/v1/chat/stream` | Streaming answer as Server-Sent Events (SSE) via `agent.run_stream()` |
| `POST` | `/v1/ingest` | Trigger `DocumentIngestionPipeline` for a folder on the server |

Run with: `uvicorn rag.api.app:app --host 0.0.0.0 --port 8000 --reload`

---

<a id="q124"></a>
**Q124. How does `POST /v1/chat` work under the hood?**

The endpoint accepts a `ChatRequest` (Pydantic model) with fields `query`, optional `user_id`, `session_id`, and `message_history`. It calls `traced_agent_run(query, user_id, session_id, message_history)` which:

1. Creates a Langfuse trace (if Langfuse is configured).
2. Instantiates `RAGState(user_id=user_id)` for lazy-initialised store/retriever/Mem0.
3. Calls `agent.run(query, deps=state)` — the Pydantic AI agent may invoke `search_knowledge_base` one or more times.
4. Updates the Langfuse trace with the final answer and flushes.
5. Closes the asyncpg pool acquired by `RAGState`.

The endpoint returns `ChatResponse(answer=result.output, session_id=session_id)`. On any exception it raises `HTTPException(500)`.

Because `traced_agent_run` creates and destroys its own `RAGState` (and therefore its own asyncpg pool) on every call, each HTTP request gets an isolated pool. For high-throughput use, share a single `RAGState` across requests via a FastAPI lifespan dependency (see Q129).

---

<a id="q125"></a>
**Q125. How does streaming work — what is the SSE format?**

`POST /v1/chat/stream` returns a `StreamingResponse` with `media_type="text/event-stream"`. It uses Pydantic AI's `agent.run_stream()` async context manager:

```python
async with agent.run_stream(query, deps=state) as streamed:
    async for delta in streamed.stream_text(delta=True):
        yield f"data: {json.dumps({'delta': delta})}\n\n"
yield f"data: {json.dumps({'done': True})}\n\n"
```

Each SSE frame is a JSON object on a `data:` line:

| Event | Payload | Meaning |
|---|---|---|
| Token delta | `{"delta": "Hello"}` | Next text fragment from the LLM |
| End of stream | `{"done": true}` | All tokens delivered |
| Error | `{"error": "..."}` | Exception during generation |

The double newline (`\n\n`) is required by the SSE spec to delimit events. Clients read the stream with `EventSource` (browser) or any SSE library. HTTP headers are flushed immediately on the first byte, so the client sees `200 OK` even if the LLM later errors — errors arrive as an `error` event rather than an HTTP status code.

---

<a id="q126"></a>
**Q126. What does `GET /health` check and what HTTP status does it return?**

Three independent async checks run concurrently (each with a 5-second timeout):

| Component | Check | How |
|---|---|---|
| DB | `PostgresHybridStore().initialize()` | Creates a real asyncpg pool and pgvector tables |
| Embedding API | `GET {embedding_base_url}/models` | HTTP probe; any non-5xx response counts as up |
| LLM API | `GET {llm_base_url}/models` | Same pattern — works for Ollama and OpenAI-compatible endpoints |

Return codes:

| Condition | `status` field | HTTP |
|---|---|---|
| All pass | `"ok"` | `200` |
| Some pass | `"degraded"` | `503` |
| All fail | `"unhealthy"` | `503` |

The `503` response body still contains the full `HealthResponse` JSON so load balancers and dashboards can see which component is down without parsing log lines.

---

<a id="q127"></a>
**Q127. How does `POST /v1/ingest` work and what are its limitations?**

The endpoint accepts an `IngestRequest` with `documents_folder`, `clean`, `chunk_size`, and `max_tokens`. It calls `create_pipeline(...)`, awaits `pipeline.initialize()`, then `pipeline.ingest_documents()`, and returns `IngestResponse` with per-document results, total document count, and total chunks created. `pipeline.close()` is called in a `finally` block regardless of success or failure.

**Limitations of the current implementation:**

| Limitation | Risk | Fix (FAQ §4) |
|---|---|---|
| Runs inside the HTTP request | Long ingest blocks the Uvicorn worker | Offload to Celery/arq background job |
| No upload support | Server must already have the files | Add `POST /v1/ingest/upload` accepting multipart files |
| No progress visibility | Caller waits blind | `ingestion_jobs` table + `GET /v1/ingest/{job_id}` polling |
| `clean=True` deletes all data | Downtime during re-index | Shadow table swap (Q106) |
| No file size guard | Huge PDFs can OOM the worker | Reject files above configurable limit before conversion |

---

<a id="q128"></a>
**Q128. Why SSE over WebSockets for streaming?**

SSE (Server-Sent Events) is a unidirectional HTTP/1.1 stream — the server pushes; the client listens. WebSockets are bidirectional. For LLM token streaming, the communication pattern is one-way (server → client), so SSE is the simpler choice:

| Factor | SSE | WebSocket |
|---|---|---|
| Protocol | Plain HTTP — passes through proxies, CDNs, load balancers | Custom upgrade; may need extra proxy config |
| Reconnect | Built-in browser auto-reconnect with `Last-Event-ID` | Manual |
| Multiplexing | One stream per HTTP connection | Full duplex over one connection |
| Overhead | Zero extra handshake | Upgrade handshake |
| Use case fit | Token streaming (server→client) | Chat with client-initiated messages |

The only time WebSockets beat SSE here is if you want the client to send mid-stream messages (e.g., cancel a generation). HTTP/2 server push or a second HTTP request can handle that without WebSockets.

---

<a id="q129"></a>
**Q129. How is the asyncpg pool lifecycle managed across HTTP requests?**

Currently each endpoint call that needs the DB creates and destroys its own `RAGState` (which creates its own asyncpg pool). This is correct for isolation but inefficient at scale — pool creation costs ~50–200ms per cold-start.

**Production pattern — FastAPI lifespan + dependency injection:**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag_state = RAGState()
    await app.state.rag_state.get_retriever()  # warm up pool once
    yield
    await app.state.rag_state.close()

app = FastAPI(lifespan=lifespan)

def get_rag_state(request: Request) -> RAGState:
    return request.app.state.rag_state

@app.post("/v1/chat")
async def chat(req: ChatRequest, state: RAGState = Depends(get_rag_state)):
    result = await agent.run(req.query, deps=state)
    ...
```

This shares one pool across all requests, respects `pool_max_size`, and shuts down cleanly on `SIGTERM`.

---

<a id="q130"></a>
**Q130. How would you add authentication to the REST API?**

The cleanest approach for an API-first service is Bearer token auth via a FastAPI dependency:

```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer()

async def require_auth(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> str:
    token = credentials.credentials
    user_id = verify_jwt(token)  # raises if invalid/expired
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user_id

@app.post("/v1/chat")
async def chat(req: ChatRequest, user_id: str = Depends(require_auth)):
    ...
```

Key decisions:

| Decision | Recommendation |
|---|---|
| Token format | JWT (RS256) — stateless, verifiable without DB lookup |
| Issuer | Auth0, Cognito, or self-hosted (Keycloak) for production |
| `user_id` extraction | Decode from JWT `sub` claim — never trust the request body |
| API keys | Hash with bcrypt and store in `api_keys` table; rate-limit per key |
| CORS | Set `allow_origins` to known frontend domains only (not `*`) |
| HTTPS | TLS at load balancer; redirect HTTP → HTTPS in Nginx/Caddy |

See FAQ §1 (Production Readiness — Auth) for the full auth roadmap including RBAC and secrets management.

---

<a id="q131"></a>
**Q131. How do I fire off a query over the REST API?**

**Start the server**
```bash
uvicorn rag.api.app:app --host 0.0.0.0 --port 8000 --reload
```

**Non-streaming query (curl)**
```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What does NeuralFlow AI do?"}'
```
```json
{"answer": "NeuralFlow AI is an...", "session_id": null}
```

**Streaming query — tokens arrive as SSE (curl)**
```bash
curl -X POST http://localhost:8000/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the PTO policy?"}' \
  --no-buffer
```
```
data: {"delta": "The"}
data: {"delta": " PTO policy"}
data: {"delta": " is 30 days..."}
data: {"done": true}
```

**With optional fields**
```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the learning budget?",
    "user_id": "alice",
    "session_id": "session-123"
  }'
```

**From Python (`httpx`)**
```python
import httpx, asyncio

async def ask(query: str) -> str:
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "http://localhost:8000/v1/chat",
            json={"query": query},
        )
        r.raise_for_status()
        return r.json()["answer"]

print(asyncio.run(ask("How many engineers work here?")))
```

**Interactive docs**

Open `http://localhost:8000/docs` — FastAPI generates a Swagger UI where you can try all endpoints in the browser without any client tooling.
