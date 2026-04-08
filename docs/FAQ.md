# Interview Answers — Pydantic AI RAG System

Code references: line numbers point to files under `rag/` in this repo.

---

## RAG Fundamentals

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

**Q2. What are the main failure modes of a naive RAG pipeline?**

- **Recall failure**: the relevant chunk is not retrieved at all — wrong embedding model, poor chunking, or the query phrasing differs too much from the document.
- **Precision failure**: the retrieved chunks are topically related but don't contain the answer — results look relevant but are useless.
- **Lost-in-the-middle**: when multiple chunks are stuffed into the LLM context, the model attends poorly to chunks in the middle of a long context window.
- **Chunk boundary mismatch**: a sentence is split across two chunks; neither chunk individually answers the question.
- **Stale index**: documents updated on disk but not re-ingested; the LLM answers from old data.

**Q3. What is the difference between standard RAG and agentic RAG?**

Standard RAG is a hardwired pipeline: embed query → retrieve → stuff context → generate. The retrieval always happens regardless of whether the question needs it. Agentic RAG gives the LLM retrieval as a *tool* it can choose to call, with control over the query string and number of results. This project uses agentic RAG: the Pydantic AI agent has a `search_knowledge_base` tool (`rag_agent.py`) and decides when to call it. The agent can also decline to retrieve if the question is trivially answerable. It is "lightweight agentic" — one retrieval tool, no multi-hop planning loops.

**Q4. How does chunking strategy affect retrieval quality?**

Smaller chunks → higher precision (each chunk is tightly scoped) but lower recall (context that spans multiple chunks is split). Larger chunks → more context per result but noisier embeddings (the embedding averages over more text, diluting the signal). For this project, `max_tokens=512` is the hard ceiling set by the embedding model's window. The HybridChunker respects structural boundaries (sections, paragraphs) rather than splitting at an arbitrary character count, which improves coherence without sacrificing precision.

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

**Q6. Why store the full document alongside chunks?**

The `documents` table holds the full text and metadata, while `chunks` holds the searchable fragments. This allows: (a) re-chunking without re-ingesting the source file (just re-process the stored content), (b) displaying the source document to users, (c) computing the content hash for incremental ingestion without re-reading the file, and (d) cascading deletes — `ON DELETE CASCADE` removes all chunks when the parent document is deleted.

---

## Hybrid Search & RRF

**Q7. Explain hybrid search. What problem does each leg solve?**

Semantic (vector) search embeds the query and finds chunks whose embeddings are close in vector space. It handles vocabulary mismatch — "compensation" matches "salary" — but struggles with exact terms, acronyms, and proper nouns. Text search (tsvector) matches exact lexemes and is ideal for keywords like "PTO", "llama3", "NeuralFlow" but fails when query and document use different words for the same concept. Hybrid search runs both in parallel (`asyncio.gather`) and merges the ranked lists with RRF, rewarding chunks that appear high in both lists.

**Q8. Walk through RRF. What is the formula and what does k=60 do?**

For each chunk, RRF assigns a score from each ranked list:

```
rrf_score(rank) = 1 / (k + rank)
```

The final score is the sum across all lists. k=60 is a smoothing constant that prevents a rank-1 result from dominating completely (1/61 ≈ 0.016 vs 1/1 = 1.0 without smoothing). It was empirically shown in the original RRF paper to work well across diverse datasets. The effect: a chunk ranked #1 in semantic and #5 in text gets a combined score of 1/61 + 1/65 ≈ 0.032, which beats a chunk ranked #1 in only one list.

**Q9. When does text search win over semantic search?**

When the query contains exact tokens that appear verbatim in the document. Examples from this corpus: "PTO" (an acronym that embeddings might not distinguish from "PT"), "NeuralFlow" (a proper noun), "llama3" (a model name). The test results confirm this: text search has lower overall Hit Rate@5 (0.40) but for queries like "What is DocFlow AI" it finds the audio transcription that mentions "DocFlow" by exact token match.

**Q10. When does semantic search win?**

When query and document use different vocabulary for the same concept. Example: querying "company culture and values" matches a document that uses "core principles" and "work environment" — no shared keywords, but the embedding vectors are close. Semantic Hit Rate@5 = 0.90 on this corpus vs text = 0.40, showing it is the stronger leg for conceptual queries.

**Q11. Semantic Hit Rate@5 = 0.90 vs hybrid = 0.80. How do you explain this?**

RRF merges the two ranked lists. If a chunk ranks #1 in semantic but is not in the text results at all (Hit Rate@5=0.80 vs 0.90 means 1 query that semantic hits but hybrid misses), the RRF score may push another chunk above it. Specifically, a chunk that ranks moderately in *both* lists gets a higher combined RRF score than a chunk that ranks #1 in only one list. The "company mission and vision" query misses in hybrid — `mission-and-goals.md` ranks high semantically but doesn't contain strong keywords, so the text leg contributes nothing and a different, keyword-rich chunk edges ahead after RRF. Fix: increase `match_count` (fetch more candidates before RRF) or tune the k constant.

**Q12. If you had to drop one leg, which would you keep?**

Semantic search. It handles the majority of query types (conceptual, paraphrased, vocabulary-mismatch). Text search is critical for exact terms and acronyms, but those cases can be partially mitigated with a better embedding model. The converse is not true — you cannot fix vocabulary mismatch with keyword search.

---

## PostgreSQL / pgvector

**Q13. Why PostgreSQL over a dedicated vector DB?**

This system already needs PostgreSQL for relational data (documents, chunks, metadata, Mem0 memory). Adding a separate vector DB means two infrastructure components to manage, two connection pools, and a JOIN across network boundaries to correlate chunks with document metadata. PostgreSQL + pgvector handles both in a single query with a JOIN. The trade-off is that pgvector's IVFFlat index is less scalable than purpose-built ANN indexes (HNSW in Pinecone/Weaviate) at hundreds of millions of vectors, but for RAG workloads in the tens-of-thousands range it is entirely adequate.

**Q14. What is IVFFlat and how does it trade accuracy for speed?**

IVFFlat (Inverted File Flat) divides the vector space into `lists` Voronoi cells. At index time, each vector is assigned to its nearest centroid. At query time, only `probes` cells are searched rather than the full table. This reduces the search space from O(n) to O(n/lists × probes) but may miss true nearest neighbours that fall in unprobed cells (approximate, not exact). `lists = sqrt(n_rows)` is the standard recommendation. Increasing `probes` raises recall but also latency.

**Q15. What does `register_vector` do and why in `init=`?**

asyncpg doesn't know how to serialize/deserialize the `vector` type from pgvector by default. `register_vector` installs custom codecs for `vector` ↔ Python `list[float]`. It must run in the `init` callback because that callback fires once for *each new connection* the pool creates. If you call it once after pool creation, it only registers on the single connection you happen to have at that moment; subsequent connections created by the pool won't have the codec.

**Q16. Why `executemany` for batch inserts?**

`executemany` sends a single prepared statement to PostgreSQL and batches the parameter rows, which is significantly faster than N separate `INSERT` statements (N round-trips vs 1). For a document with 20 chunks, this reduces network overhead by 95%. The alternative `COPY` would be even faster for bulk loads but is more complex to use with asyncpg and embeddings.

**Q17. `ON DELETE CASCADE` — what does it do and why is it critical?**

When a row in `documents` is deleted, PostgreSQL automatically deletes all rows in `chunks` where `document_id` matches. Without it, deleting a document during re-ingestion would leave orphaned chunks in the DB — chunks with no parent document, wasting storage and polluting search results with unreachable content. The pipeline's `delete_document_and_chunks()` method relies on this: it only needs to delete the document row and the database handles chunk cleanup.

**Q18. Why UUID primary keys over auto-increment?**

Auto-increment integers are sequential and predictable (an attacker who gets chunk ID 100 knows IDs 1–99 exist). UUIDs are random, unpredictable, and globally unique — safe to expose in APIs. They also work correctly in distributed/multi-node settings where two nodes generating IDs simultaneously would collide with integer sequences. `gen_random_uuid()` runs inside PostgreSQL so no application-side UUID generation is needed.

**Q19. `GENERATED ALWAYS AS (...) STORED` — what does this mean?**

It is a PostgreSQL *generated column*. The value of `content_tsv` is automatically computed by PostgreSQL as `to_tsvector('english', content)` whenever a row is `INSERT`ed or `UPDATE`d. `STORED` means the computed value is written to disk (not recomputed at query time). You never write to this column manually — PostgreSQL enforces this (`GENERATED ALWAYS` prevents explicit writes). On `UPDATE` to `content`, the column is recalculated automatically.

---

## Full-Text Search

**Q20. What is a `tsvector` and how does it differ from the original text?**

A `tsvector` is not the original text — it is a sorted, de-duplicated list of *lexemes* with position tags. Three transformations happen: (1) stop words are removed ("the", "is", "a"), (2) remaining words are stemmed to their root form ("employees" → `employe`, "entitled" → `entitl`), (3) each lexeme is tagged with its position(s) in the original text (for phrase queries). Example: `to_tsvector('english', 'The employees are entitled to PTO')` → `'employe':2 'entitl':4 'pto':6`.

**Q21. What does stemming do and when can it cause false positives?**

Stemming reduces word variants to a common root so queries match all inflections. "run", "running", "runs" all become `run`. False positive example: "university" and "universe" both stem to `univers` in some stemmers, so a query for "universe" could match a document about a university. In this codebase, "PTO" and "PT" would both become `pt` — a query for "PT" (physical therapy) could match PTO documents.

**Q22. Why `plainto_tsquery` instead of `to_tsquery` for user input?**

`to_tsquery` requires the user to supply valid tsquery syntax (`'pto & policy'`). If a user types `'PTO policy?'` the `?` causes a parse error. `plainto_tsquery` takes raw prose, tokenizes it, and ANDs the non-stop-word lexemes. It never throws a syntax error on user input, making it safe for direct use without sanitisation.

**Q23. Why is a GIN index better than B-tree for tsvector?**

A B-tree index works on ordered scalar values (numbers, strings with natural ordering). A `tsvector` is a set of lexemes — there is no natural ordering of the whole vector. A GIN (Generalized Inverted Index) is an inverted index: for each lexeme, it stores the list of rows containing that lexeme. The `@@` operator can look up each lexeme in the tsquery directly in the index rather than scanning every row.

**Q24. "30 days PTO" — the number 30 is dropped. Why? How would you handle numeric search?**

Numbers are stop words under the `'english'` configuration. `to_tsvector('english', '30 days PTO')` → `'day':2 'pto':3`. If you need number search, use `'simple'` configuration (no stemming, no stop words) for a second tsvector column, or store structured numeric fields separately and search them with standard SQL comparisons.

**Q25. What happens when `plainto_tsquery` produces an empty query?**

If all query words are stop words (e.g. "what is the"), `plainto_tsquery` returns an empty `tsquery`. The `@@` operator against an empty tsquery returns `false` for every row, so the text search leg returns 0 results. In the hybrid search, this means only the semantic leg contributes. The codebase handles this gracefully because both searches run in parallel and the RRF merger works fine with one empty result list — it just returns the semantic results ordered by their semantic rank.

---

## Chunking & DoclingHybridChunker

**Q26. What does Docling's `HybridChunker` do that sliding-window cannot?**

Sliding-window splits at fixed character counts with no awareness of document structure. It can cut mid-sentence, mid-table, or mid-code-block. `HybridChunker` operates on the structured `DoclingDocument` produced by `DocumentConverter`, which knows about section headings, paragraph boundaries, table cells, lists, and code blocks. It splits at structural boundaries first (end of a section, paragraph break) and only uses token limits as a hard ceiling. The result is chunks that are semantically coherent units.

**Q27. What is `contextualize()` and why does it improve embedding quality?**

`contextualize(chunk)` prepends the heading hierarchy to the chunk's body text. For example, a chunk under "## Benefits > ### PTO" about time-off details becomes: `"Benefits > PTO\n\nEmployees receive 20 days of PTO per year..."`. Without this, the chunk reads "Employees receive 20 days per year..." with no indication of what "days" refers to. The embedding of the contextualized chunk is more specific and matches queries like "PTO policy" better because the topic is explicit in the text being embedded.

**Where is it called?** `rag/ingestion/chunkers/docling.py:191`, inside `DoclingHybridChunker.chunk_document()`. The loop iterates over structural chunks produced by Docling's `HybridChunker.chunk()`, then calls `self.chunker.contextualize(chunk=chunk)` on each — `self.chunker` is the Docling `HybridChunker` instance. The returned contextualized text is what gets embedded and stored, not the raw chunk text. Call chain: `pipeline.py` → `chunker.chunk_document()` → `docling.py:191` → Docling's `HybridChunker.contextualize()`.

**Q28. Describe the fallback chunking path exactly.**

Triggered when `docling_doc=None` (plain text, `.txt` files, or conversion failure). The `_simple_fallback_chunk` method (`chunkers/docling.py:228`) uses a sliding window: start at position 0, set `end = start + chunk_size`. It then walks backwards from `end` up to `max(start + min_chunk_size, end - 200)` looking for a sentence boundary (`.`, `!`, `?`, `\n`). If found, it cuts there; otherwise cuts at `end`. The next window starts at `end - overlap` (overlap = 100 chars by default). Token count is computed with the same HuggingFace tokenizer. The `chunk_method` metadata field is set to `"simple_fallback"` so you can distinguish these at query time.

**Q29. Why is `DocumentConverter` cached via `_get_converter()`?**

`DocumentConverter` loads several ML models on first instantiation — layout detection, table structure recognition, equation parsing — which takes several seconds and significant memory. Caching it means the cost is paid once per pipeline instance, not once per document. For a batch of 13 documents (this corpus), that's 12 avoided re-initializations. The cache is an instance variable (`_doc_converter`) so it's garbage collected when the pipeline is closed.

**Q30. What is `merge_peers=True`?**

When HybridChunker splits a document, it sometimes produces adjacent small chunks that are "peers" — they belong to the same structural level (e.g. consecutive short paragraphs under the same heading). `merge_peers=True` joins these small siblings into a single chunk if the combined token count stays under `max_tokens`. This reduces the number of very short chunks (which have poor embedding signal) and ensures each chunk has sufficient context to be meaningful.

**Q31. Why cache `DocumentConverter` as an instance attribute, not a module-level singleton?**

A module-level singleton would be shared across all `DocumentIngestionPipeline` instances (e.g. in tests). Different pipelines might be configured differently. More importantly, during tests each test creates and tears down its own pipeline, and a singleton would leak state across tests. Instance-level caching gives lifetime tied to the pipeline object, which is correct.

---

## Embeddings

**Q32. What does `nomic-embed-text` produce and why 768 dimensions?**

`nomic-embed-text` is a general-purpose text embedding model optimized for retrieval, producing 768-dimensional dense vectors. 768 is a common embedding size (BERT-base is also 768). Higher dimensions capture more nuance but increase storage (768 × 4 bytes = 3KB per chunk) and slow down vector similarity computation. For this corpus size the trade-off is fine.

**Q33. Cosine similarity vs Euclidean distance — why cosine?**

Cosine similarity measures the angle between vectors, ignoring magnitude. Two texts with the same meaning but different lengths produce vectors pointing in the same direction but at different magnitudes (longer text → larger magnitude). Cosine similarity normalises this away. Euclidean distance treats magnitude differences as semantic differences, which is wrong for text embeddings. pgvector uses `<=>` for cosine distance (`1 - cosine_similarity`).

**Q34. The embedder has an in-memory cache — what is the cache keyed on and what are its limits?**

The cache key is the query string (exact text match). This is appropriate for the retriever's query embedding (the same user question typed twice). Limits: (1) it is in-process memory — lost on restart; (2) no eviction policy visible in the code, so it grows unbounded; (3) it only helps for repeated identical queries, not paraphrased queries. In a long-running service, this could cause a memory leak for a large query vocabulary.

**Q35. Switching from nomic-embed-text (768-dim) to text-embedding-3-small (1536-dim) — what changes?**

- `EMBEDDING_DIMENSION=1536` in `.env`
- Drop and recreate the `chunks` table (column type changes: `vector(768)` → `vector(1536)`)
- Recreate the IVFFlat index with the new dimension
- Re-ingest all documents (old embeddings are incompatible)
- Update `EMBEDDING_MODEL` and `EMBEDDING_BASE_URL` / `EMBEDDING_PROVIDER`
- The `register_vector` call handles any dimension, so no code change there

**Q36. Symmetric vs asymmetric embedding models — which for RAG?**

Symmetric models produce embeddings where query and document live in the same space — comparing a short query to a short sentence. Asymmetric models (like `nomic-embed-text` with `search_query:` / `search_document:` prefixes, or `e5-` models) are trained on (query, passage) pairs where queries are short and documents are long. Asymmetric is more appropriate for RAG because queries and chunks are structurally different — you want the model to understand "query intent" vs "document content". `nomic-embed-text` supports this via instruction prefixes.

---

## HyDE

**Q37. Explain HyDE. Why might it outperform raw query embedding?**

HyDE (Hypothetical Document Embeddings, Gao et al. 2022): instead of embedding the raw query ("What is the PTO policy?"), the LLM generates a *hypothetical answer* ("NeuralFlow AI provides 20 days of PTO per year with a 15-day minimum..."), and *that text* is embedded. The intuition: the hypothetical answer is structurally similar to the actual document chunk — same vocabulary, same style. The embedding of the hypothetical answer therefore sits closer in vector space to real chunks than the embedding of a question. It effectively bridges the query-document vocabulary gap.

**Q38. What are the risks of HyDE?**

- **Hallucination propagation**: if the LLM generates a plausible-but-wrong hypothetical ("30 days PTO"), the embedding drifts toward chunks about vacation rather than the specific policy document.
- **Added latency**: one extra LLM call before retrieval.
- **Cost**: one LLM API call per query.
- **Worse for factual queries**: when the LLM has no relevant prior knowledge, the hypothetical can be completely off.

**Q39. When would you enable HyDE?**

When queries are highly conceptual or domain-specific and the vocabulary gap between queries and documents is large. Good candidates: legal documents (users ask in plain English, documents use legal terminology), medical records, technical patents. Not worth enabling for this corpus where queries are already close to the document language.

**Q40. How is HyDE implemented in the retriever?**

In `retriever.py`, if `settings.hyde_enabled` is True, before the search step the retriever calls `_get_hyde()` (lazy init) to get a `HyDEGenerator` instance. It calls `hyde.generate(query)` which makes an LLM API call to get a hypothetical document, then embeds that hypothetical text instead of the raw query. The rest of the pipeline (search, rerank, cache) is unchanged.

---

## Reranking

**Q41. What problem does a cross-encoder solve that bi-encoder retrieval cannot?**

Bi-encoders embed query and document independently — they cannot compare them token-by-token. A cross-encoder takes the (query, document) pair concatenated as input and scores relevance jointly with full cross-attention. This is much more accurate because the model sees both texts simultaneously and can model fine-grained interactions ("the document mentions '20 days' in the context of vacation, which matches the query about PTO"). The cost is O(n) cross-encoder calls where n = candidate count.

**Q42. LLM reranker vs CrossEncoder — trade-offs?**

| | LLM reranker | CrossEncoder |
|---|---|---|
| Model | Remote LLM API (same as generation) | Local `sentence-transformers` model |
| Quality | High (powerful model) | High (specialized for reranking) |
| Latency | ~1s per chunk via API | ~100ms per chunk, local |
| Cost | API cost per chunk scored | Hardware cost only |
| Privacy | Chunks sent to API | Data stays local |

LLM reranker is the default because it requires no additional model deployment. CrossEncoder is better for latency-sensitive or privacy-sensitive deployments.

**Q43. Why `asyncio.gather` for LLM reranker scoring?**

Each chunk scoring is an independent LLM API call. `asyncio.gather` fires all calls concurrently, so n chunks take approximately the time of one call (network-bound), not n × one call (sequential). Without it, reranking 10 candidates at 500ms each = 5 seconds; with `asyncio.gather` ≈ 500ms.

**Q44. At what corpus size or query volume would you enable the reranker?**

Enable when: (a) the corpus is large enough that top-K retrieval frequently returns marginally relevant results (usually >50K chunks), or (b) precision matters more than latency (e.g. agent needs exactly the right chunk to answer a specific factual question). For this 13-document, ~150-chunk corpus the retrieval precision is already high and reranking adds latency for marginal gain.

**Q45. Retrieval recall vs reranking precision — how do they compose?**

First stage (retrieval): maximize recall — use `match_count * reranker_overfetch_factor` to fetch more candidates than needed. If recall is 80% at K=5 but 95% at K=20, over-fetch to K=20. Second stage (reranker): maximize precision — take the top-K of the 20 reranked candidates. The two-stage pipeline lets you optimize each stage independently. If the first stage misses the relevant chunk (recall failure), no reranker can recover it — this is why recall in retrieval is the first thing to fix.

---

## Agentic RAG & Pydantic AI

**Q46. What makes this system "agentic"?**

The LLM (via Pydantic AI) decides *whether* to call retrieval, *what query* to use, and *how many results* to fetch. In a standard RAG pipeline these decisions are hardwired. The agent also has access to a separate `Mem0` memory store and can combine retrieved chunks with user history. It is lightweight agentic — one retrieval tool, one memory tool, no multi-step planning — but the control flow is driven by the model.

**Q47. How does Pydantic AI's tool system work?**

The `@agent.tool` decorator registers a Python async function as a tool available to the model. Pydantic AI serializes the function signature (name, parameters, docstring) into the tool schema and includes it in the system prompt / tool list sent to the LLM. When the LLM outputs a tool call (in its structured response), Pydantic AI deserializes the arguments, calls the Python function, and feeds the return value back to the LLM as a tool result. Type annotations are used for the schema — changing a parameter type changes what the LLM knows about the tool.

**Q48. What is `RAGState` and why are its attributes `PrivateAttr`?**

`RAGState` is the dependency injection container passed as `deps` to every tool call. It holds the `user_id`, `store`, `retriever`, and `mem0_store`. These are declared as `PrivateAttr(...)` because `RAGState` extends `BaseModel` — regular fields would be included in Pydantic's schema/validation/serialization, which is wrong for internal service objects. `PrivateAttr` tells Pydantic "this field exists but is not part of the data model."

**Q49. Why `ContextVar` for Langfuse trace context?**

In async Python, multiple coroutines run concurrently on the same thread. A class-level attribute like `_current_trace = None` is shared across all concurrent requests — request A's trace would overwrite request B's. `ContextVar` is Python's mechanism for per-coroutine (async task) local storage. Each concurrent `traced_agent_run` invocation gets its own trace reference that is invisible to all other concurrent invocations.

**Q50. Why is per-user state important in a multi-user chat app?**

`RAGState(user_id=user_id)` is created once per conversation turn with the specific user's ID. This ID is used to look up Mem0 memories for that user only (`mem0_store.get_context_string(user_id)`). Without per-user state, all users would see the same memory context — a major privacy and correctness failure.

**Q51. How does the agent handle tool call failures?**

The `search_knowledge_base` tool has a `try/except` that returns a formatted error string ("Error searching knowledge base: ...") rather than raising an exception. The LLM receives this error string as the tool result and is expected to gracefully inform the user that retrieval failed. The agent itself won't crash — Pydantic AI propagates the tool result back to the model regardless of whether it indicates success or failure.

---

## Memory (Mem0)

**Q52. What problem does Mem0 solve that conversation history cannot?**

Conversation history is ephemeral — it's the message list for the current session. Mem0 persists semantic facts across sessions: "The user prefers detailed explanations", "User is in the engineering team", "User asked about PTO last week." When a user starts a new conversation, Mem0 provides relevant context from past interactions so the agent doesn't start from zero.

**Q53. How is Mem0 stored in this project?**

Mem0 uses the same PostgreSQL database configured via `DATABASE_URL`. It creates its own tables (managed by the mem0 library, separate from `documents` and `chunks`). Memories are stored as text with embeddings, supporting vector similarity search to retrieve the most relevant past memories for a given query.

**Q54. `add()` vs `get_context_string()` — difference?**

`add(user_id, messages)` takes the current conversation messages, extracts salient facts (via the LLM), and stores them in PostgreSQL for that user. `get_context_string(user_id)` retrieves the most relevant stored memories for that user and formats them as a single string ready to be injected into the system prompt. The agent calls `get_context_string` at the start of each turn and `add` at the end.

**Q55. Why is Mem0 disabled by default?**

Mem0 requires an extra LLM call to extract memories from each conversation, adding latency and cost. It also requires the `mem0ai` package which has its own dependencies. For simple single-turn queries (most RAG use cases), it provides no benefit. It is valuable for multi-session, personalized assistants.

**Q56. How would you prevent Mem0 from storing sensitive information?**

Options: (a) post-process extracted memories through a PII detection model before storage, (b) configure the memory extraction prompt to explicitly exclude personal details ("do not store names, ages, financial information"), (c) add a content filter in the `add()` wrapper that scans for SSN/credit card patterns before calling the underlying mem0 library.

---

## Async Python & Performance

**Q57. Why must all I/O be async? What happens with a blocking call?**

Python's asyncio event loop is single-threaded. A blocking call (e.g. `time.sleep(1)`, `requests.get(url)`) blocks the *entire thread*, meaning no other coroutine can run while it's blocked. With `await asyncio.sleep(1)`, the event loop switches to other coroutines during the wait. A blocking DB call or HTTP call in an otherwise async service would serialize all requests, eliminating the concurrency benefit.

**Q58. What is an asyncpg connection pool and why use it?**

A pool maintains a set of pre-established PostgreSQL connections ready to be borrowed. Creating a new TCP connection + TLS handshake + PostgreSQL authentication takes 50–200ms. With a pool, a request borrows an existing connection (~0ms), runs the query, and returns it. `asyncpg.create_pool(min_size=1, max_size=10)` keeps 1–10 connections alive, allowing up to 10 concurrent queries without queuing.

**Q59. Maximum latency improvement from `asyncio.gather` on semantic + text search?**

If semantic search takes T_s and text search takes T_t, sequential execution takes T_s + T_t. `asyncio.gather` runs them concurrently, so total time ≈ max(T_s, T_t). Maximum improvement ≈ 50% when both take equal time. In practice, semantic search (vector cosine computation) is slower than text search (GIN index lookup), so the improvement is typically 30–40% — close to the dominant leg's latency.

**Q60. Why `init=register_vector` rather than registering after pool creation?**

See Q15. When `init=register_vector` is passed to `asyncpg.create_pool`, asyncpg calls it with each newly created connection before adding it to the pool. If you instead call `await conn.fetch(...)` to register after the pool exists, you only register on the one connection in your hand. The pool creates additional connections lazily as load increases — those connections would not have the codec. The `init` callback guarantees every pooled connection is properly configured.

**Q61. If `asyncio.gather` has two coroutines and one raises an exception — what happens?**

By default, `asyncio.gather` re-raises the first exception and cancels the other tasks (in Python 3.11+ with `return_exceptions=False`). In `postgres.py`, each search is wrapped in its own `try/except` that catches errors and logs them, returning an empty list. So both searches always return a list (possibly empty) and `gather` always completes. The RRF merger then works correctly on two lists, one possibly empty.

---

## Evaluation & Retrieval Metrics

**Q62. Hit Rate@K vs Precision@K — when do they diverge?**

Hit Rate@K is binary: 1.0 if *any* relevant doc is in top-K, 0.0 if none. Precision@K is the fraction of returned results that are relevant. They diverge when: a query has multiple relevant documents and some are retrieved. Example — 1 relevant result out of 5: Hit Rate@5 = 1.0, Precision@5 = 0.2. You care about Precision when you're stuffing all K results into the LLM context (you don't want 4 out of 5 to be noise). You care about Hit Rate when you're reranking — as long as the relevant doc is in the candidate set, the reranker can surface it.

**Q63. What does MRR measure that Hit Rate doesn't?**

MRR = mean of `1 / rank_of_first_relevant_result`. Example: Hit Rate@5 = 1.0 for both queries, but if query A's first relevant doc is rank 1 (MRR contribution = 1.0) and query B's is rank 5 (MRR contribution = 0.2), the average MRR = 0.6. Hit Rate would show both as 1.0 — misleading. MRR is better when you only show the user the top result, or when the LLM is most influenced by the first chunk in context.

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

**Q65. Is 10 queries a sufficient gold dataset?**

No — 10 queries gives high variance estimates. Changing one query outcome flips metrics by 10%. A production evaluation dataset should have 100–500 queries. To build it: (1) sample real user queries from logs, (2) manually annotate relevant documents for each, (3) use LLM-as-judge to scale annotation. The current gold dataset is appropriate for CI regression testing (did a code change break retrieval?) but not for publication-quality evaluation.

**Q66. Why do "company mission and vision" and "DocFlow AI" miss consistently?**

"Company mission and vision" — `mission-and-goals.md` is a small document with relatively generic language. After RRF, company-overview chunks (which appear in both semantic and text lists) may outscore mission-and-goals chunks (which only appear in the semantic list). Fix: investigate which document actually contains the answer and tune the query, or expand the relevant_sources list.

"DocFlow AI" — this content is in `Recording2.mp3`. If Whisper is not installed, the audio was never transcribed. In the DB there is either a `[Error: Could not transcribe...]` stub chunk or nothing. Fix: install Whisper and re-ingest.

**Q67. Recall@5 shows values above 1.0 — is that a bug?**

Not a bug in the metric code, but a bug in the gold dataset definition. Recall@K = `relevant_found / total_relevant`. `total_relevant` is set to `len(entry["relevant_sources"])` — the number of *documents* in the relevant_sources list, not the number of *chunks* retrieved. When a relevant document has multiple chunks in top-K (e.g. 3 chunks from `team-handbook`), `relevant_found` = 3 but `total_relevant` = 1, giving Recall = 3.0. The fix is to count distinct relevant *documents* found in top-K rather than chunks. This is a known ambiguity in chunk-level vs document-level recall.

**Q68. Why are unit tests and integration tests in the same file?**

The metric functions (`hit_rate`, `ndcg_at_k`, etc.) are directly imported by the integration tests. Keeping them co-located avoids a split where you'd need to import from a utility module. The `TestMetricFunctions` class has no async fixtures and runs in milliseconds — it acts as a correctness gate for the math before the expensive DB tests run. Separating them would add a module boundary with no organisational benefit.

**Q69. How would you use these metrics to decide whether to enable HyDE or the reranker?**

Run the gold dataset with each configuration: baseline (off/off), HyDE only, reranker only, both. Compare Hit Rate@5, MRR@5, NDCG@5, and mean latency. Enable the component if: (a) the metric improvement exceeds a threshold (e.g. +0.05 on MRR), and (b) the latency increase is acceptable for the use case. If HyDE helps MRR but adds 800ms latency for a chatbot, skip it. If the reranker helps NDCG@5 by 0.1 (better ranking quality), enable it.

---

## Observability & Langfuse

**Q70. What does Langfuse trace in this project?**

Each `traced_agent_run` call creates a Langfuse trace covering the full agent turn. Within it: the initial user message, the `search_knowledge_base` tool call (inputs + output), the Mem0 memory lookup, and the final LLM generation. This gives a per-turn view of what was retrieved, what context was provided, and what the model generated — essential for debugging wrong answers.

**Q71. Why `ContextVar` rather than function arguments?**

Passing a trace object through every function argument would require touching every function signature in the call chain (agent → tool → retriever → store). `ContextVar` stores the trace implicitly, available to any coroutine in the same async task without parameter threading. This is the standard Python pattern for request-scoped context (similar to `flask.g` in sync Flask).

**Q72. Using Langfuse traces to debug a wrong answer.**

1. Find the trace for the query.
2. Check the `search_knowledge_base` tool call — what query was sent to the retriever?
3. Check the retrieved chunks — are the relevant documents present? Are they ranked first?
4. Check the Mem0 context — is any stale/wrong memory being injected?
5. Check the final generation — is the LLM ignoring the correct chunk? (Lost in the middle?)
This narrows the bug to one of: retrieval failure, ranking failure, memory contamination, or LLM generation failure.

**Q73. Trace vs span vs generation in Langfuse?**

A **trace** is the top-level unit — one user request end-to-end. A **span** is a named sub-operation within a trace (e.g. "retrieve", "rerank") with start/end times. A **generation** is a special span that captures an LLM API call — it records the prompt, completion, model name, token usage, and cost. Langfuse aggregates generations for cost tracking.

---

## System Design

**Q74. Two-table schema (documents + chunks) — why not one table?**

Storing full document content in every chunk row would be massive redundancy (a 10-page PDF split into 20 chunks → the full PDF stored 20 times). The two-table design stores the full document once in `documents` and references it from `chunks` via FK. It also enables document-level operations (update, delete, list) without touching chunks, and supports the `ON DELETE CASCADE` pattern for clean teardown.

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

**Q76. Scale to 10M documents — what breaks first?**

1. **IVFFlat vector index** — at 10M rows with 768-dim vectors, IVFFlat recall degrades unless `lists` is tuned to ~3162 (sqrt of 10M) and `probes` increased. Switch to HNSW (supported in pgvector ≥0.5) which maintains recall at scale.
2. **Ingestion throughput** — `DocumentConverter` is single-threaded and CPU-bound. A single process cannot ingest fast enough. Need a message queue (Kafka/SQS) + worker pool.
3. **PostgreSQL write throughput** — 10M documents × ~20 chunks × 3KB embeddings ≈ 600GB. Need table partitioning, read replicas, and potentially a separate vector store.

**Q77. Implementing true incremental ingestion with deduplication.**

The pipeline already does this (`clean_before_ingest=False`): compute MD5 hash of the file, compare against `metadata.content_hash` stored in the `documents` table. If equal → skip. If different → `delete_document_and_chunks(source)` then re-ingest. If the source doesn't exist → ingest as new. Deleted files are handled by comparing `current_sources` (files on disk) against `get_all_document_sources()` (files in DB) and deleting any that are in DB but not on disk.

**Q78. Multi-tenancy with isolated document stores.**

Options:
- **Row-level security (RLS)**: add `tenant_id` column to `documents` and `chunks`, enable PostgreSQL RLS policies so each connection only sees its own tenant's rows. Single schema, strong isolation via policy enforcement.
- **Schema-per-tenant**: each tenant gets a separate PostgreSQL schema (`tenant_a.chunks`, `tenant_b.chunks`). More isolation, more schema management overhead.
- **Database-per-tenant**: separate PostgreSQL databases or Neon branches. Maximum isolation, highest cost.
For this codebase, RLS would require adding `tenant_id` to all queries and the pool connection setup.

**Q79. Risk of changing the embedding model after ingestion.**

All existing chunk embeddings are in the old model's vector space. New query embeddings are in the new model's space. Vector similarity between old and new spaces is meaningless — cosine similarity of incomparable vectors would return arbitrary scores. Result: total retrieval failure. Fix: re-ingest all documents with the new model before switching query embedding. Zero-downtime approach: dual-write to a new index during migration, switch queries over once the new index is complete.

**Q80. Sub-100ms latency — what to sacrifice first?**

Drop HyDE first (saves one LLM call, ~500ms). Then disable reranking (saves n API calls). Then consider switching from hybrid to semantic-only (saves the text search + RRF merge, ~50ms). Finally, switch from remote Ollama embedding to a locally loaded model with caching. The embedding call is the dominant latency outside the DB query itself.

---

## Code Quality

**Q81. Why `pydantic-settings` instead of `os.environ`?**

`pydantic-settings` provides: (1) type validation — `EMBEDDING_DIMENSION=abc` raises a `ValidationError` immediately rather than failing at runtime with a cryptic type error; (2) automatic `.env` file loading; (3) default values with documentation in the model definition; (4) credential masking in `__repr__` (API keys shown as `***`). Raw `os.environ` gives you a dict of strings with no validation, defaults, or type coercion.

**Q82. What does `ruff` check for vs `flake8 + black`?**

`ruff` is a Rust-based linter that replaces both `flake8` (style + lint rules) and `black` (formatting) in a single tool. It is 10–100× faster than the Python equivalents and checks for: unused imports, undefined names, import ordering (isort), type annotation style, security issues (bandit-equivalent rules), and more. `ruff format` handles formatting (black-compatible). The key benefit over flake8 + black is a single configuration file and a single command.

**Q83. Why Pydantic models for `ChunkData` and `SearchResult` instead of plain dataclasses?**

Pydantic provides runtime type validation — if a search result is returned with `similarity` as a string instead of a float, Pydantic raises a `ValidationError` immediately rather than a downstream `AttributeError`. Pydantic models also have automatic `__repr__`, JSON serialisation, and schema generation. For data flowing between system boundaries (DB → Python → LLM context), the validation guarantees are worth the overhead.

**Q84. Why `from collections.abc import Callable` rather than `callable`?**

`callable` is a built-in function, not a type. `Callable[[int], str]` is a type annotation saying "a function that takes an int and returns a str". In Python ≤3.8, `typing.Callable` was the way; in 3.9+, `collections.abc.Callable` is preferred (the `typing` versions are being deprecated). The CLAUDE.md convention exists because a previous bug was introduced by using lowercase `callable` as a type annotation — it evaluated to `True`/`False` rather than the type spec.

**Q85. How does `IngestionConfig` → `ChunkingConfig` separation keep concerns clean?**

`IngestionConfig` is the pipeline-level config — it owns parameters relevant to the pipeline as a whole (chunk_size, chunk_overlap, max_chunk_size, max_tokens). `ChunkingConfig` is the chunker's own config — it's what the `DoclingHybridChunker` constructor accepts. The pipeline translates one into the other. This means the chunker is usable independently of the pipeline (e.g. in tests, in the notebook) without constructing a full `IngestionConfig`. It also means the chunker's interface can evolve without changing the pipeline's public API.

---

## Ingestion Pipeline Deep Dive

**Q91. Walk through full ingestion step by step.**

See Q75 — detailed answer there. Summary path: `_find_document_files()` → `_compute_file_hash()` → `_read_document()` (Docling → markdown + DoclingDocument) → `_extract_title()` → `_extract_document_metadata()` → `chunker.chunk_document()` (HybridChunker → contextualize → ChunkData list) → `embedder.embed_chunks()` (POST /v1/embeddings) → `store.save_document()` (INSERT documents) → `store.add()` (executemany INSERT chunks) → `_result_cache.clear()`.

**Q92. How does `DocumentConverter` differ from PyPDF2 / pdfplumber?**

PyPDF2 and pdfplumber extract raw text streams from PDF content streams — they are layout-unaware. A two-column PDF produces interleaved text from both columns. Tables become unformatted text. Docling's `DocumentConverter` runs a full ML pipeline: (1) layout detection (identifies text blocks, tables, figures, headers/footers per page using a vision model), (2) reading order determination (correct multi-column flow), (3) table structure recognition (identifies rows/cols in table images), (4) formula detection. The output is a structured `DoclingDocument` with typed elements: `TextItem`, `TableItem`, `PictureItem`, `SectionHeaderItem` etc., preserving semantic structure.

**Q93. What internal representation does `DoclingDocument` provide and how does `HybridChunker` use it?**

`DoclingDocument` is a hierarchical document object with: a `body` containing a tree of typed items (`SectionHeaderItem`, `TextItem`, `TableItem`, `ListItem`, etc.), each tagged with its heading path (e.g. item is under "## Architecture > ### Storage"). `HybridChunker` traverses this tree, grouping items into chunks such that: (a) a `SectionHeaderItem` starts a new chunk boundary, (b) `TextItem`s within the same section are merged until `max_tokens` is exceeded, (c) a `TableItem` is kept as a single chunk (never split mid-table), (d) `merge_peers=True` merges adjacent small chunks at the same structural level.

**Q94. Explain `contextualize()` — what exactly gets prepended?**

`contextualize(chunk)` reads the `heading_path` attribute of the chunk (set by HybridChunker from the parent `SectionHeaderItem` ancestors) and prepends it as a breadcrumb: `"Level1 > Level2 > Level3\n\n"` followed by the chunk's raw text. For a chunk about PTO under `## Benefits > ### Time Off Policy`, the output is:

```
Benefits > Time Off Policy

Employees are entitled to 20 days of PTO per year...
```

The embedding of this contextualized text places it closer in vector space to queries about "benefits PTO policy" than the embedding of the raw text alone.

**Q95. What is `merge_peers=True` — give an example.**

Consider a document with three consecutive short paragraphs under "## Goals", each 50 tokens:
- Without `merge_peers`: three separate chunks of 50 tokens each — too short, poor embedding signal.
- With `merge_peers=True`: the three paragraphs are merged into one chunk of ~150 tokens, under the shared heading context. Better semantic coherence and one fewer DB row to search.

You'd turn it off if you need maximum granularity for a corpus with very long sections where merging pushes chunks over `max_tokens`.

**Q96. What happens to a table in a PDF during chunking?**

Docling's `DocumentConverter` identifies table regions and applies table structure recognition to parse rows and columns. The table becomes a `TableItem` in `DoclingDocument` with structured data. `HybridChunker` treats a `TableItem` as an atomic unit — it is never split across chunk boundaries. The table is serialized to a text representation (usually a markdown table or CSV-like format) and included as a single chunk. This preserves the relational structure of the table for embedding.

**Q97. Tokenizer mismatch: `all-MiniLM-L6-v2` for chunking, `nomic-embed-text` for embedding.**

`all-MiniLM-L6-v2`'s tokenizer is used by HybridChunker to count tokens and enforce the 512-token limit. `nomic-embed-text` uses a different tokenizer (based on GPT-style BPE). The two tokenizers have different vocabularies — a chunk that is 512 tokens by `all-MiniLM` may be 530 tokens by `nomic-embed-text`'s tokenizer, causing silent truncation when the embedding model processes it. Mitigation: use the embedding model's own tokenizer for chunk boundary decisions. In practice, the difference is small (~5%) and rarely causes significant truncation.

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

**Q99. Why cache `DocumentConverter`?**

Creating a `DocumentConverter` loads PyTorch ML models (layout detection ~200MB, table structure ~100MB) from disk into memory, initializes GPU/CPU compute contexts, and allocates memory. On CPU this takes 5–15 seconds. With caching (`_get_converter()` returns `self._doc_converter` if already set), this cost is paid once per pipeline instance. For a batch of 13 documents, that's 12 avoided re-loads, saving up to 3 minutes of startup time.

**Q100. MD5 for content hashing — how it works and limitations.**

`_compute_file_hash()` reads the file in 8192-byte blocks and feeds them to `hashlib.md5()`, returning the hex digest. Stored in `metadata.content_hash` in the `documents` table. Incremental ingestion compares this hash with the stored one: equal → skip, different → delete + re-ingest.

Limitations: (1) MD5 has known collision vulnerabilities — not cryptographically safe, but fine for file change detection (not security). (2) If a file's bytes change but its content is semantically unchanged (e.g. PDF metadata update, BOM encoding change), the hash changes and triggers unnecessary re-ingestion. (3) Conversely, a semantically meaningful change that happens to produce the same MD5 (collision) would be silently skipped — extremely unlikely in practice.

**Q101. Incremental ingestion — walk through all four cases.**

In `ingest_documents()` with `clean_before_ingest=False`:

- **New file**: `get_document_hash(source)` returns `None` (not in DB). Log `[NEW]`. Call `_ingest_single_document()`. Increment `new_count`.
- **Unchanged file**: `get_document_hash(source)` returns a hash that matches `_compute_file_hash()`. Log `[SKIP]`. Increment `skipped_count`. No processing.
- **Modified file**: hash mismatch. Log `[UPDATE]`. Call `delete_document_and_chunks(source)` (deletes document + cascades to chunks). Then call `_ingest_single_document()` to re-ingest. Increment `updated_count`.
- **Deleted file**: after processing all files on disk, call `get_all_document_sources()` from DB and compare against `current_sources` (files found on disk). Any source in DB but not on disk → `delete_document_and_chunks()`. Increment `deleted_count`.

**Q102. Why `_result_cache.clear()` after ingestion?**

The retriever has a module-level `_result_cache` (LRU + TTL) that stores query → results mappings. After ingestion, new chunks exist in the DB that were not there when the cache entries were computed. If a user queries "What is the PTO policy?" 1 minute before ingestion and again 1 minute after, the cache would return stale results (missing the newly ingested chunks). Clearing the cache forces re-queries against the updated DB immediately. The cache is module-level and shared across all `Retriever` instances, so a single `.clear()` suffices.

**Q103. YAML frontmatter — where stored, how used?**

`_extract_document_metadata()` checks if the content starts with `---` and tries to parse the YAML block between the first `---` and the next `\n---\n`. The parsed key-value pairs are merged into the `metadata` dict, which is stored in the `documents.metadata` JSONB column and also copied into each `ChunkData.metadata`. At query time, metadata is returned in `SearchResult` objects and can be used for filtering (e.g. `WHERE metadata->>'author' = 'Alice'`) or display. Currently it is stored but not used for search filtering — a future enhancement would be metadata-filtered retrieval.

**Q104. Top three bottlenecks at 10,000 docs/day and fixes.**

1. **`DocumentConverter` (CPU-bound, sequential)**: single ML inference pipeline processes ~1 doc/sec on CPU. At 10K/day that's ~2.8 hours. Fix: parallelize with a worker pool (`asyncio.to_thread` wrapping the sync converter call), multiple processes, or GPU-enabled instances.

2. **Embedding API calls (network-bound, sequential per document)**: `embed_chunks()` calls the embeddings API once per document's chunks. At 20 chunks/doc × 10K docs = 200K API calls. Fix: batch across documents (accumulate chunks from multiple documents and embed in large batches), use a local embedding model, or async-parallel embed across multiple documents.

3. **PostgreSQL write throughput**: `executemany` is fast per document but 10K documents × 20 chunks × 3KB vectors = 600MB of data per day. Fix: use PostgreSQL `COPY` protocol for bulk load, partition the `chunks` table by ingestion date, and tune `work_mem`/`checkpoint_segments` for write performance.

**Q105. Parallelizing ingestion while sharing `DocumentConverter` and the asyncpg pool.**

`DocumentConverter` is not thread-safe (PyTorch models share state). The pattern: run conversion in `asyncio.to_thread` — each conversion call gets its own thread where it calls a fresh (not cached) `DocumentConverter`. Alternatively, use a `multiprocessing.Pool` where each worker process has its own converter instance. The asyncpg pool is already thread-safe and handles concurrent connections. A semaphore limits concurrent conversions to avoid OOM:

```python
sem = asyncio.Semaphore(4)  # 4 concurrent conversions
async def ingest_with_limit(file):
    async with sem:
        return await asyncio.to_thread(convert_and_ingest, file)
await asyncio.gather(*[ingest_with_limit(f) for f in files])
```

**Q106. Zero-downtime re-index when `clean_before_ingest=True`.**

The current `clean_collections()` drops all data before re-ingesting — there is a window where the DB is empty and queries return nothing. Zero-downtime approach:
1. Create a new set of tables (`documents_v2`, `chunks_v2`).
2. Ingest all documents into the new tables.
3. Atomically swap table names (PostgreSQL `ALTER TABLE RENAME` is transactional).
4. Drop the old tables.

Or use PostgreSQL table inheritance / partitioning with a read view that spans both versions during migration. Neon's branching feature makes this even cleaner — ingest on a branch, then merge.

**Q107. Scanned PDFs with no text layer.**

`DocumentConverter` runs OCR (via Tesseract or a built-in OCR pipeline) when no text layer is detected. This is slower than digital PDF processing. If OCR is not configured or fails, `export_to_markdown()` returns an empty or near-empty string. The pipeline falls through to the raw UTF-8 read fallback, which for a scanned PDF with no text layer returns binary garbage or an empty file. Fix: detect empty conversion output before chunking (`if len(markdown_content.strip()) < 100: raise`) and log a clear error rather than creating empty chunks.

**Q108. Why return both markdown string and `DoclingDocument`?**

Re-parsing the markdown string would lose the original structure. `DoclingDocument` is Docling's in-memory structured representation — it has typed elements, heading trees, table data. If you serialise to markdown and re-parse, you get a flat text representation: headings become `# text`, tables become text grids, and the structural hierarchy is lost. `HybridChunker` needs the original `DoclingDocument` to use structural boundaries. The markdown string is stored in `documents.content` for human readability and full-text indexing; the `DoclingDocument` is used only during the chunking step.

**Q109. Audio files — how are they different from PDF chunks?**

Audio transcription goes through `AsrPipeline` (Whisper Turbo). The output `DoclingDocument` contains `TextItem`s with timestamps (`[time: 0.0-5.2]` markers embedded in the text) rather than heading-structured text. There is no heading hierarchy, so `contextualize()` has nothing to prepend — the heading path is empty. The result is that audio chunks behave like the simple fallback path for contextualization but use HybridChunker's token-aware splitting. The `[time: X.X-Y.Y]` markers in chunk text allow the retrieval system to surface the exact timestamp in the audio file where a topic was discussed.

**Q110. Impact of raw text fallback when PDF conversion fails.**

When Docling fails, `_read_document()` falls back to `open(file_path, encoding='utf-8').read()`. For PDFs this returns binary-encoded garbage (PDF syntax: `%PDF-1.4`, object streams, xref tables). This is passed to `_simple_fallback_chunk()` which creates chunks of garbage text. These chunks get embedded (producing meaningless vectors) and stored. At query time they score low on semantic search but may accidentally score on text search (e.g. if the PDF binary happens to contain the word "PTO" in a content stream). Better fallback: detect the file type before the UTF-8 read and return `("[Error: could not convert PDF]", None)` immediately, so the document is recorded in the DB with an error but no garbage chunks are created.

---

## Tricky / Deep-Dive

**Q86. RRF scores of 0.01–0.03 — why isn't this low confidence?**

RRF scores are not probabilities. They are sums of `1/(k+rank)` terms. With k=60, the maximum possible RRF score for a chunk that ranks #1 in both semantic and text is `1/61 + 1/61 ≈ 0.033`. A score of 0.016 (rank #1 in one list only) is high — it's half the maximum. The intuition: RRF scores are relative within a result set, not absolute confidence levels. Compare results by their RRF scores against each other, not against 1.0.

**Q87. After re-ingestion, previously passing tests now fail. Possible causes.**

1. **Chunk boundaries changed**: `DocumentConverter` is non-deterministic at the token boundary when `merge_peers` adjustments occur. A chunk that previously contained the key phrase may now be split differently.
2. **Different embedding values**: embedding models have non-deterministic temperature or the model was updated — same text produces slightly different vectors, shifting rankings.
3. **New documents added**: new chunks from additional documents may outscore the previously top-ranked chunk for some queries.
4. **Content hash collision**: a file was modified but `_compute_file_hash` incorrectly returned the old hash (race condition), so the document was not re-ingested with the new content.
5. **Result cache stale**: `_result_cache.clear()` was not called after ingestion in the test setup (check that `ingest_documents()` was awaited fully before running test queries).

**Q88. Query "PTO" — what happens in tsvector and why might it miss "paid time off"?**

`plainto_tsquery('english', 'PTO')` → `'pto'`. This lexeme is searched in `content_tsv`. A document chunk that contains "paid time off" but never mentions "PTO" → `to_tsvector('english', '... paid time off ...')` → `'paid':2 'time':3 'off':4`. The lexeme `'pto'` is absent. Text search returns 0 for this chunk. The semantic leg would still match (embedding of "PTO" is close to "paid time off" in vector space). This is exactly the use case for hybrid search — text search misses the vocabulary mismatch, semantic catches it.

**Q89. LLM reranker with partial failure (rate limiting).**

`asyncio.gather(*scoring_calls)` with default settings: if one call raises an exception, `gather` re-raises it and other tasks are not cancelled (Python 3.11 with `return_exceptions=False` cancels them; with `return_exceptions=True` returns exceptions as values). The safe approach used should be `return_exceptions=True`, then filter out exceptions from the results and assign a neutral score (0.0) to failed chunks. The reranker should then return the subset of successfully scored chunks rather than erroring entirely. If all calls fail, fall back to the original retrieval order.

**Q90. Changing `chunk_overlap` from 100 to 0 — improve some metrics, hurt others?**

With overlap=0: fewer total chunks (no duplicated content at boundaries), cleaner boundaries, no duplicate information in the index. Precision@K may improve (less redundant chunks in results). Recall@K may drop: a sentence that straddles a boundary is now fully in one chunk rather than partially in two — if it's in the "wrong" chunk, the query misses it. MRR could go either way. The improvement is most visible in small corpora where duplicate chunks from overlap pollute results. For large corpora, overlap is important to prevent boundary-straddling losses.
