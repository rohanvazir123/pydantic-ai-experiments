# RAG System Design — Top 20 Curated Topics

The 20 highest-signal topics from the full set of 55. Each one either reveals a design instinct, exposes a production failure mode, or forces a real architectural trade-off. Covers pipeline, chunking, retrieval, generation, evaluation, ingestion, and security.

Full detailed answers for each are in the individual section files linked below.

## Table of Contents

- [Selection Criteria](#selection-criteria)
- [The Top 20](#the-top-20)
- [Quick Study Map](#quick-study-map)

---

## Selection Criteria

A topic made this list if it satisfies at least two of:
- Tests a design decision with no obviously correct answer
- Exposes a failure mode easy to miss until it hits production
- Has meaningful token cost implications at scale
- Requires knowledge spanning multiple layers of the system

---

## The Top 20

---

### 1. Document ingestion pipeline for 500K documents with continuous updates

> How do you design the document ingestion pipeline for 500K documents with continuous updates?

**Why it's top 20:** Production RAG ingestion is an engineering problem, not a script problem. Tests whether the candidate has designed for failure recovery, incremental updates, and scale.

**What a strong answer covers:**
- Queue-based architecture: one message per document change event, horizontally scalable workers
- Idempotency: content hash as idempotency key — re-processing the same document is safe
- Batched embedding: 100–500 chunks per API call for throughput efficiency
- Dead letter queue for permanently failing documents
- Token cost table for initial ingestion at 500K documents ($1,550–$8,550)

**Full answer:** [08_knowledge_base_ingestion.md](08_knowledge_base_ingestion.md#q37)

---

### 2. Document updates without re-indexing the entire corpus

> How do you handle document updates and deletions without re-indexing the entire corpus?

**Why it's top 20:** Re-indexing 500K documents daily is expensive and slow. Chunk-level incremental updates are the production approach but have consistency edge cases that few candidates anticipate.

**What a strong answer covers:**
- Change detection: webhooks (S3 events, SharePoint Graph API) or content hash polling
- Chunk-level comparison: only chunks that actually changed are re-embedded (cost-efficient)
- Atomic swap: index new chunks before deleting old ones to maintain query continuity
- Soft delete before hard delete: immediate query filtering, delayed physical removal
- Coordinated multi-document updates: atomic activation group for related policy changes

**Full answer:** [08_knowledge_base_ingestion.md](08_knowledge_base_ingestion.md#q38)

---

### 3. RAG latency components — critical path and token cost trade-offs

> What are the latency components of a RAG pipeline and which are on the critical path?

**Why it's top 20:** Latency optimisation requires knowing which components can be parallelised and what the token cost implications of each optimisation are.

**What a strong answer covers:**
- Critical path: query embedding → retrieval → reranking → context assembly → LLM generation
- Parallelisable: dense + sparse retrieval, faithfulness check + citation verification (async after streaming)
- Token cost leverage: reducing context from 10,000 to 3,000 tokens saves ~70% of input token cost and reduces TTFT by 20–30ms
- Self-hosted embedding: eliminates API rate limits, reduces embedding latency from 50ms to 5–10ms

**Full answer:** [09_latency_performance.md](09_latency_performance.md#q42)

---

### 4. What can be cached and what invalidation is required

> What can be cached in a RAG system — embeddings, retrieved chunks, generated answers — and what are the invalidation conditions?

**Why it's top 20:** Caching is the highest-ROI latency and cost optimisation, but wrong invalidation causes silent wrong results. This tests both performance and correctness thinking.

**What a strong answer covers:**
- Query embedding cache: TTL indefinite (until model changes), 20–30% hit rate for recurring queries
- Retrieval result cache: invalidate when any document in the returned set is updated
- Generated answer cache: key must include document_version_hash + user permission scope — never serve Tenant A's cached answer to Tenant B
- Token cost savings: a cached answer saves all LLM generation tokens (~$0.03/query at $0.06/1K tokens)

**Full answer:** [09_latency_performance.md](09_latency_performance.md#q43)

---

### 5. End-to-end RAG pipeline architecture and failure detection

> Walk me through the end-to-end architecture of a production RAG system. Where are the failure modes at each stage and how do you detect them without a human in the loop?

**Why it's top 20:** Sets the baseline. A strong answer identifies six distinct stages, names a specific observable failure signal for each, and distinguishes "user sees an error" from "user sees wrong data" — the latter is more dangerous.

**What a strong answer covers:**
- Six stages: intent classification → query transformation → retrieval → context assembly → generation → post-processing
- Per-stage failure: retrieval miss (relevant chunk not retrieved), hallucination (claim not in context), citation fabrication
- Cross-cutting confidence score aggregating signals from all stages into a review queue
- Automated faithfulness check as the primary hallucination detection mechanism

**Full answer:** [01_pipeline_architecture.md](01_pipeline_architecture.md#q1)

---

### 6. RAG vs fine-tuning vs full context — when does each win?

> When do you use RAG versus fine-tuning versus stuffing the full corpus into the context window?

**Why it's top 20:** The most common architectural decision for knowledge-intensive LLM applications. Candidates who say "just use RAG" or "just fine-tune" without understanding the trade-offs are not ready for a production decision.

**What a strong answer covers:**
- Full context: works for small corpora (< 50 pages), fails on cost and attention quality at scale
- Fine-tuning: best for stable knowledge and style, fails for frequently-updated knowledge and attribution requirements
- RAG: best for large, changing corpora requiring attribution; fails for full-corpus reasoning and implicit world knowledge
- Hybrid (fine-tune + RAG): fine-tune for instruction following, RAG for knowledge — the production recommendation

**Full answer:** [01_pipeline_architecture.md](01_pipeline_architecture.md#q5)

---

### 7. Chunk size — failure modes of too small vs too large, and token cost

> How do you decide chunk size? Walk through failure modes of chunks that are too small versus too large and how your choice affects token cost.

**Why it's top 20:** Chunk size is the most consequential parameter in RAG and the one most teams get wrong by defaulting to a library default. It directly multiplies token cost at scale.

**What a strong answer covers:**
- Too small: context sufficiency failure (answer present but meaningless without surrounding text)
- Too large: retrieval precision failure (relevant sentence buried in 1,024 tokens of noise)
- Token cost: chunk size × k × input_price is a significant daily cost at scale (example: 1,024 vs 256 tokens is 4× the context cost)
- Per-document-type sizing: FAQ 256–512, technical docs 512–768, legal 768–1,024, code at function boundaries

**Full answer:** [02_chunking_strategy.md](02_chunking_strategy.md#q6)

---

### 8. Retrieved chunk has the answer but lacks context — parent-child retrieval

> A retrieved chunk contains the answer but the heading is in the previous chunk, the table is in the next. How do you handle this?

**Why it's top 20:** The context boundary problem is the #1 cause of "correct retrieval, wrong answer" failures. It has specific solutions that most teams don't know about.

**What a strong answer covers:**
- Chunk overlap: 10–20% overlap to preserve boundary context (with deduplication at assembly time)
- Parent-child retrieval: index small child chunks for precision, return large parent chunks for context
- Metadata injection at chunk time: document title, section heading, page number prepended to every chunk
- Table-specific handling: convert tables to prose, or index as markdown, never split mid-table

**Full answer:** [02_chunking_strategy.md](02_chunking_strategy.md#q8)

---

### 9. Dense vs sparse vs hybrid retrieval

> Dense vs BM25 vs hybrid — specific failure modes of each and the conditions under which each outperforms.

**Why it's top 20:** Most teams default to dense-only retrieval. A strong answer knows exactly when BM25 wins (exact identifier queries) and understands hybrid RRF without handwaving.

**What a strong answer covers:**
- Dense fails on: product codes, model numbers, contract IDs, rare proper nouns
- BM25 fails on: vocabulary mismatch ("car" vs "automobile"), paraphrase, semantic queries
- Hybrid (RRF): almost always better, maintains two indexes, parallelise retrieval calls
- Dynamic alpha: tune the dense/sparse weight per query type using a lightweight classifier

**Full answer:** [03_embedding_and_indexing.md](03_embedding_and_indexing.md#q12)

---

### 10. Reranking architecture — what a cross-encoder adds and what it costs

> Walk me through your full reranking architecture. What does a cross-encoder add over embedding similarity?

**Why it's top 20:** Reranking is the single highest-ROI improvement in most RAG pipelines and is often skipped. Tests whether the candidate understands why bi-encoders are fast but imprecise.

**What a strong answer covers:**
- Bi-encoder: fast, encodes query and document independently, cannot model query-document interaction
- Cross-encoder: reads both simultaneously, models negation, conditionals, precise factual alignment
- Typical improvement: 5–15pp recall@5 over bi-encoder alone
- Cost table: 100–400ms, $0.05–0.20 per 1,000 queries — worth it for most corpora

**Full answer:** [04_retrieval.md](04_retrieval.md#q16)

---

### 11. Retrieval failure detection — before the LLM hallucinates

> How do you detect when retrieval has failed before the LLM generates a hallucinated answer?

**Why it's top 20:** Retrieval failure is invisible in naive pipelines. The LLM generates a confident answer from bad context with no signal that something is wrong.

**What a strong answer covers:**
- Query-context similarity threshold: if max cosine similarity < 0.3, no chunk is relevant
- LLM faithfulness check: "Is this answer supported by the provided context?" on 100% of queries with a small model (~$0.001/query)
- Empty retrieved set as unambiguous failure
- What to do on failure: retry with query transformation, then graceful "I don't know"
- Corpus gap logging: cluster failed queries to identify missing content

**Full answer:** [04_retrieval.md](04_retrieval.md#q19)

---

### 12. The lost-in-the-middle problem and how to mitigate it

> What is the lost-in-the-middle problem and how does it affect RAG system design?

**Why it's top 20:** Almost no one mentions this in a basic RAG description. Knowing it exists — and having specific mitigations — signals production experience.

**What a strong answer covers:**
- Empirical finding: LLMs attend poorly to content in the middle of long contexts
- Impact: the most relevant chunk at position 8 of 10 receives less attention than chunks at positions 1 and 10
- Book-end ordering: place highest-scored chunk first, second-highest last — free, immediate improvement
- Reduce k: fewer chunks keep relevant content in high-attention zones
- Context compression: extract only the relevant sentences from each chunk before assembly

**Full answer:** [05_context_assembly.md](05_context_assembly.md#q25)

---

### 13. Hallucination prevention — layers and what each still fails to prevent

> How do you prevent the LLM from generating facts not grounded in the retrieved context?

**Why it's top 20:** Every RAG system must address hallucination. A shallow answer says "use a good system prompt." A strong answer layers four independent controls and names what each one still fails to prevent.

**What a strong answer covers:**
- System prompt grounding: reduces hallucination 40–60% but not sufficient alone
- Attribution forcing: every claim must cite a source — uncited claims are flagged
- Post-generation faithfulness check: claim decomposition + NLI entailment or LLM judge ($0.002/query)
- Temperature 0: removes random sampling, grounds generation in the immediate context
- Token cost of faithfulness checking: small model, $0.002/query — worth running on 100%
- Hybrid judge in practice: a secondary LLM-as-judge is strong, but pair it with a deterministic layer — regex/rule checks for known-bad patterns plus a prebuilt FAQ fast-path for common queries — so not every decision is left to the model
- The core tension: pure-LLM (flexible, non-deterministic, costly) vs secondary-LLM + rules engine/regex (predictable, cheaper, but the FAQ/rules can feel like they defeat the "just let the LLM do it" purpose); most production systems settle on the hybrid

**Full answer:** [06_generation_hallucination.md](06_generation_hallucination.md#q26)

---

### 14. When the context is insufficient — detecting and handling gracefully

> The retrieved context is insufficient to answer the question. How does your system detect this and what does it return?

**Why it's top 20:** Most RAG systems either hallucinate when context is missing or crash with empty results. Graceful handling is a production maturity marker.

**What a strong answer covers:**
- Detection signals: retrieval similarity < 0.3, faithfulness check returns "no", empty retrieved set
- Retry with query transformation before declaring failure
- Scoped answer: answer the part that IS covered, explicitly state what's missing
- Corpus gap logging: "I don't know" responses cluster into content gaps worth filling
- Never: generate a confident hallucinated answer to fill the gap

**Full answer:** [06_generation_hallucination.md](06_generation_hallucination.md#q28)

---

### 15. Faithfulness, relevance, context recall — what each metric misses

> What metrics do you use to evaluate a RAG system? What does each measure and what does it fail to capture?

**Why it's top 20:** RAG evaluation is genuinely hard and most teams measure the wrong thing. Knowing the specific blind spots of each metric is essential for designing a meaningful eval pipeline.

**What a strong answer covers:**
- Context recall: measures chunk coverage but a chunk can be "relevant" without containing the actual answer
- Faithfulness: measures hallucination but a faithful answer can be misleadingly incomplete
- Answer relevance: measures whether the response addresses the question but not factual correctness
- The metric no single automated score captures: answer completeness on open-ended synthesis tasks
- Token cost of automated eval: $0.002–0.05/query, sample 5% of production for cost efficiency

**Full answer:** [07_evaluation.md](07_evaluation.md#q31)

---

### 16. DeepEval — what it is and its specific limitations

> What is DeepEval and what are its specific limitations? When does it give misleading scores?

**Why it's top 20:** DeepEval is widely used but poorly understood. Teams that cite DeepEval scores without understanding its failure modes are making decisions on flawed data.

**What a strong answer covers:**
- What it is: a Pytest-style, open-source LLM evaluation framework — metrics like G-Eval (LLM-as-judge with chain-of-thought), faithfulness, answer relevancy, and contextual precision/recall/relevancy, runnable as unit tests in CI
- LLM judge bias: if you generate with GPT-4 and judge with GPT-4, they share biases and scores are inflated
- Judge non-determinism: the same input can score differently across runs — a single pass/fail is noisy without averaging or seeding
- Threshold arbitrariness: the pass/fail cutoff (e.g. 0.5) is user-chosen, not calibrated to your domain — a green suite can still ship bad answers
- Ground-truth dependency: contextual recall and correctness metrics need a labelled `expected_output` golden dataset — expensive to build and keep current
- Cost: LLM-judge metrics cost $0.05–0.10/query — only practical on 1–5% of production traffic
- What to supplement with: NLI-based faithfulness at scale, human eval for quality, domain-specific correctness checks

**Full answer:** [07_evaluation.md](07_evaluation.md#q35)

---

### 17. HyDE — when it helps, when it hurts, and the token cost

> What is HyDE and when does it improve retrieval quality?

**Why it's top 20:** HyDE is a widely cited technique but is often applied indiscriminately. Knowing when it helps (abstract queries, vocabulary mismatch) and when it hurts (factual lookups, proprietary knowledge) demonstrates practical understanding.

**What a strong answer covers:**
- Mechanism: generate a hypothetical document that answers the query, embed it, retrieve similar real documents
- When it helps: short abstract queries (< 5 words), domain vocabulary mismatch, informational "how/why" queries
- When it hurts: factual lookups (LLM may hallucinate wrong facts), proprietary knowledge the LLM hasn't seen
- Token cost: one additional LLM call, ~$0.0002/query with GPT-4o-mini — negligible; latency (200–800ms) is the real constraint

**Full answer:** [10_advanced_patterns.md](10_advanced_patterns.md#q47)

---

### 18. Document-level access control — where enforcement must happen

> How do you enforce document-level access controls in a RAG system?

**Why it's top 20:** Access control in RAG is a security requirement, not a product feature. The specific placement of enforcement (inside the vector store query, not as post-retrieval filter) is a concrete failure mode most candidates miss.

**What a strong answer covers:**
- Filter inside the vector store query: post-retrieval filtering silently drops authorised results when the top-k is full of unauthorised documents
- Namespace isolation for multi-tenant: a namespace scoping error returns zero results (fails safe), not cross-tenant results
- Service account with least privilege: SELECT-only access prevents non-retrieval data exposure
- Audit logging: every retrieval event logged with user ID and document IDs for compliance

**Full answer:** [11_security_access_control.md](11_security_access_control.md#q53)

---

### 19. Prompt injection through document content

> How do you protect against prompt injection through document content?

**Why it's top 20:** Prompt injection through retrieved documents is a real attack vector that most RAG implementations are vulnerable to. Tests security awareness beyond the typical "use HTTPS" level.

**What a strong answer covers:**
- The attack: a malicious document contains "IGNORE PREVIOUS INSTRUCTIONS..." and the LLM follows it
- Input scanning at ingestion: regex/pattern matching for common injection phrases — weak but catches unsophisticated attempts
- Context sandboxing: clearly delimit retrieved content with `<context>` tags and instruct the LLM to treat it as untrusted data
- Output validation: flag responses that contain unexpected content (system instructions, user data) before delivery
- Regular red-team testing: the only way to validate defences against novel attacks

**Full answer:** [11_security_access_control.md](11_security_access_control.md#q54)

---

### 20. PII in documents — preventing surfacing in responses

> Your corpus contains documents with PII. How do you prevent PII from surfacing in responses to unauthorised users?

**Why it's top 20:** PII handling in RAG is a data governance problem that spans ingestion, retrieval, and generation. A complete answer addresses all three stages and names the specific tools and compliance implications.

**What a strong answer covers:**
- Redaction at ingestion: Presidio or equivalent PII detector before any content is indexed
- Access-controlled PII documents: documents that must contain PII are only retrievable by authorised users (Q18's access control)
- Generation-time PII scan: run PII detection on every generated response before delivery
- Differential privacy for aggregate queries: prevent PII leakage through aggregation on small groups
- Audit log requirements for GDPR, HIPAA, SOC 2

**Full answer:** [11_security_access_control.md](11_security_access_control.md#q55)

---

## Quick Study Map

| # | Topic | Section file | Core concept |
|---|-------|-------------|--------------|
| 1 | Ingestion pipeline at 500K docs | [08](08_knowledge_base_ingestion.md) | Queue + idempotency + batching |
| 2 | Incremental updates without full re-index | [08](08_knowledge_base_ingestion.md) | Chunk-level atomic swap |
| 3 | Latency components + token cost trade-offs | [09](09_latency_performance.md) | Critical path + parallelism |
| 4 | Caching + invalidation | [09](09_latency_performance.md) | Schema version hash in cache key |
| 5 | End-to-end architecture + failure detection | [01](01_pipeline_architecture.md) | Per-stage failure domain |
| 6 | RAG vs fine-tuning vs full context | [01](01_pipeline_architecture.md) | Trade-off matrix |
| 7 | Chunk size + token cost | [02](02_chunking_strategy.md) | Precision vs sufficiency vs cost |
| 8 | Parent-child retrieval for context boundaries | [02](02_chunking_strategy.md) | Index small, return large |
| 9 | Dense vs sparse vs hybrid (RRF) | [03](03_embedding_and_indexing.md) | Each fails differently |
| 10 | Cross-encoder reranking | [04](04_retrieval.md) | Interaction modelling |
| 11 | Retrieval failure detection | [04](04_retrieval.md) | Catch before LLM hallucinates |
| 12 | Lost-in-the-middle + book-end ordering | [05](05_context_assembly.md) | Attention is not uniform |
| 13 | Hallucination prevention — 4 layers | [06](06_generation_hallucination.md) | No single technique sufficient |
| 14 | Insufficient context — graceful handling | [06](06_generation_hallucination.md) | Never hallucinate to fill gap |
| 15 | Evaluation metrics + blind spots | [07](07_evaluation.md) | Layered metric |
| 16 | DeepEval limitations | [07](07_evaluation.md) | LLM judge bias |
| 17 | HyDE — when it helps vs hurts | [10](10_advanced_patterns.md) | Vocabulary bridge, not a silver bullet |
| 18 | Document-level access control | [11](11_security_access_control.md) | Filter inside the vector store |
| 19 | Prompt injection through documents | [11](11_security_access_control.md) | Context sandboxing + red-team |
| 20 | PII in corpus | [11](11_security_access_control.md) | Redact at ingestion, scan at output |
