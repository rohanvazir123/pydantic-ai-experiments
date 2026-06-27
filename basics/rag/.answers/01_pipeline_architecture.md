# Pipeline Architecture — Answers

## Q1. Walk me through the end-to-end architecture of a production RAG system.

**Answer:**

A production RAG pipeline has six stages, each with its own failure class:

**Stage 1 — Query understanding and intent classification (< 20ms):**
Classify the query before anything else: Is it answerable from the corpus (analytical), conversational, or out of scope? A lightweight classifier handles this. Failure mode: sending a conversational query through retrieval, wasting latency and producing an LLM response that synthesises irrelevant chunks. Detection: monitor queries where retrieved context has near-zero similarity to the question — these are retrieval misroutes.

**Stage 2 — Query transformation (0–500ms, optional):**
Rewrite or expand the query to improve retrieval. Techniques: HyDE (generate a hypothetical answer and embed it), query expansion (add synonyms and related terms), multi-query (generate N variants). Failure mode: the transformation makes the query worse — adding noise or shifting the semantic focus. Detection: A/B test transformation vs raw query on a retrieval recall benchmark before enabling in production.

**Stage 3 — Retrieval (50–200ms):**
Embed the query, run ANN search against the vector index, optionally fuse with sparse (BM25) results, rerank with a cross-encoder. Returns top-k chunks. Failure mode: the relevant chunk is not in top-k (retrieval miss). Detection: post-generation faithfulness check — if the generated answer cannot be grounded in any retrieved chunk, retrieval likely missed. Track this rate per query class.

**Stage 4 — Context assembly (< 10ms):**
Select, order, deduplicate, and format the retrieved chunks into the prompt context. Failure mode: ordering chunks in an order that confuses the LLM (relevant chunk buried in the middle), including redundant chunks that dilute the signal, or including contradictory chunks without flagging the contradiction. Detection: validate that context token count is within budget; flag queries where retrieved chunks have pairwise similarity > 0.95 (likely duplicates).

**Stage 5 — Generation (500ms–3s):**
The LLM receives the query and the assembled context and generates an answer, ideally with citations. Failure modes: hallucination (generating facts not in context), over-refusal (saying "I don't know" when the context contains the answer), citation fabrication. Detection: automated faithfulness check (does each claim in the answer appear in the retrieved context?).

**Stage 6 — Post-processing and citation verification (< 50ms):**
Verify citations are valid (the cited chunk actually supports the claim), format the response, apply any content filters. Failure mode: a cited chunk does not actually support the claim it is attributed to — citation hallucination. Detection: for each (claim, citation) pair, compute semantic similarity between the claim and the cited chunk. Similarity < 0.5 is a likely fabricated citation.

**Cross-cutting without humans:** Build a confidence score from: retrieval rank of top chunk, faithfulness score of generated answer, citation verification score. Responses below a threshold enter an async human review queue.

---

## Q2. Where do you draw the boundary between retrieval and generation?

**Answer:**

**What retrieval handles:**
- Surface the right information from the corpus
- Narrow the information space from millions of documents to tens of passages
- Guarantee that the answer is grounded in a specific source

**What retrieval fundamentally cannot do:**
- Synthesise, reason across, or reconcile conflicting information — that is the LLM's job
- Handle queries requiring procedural reasoning ("if X then Y then Z") — retrieval finds facts, not reasoning chains
- Fill gaps when the corpus does not contain the answer — retrieval can only surface what exists

**Consequences of asking retrieval to do too much:**

*Over-filtering:* If you try to use retrieval to "pre-answer" the question (retrieve the single best chunk and skip context assembly), you lose the LLM's ability to synthesise across multiple sources. A question like "What are the trade-offs between A and B?" may require three chunks — one about A, one about B, and one comparing them — and no single chunk contains the full answer.

*Under-retrieval:* Trusting retrieval to rank chunks correctly without a reranker means the LLM receives the embedding-ranked top-k, which may have the most relevant chunk at position 8 out of 10 due to embedding model limitations. The LLM should get the best chunks, not the embedding-similarity-ranked chunks.

**The correct boundary:** Retrieval handles surface and scope. The LLM handles synthesis, reasoning, and formulation. Never ask the retrieval layer to make judgements about answer quality — that conflates two different problems and makes both harder to debug.

---

## Q3. How does your system handle single-document vs multi-document queries?

**Answer:**

The same pipeline handles both, but the failure modes are different and require specific mitigations.

**Single-document queries:**
"What is the refund policy?" — answerable from one passage. Here the primary risk is retrieval miss (the right chunk is not retrieved) or context contamination (irrelevant chunks are included alongside the correct one and the LLM hedges its answer because of the noise). Mitigation: high-precision retrieval with a tight similarity threshold.

**Multi-document queries:**
"Compare the refund policy in the 2023 handbook with the current policy" or "Summarise the three proposals submitted in Q3" — requires synthesising across multiple documents. The pipeline must:

1. Retrieve chunks from the correct documents (retrieval breadth over precision)
2. Assemble context that preserves document attribution — which chunk came from which document
3. Prompt the LLM to synthesise across sources, not just summarise the first chunk

**What breaks for multi-document queries in a naive pipeline:**
- Retrieval returns the top-k most similar chunks globally, which may all come from one document
- The LLM generates an answer based on the dominant document and ignores the others
- Citations attribute all claims to the same source even though the question required cross-document synthesis

**Mitigation — diversity-aware retrieval:**
After initial retrieval, enforce document diversity: if top-k contains > 3 chunks from the same document, replace lower-ranked same-document chunks with the top-ranked chunk from the next document in the ranking. This ensures the context represents multiple sources.

**Mitigation — explicit multi-document prompting:**
Structure the context with clear document boundaries:
```
[Document 1: 2023 Employee Handbook, Section 4.2]
Employees may request refunds within 30 days...

[Document 2: 2024 Policy Update, Section 2.1]
Effective January 2024, refund requests must be submitted within 14 days...
```
The LLM can then compare and synthesise rather than treating all chunks as a single undifferentiated mass.

---

## Q4. Advanced RAG vs naive RAG — when is the complexity justified?

**Answer:**

Naive RAG: embed query → retrieve top-k by cosine similarity → generate. Simple, fast, debuggable.

Advanced RAG adds: query transformation, reranking, multi-query, HyDE, context compression, iterative retrieval. Each addition adds latency, cost, and failure modes.

**When naive RAG is sufficient:**
- Single-domain corpus with consistent terminology (query language closely matches document language)
- Short, precise factual queries ("What is the return policy?")
- High-quality, well-structured documents with minimal noise
- Latency SLA is tight (< 1s total)

**When advanced RAG is justified — specific conditions:**

*Reranking:* Justified when embedding similarity recall@k is < 80% on your evaluation set. A cross-encoder reranker adds 100–300ms but improves precision significantly for ambiguous queries. Rule of thumb: if your queries are longer than 10 words on average or involve domain-specific terminology, add reranking.

*Query expansion / HyDE:* Justified when queries are short and abstract ("AI safety regulations") but documents are long and specific. Expanding the query to match document vocabulary improves recall. Adds one LLM call (~500ms). Not justified when queries already closely match document terminology.

*Multi-query:* Justified for queries with multiple sub-questions or when queries are known to be ambiguous. Generates N query variants, retrieves for each, deduplicates. Adds N embedding calls + retrieval latency. Justified when single-query retrieval misses > 15% on your evaluation set.

*Context compression:* Justified when retrieved chunks are long (> 500 tokens) and only a small part is relevant. A compression step (extract the relevant sentences from each chunk) reduces context length by 40–60%, improving both LLM attention and cost. Adds one LLM call per chunk.

**The honest answer:** Measure first. Add each advanced component only after you have benchmarked that naive RAG is failing on a specific query class, and verify the addition improves that class without regressing others.

---

## Q5. RAG vs fine-tuning vs full context — when does each win?

**Answer:**

**Full context window (stuff everything in):**
Put all documents in the context window. Simple, no retrieval infrastructure, always has the right information.

*When it works:* Corpus is small (< 50 pages), queries need full-corpus understanding, latency is not a concern, cost is acceptable.
*When it breaks:* Corpus exceeds context window, cost is proportional to corpus size × queries per day, LLM attention degrades at long context (lost-in-the-middle problem), documents update frequently (re-inject every time).

**Fine-tuning:**
Bake knowledge into model weights.

*When it works:* Knowledge is stable (rarely changes), the knowledge is stylistic or behavioural ("always respond in the tone of our brand"), query patterns are highly repetitive, you need fast inference with no retrieval overhead.
*When it breaks:* Knowledge needs to be updated frequently — fine-tuning on updated data is expensive and slow. Fine-tuned models still hallucinate and do not provide citations. Cannot answer "where did you get that?" — the knowledge is opaque. Knowledge cutoff issues: if you fine-tune today and the document changes tomorrow, the model is wrong until the next fine-tuning run.

**RAG:**
Retrieve relevant context at query time, generate grounded in that context.

*When it works:* Knowledge changes frequently, source attribution is required, corpus is large and heterogeneous, you need to add new documents without retraining.
*When it breaks:* Queries require reasoning across the entire corpus (not just a few chunks), queries require implicit world knowledge not in the corpus (RAG cannot hallucinate usefully), retrieval latency is prohibitive, corpus has many near-duplicate documents that confuse retrieval.

**Hybrid (fine-tune + RAG):**
Fine-tune the model to follow RAG-style instructions well (prefer retrieved context over parametric memory), then use RAG for knowledge. This is the production recommendation for most serious deployments: fine-tuning improves instruction following and grounding behaviour; RAG provides current, attributable knowledge.
