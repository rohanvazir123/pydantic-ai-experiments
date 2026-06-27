# Embedding and Indexing — Answers

## Q11. How do you choose an embedding model for a new domain?

**Answer:**

The embedding model is the most critical component decision in RAG — it determines what "similar" means at retrieval time. Getting it wrong means retrieval will never be reliable regardless of how good your chunking and generation are.

**Evaluation criteria:**

*1. Domain terminology coverage:*
General-purpose models (OpenAI text-embedding-3-large, nomic-embed-text, BGE-large) are trained on web-scale data. They handle general business vocabulary well but may not understand domain-specific abbreviations, product names, or technical jargon.

Test: take 20 domain-specific terms from your corpus. Compute their embeddings. Check that semantically related terms cluster together (high cosine similarity) and unrelated terms are far apart. If "ARR" clusters near "array" instead of "annual recurring revenue", the model needs domain adaptation.

*2. Retrieval benchmark performance:*
Run your golden query set against candidate models. Measure recall@10 (percentage of queries where the correct chunk is in the top 10). Do this before committing to a model — a model that ranks 3rd on MTEB might rank 1st on your specific domain.

*3. Embedding dimension and cost:*
| Model | Dimensions | Cost | Notes |
|-------|-----------|------|-------|
| text-embedding-3-small | 1,536 | $0.02/1M tokens | Fast, cheap, strong general performance |
| text-embedding-3-large | 3,072 | $0.13/1M tokens | Best general accuracy, 6.5× cost |
| nomic-embed-text v1.5 | 768 | Free (self-hosted) | Strong open-source, runs locally |
| BGE-large-en-v1.5 | 1,024 | Free (self-hosted) | SOTA open-source for English |
| domain fine-tuned | varies | Training cost amortised | Best for specialised domains |

*4. Sequence length:*
Most models have a 512-token input limit. text-embedding-3-large handles up to 8,191 tokens. For long-chunk strategies (1,024+ tokens), use a model with a longer input window or chunk below the model's limit.

**General-purpose vs domain fine-tuned:**

General-purpose models are the right starting point for 90% of deployments. They generalise well and require no training infrastructure. Domain fine-tuning adds meaningful improvement when:
- Your corpus uses technical vocabulary that general models conflate with common words (e.g., "transformer" means a neural network architecture, not an electrical device)
- Your queries are short and specialised (a 3-word query requires the embedding model to understand the exact domain context)
- You have > 10,000 labeled (query, relevant document) pairs to train on

**Token cost for embedding the corpus:**
Embedding 500,000 chunks × 512 tokens = 256M tokens. At $0.02/1M tokens (text-embedding-3-small): $5.12 initial ingestion cost. Incremental updates are much cheaper. This is negligible — model quality matters far more than embedding cost.

---

## Q12. Dense vs sparse (BM25) vs hybrid retrieval.

**Answer:**

**Dense retrieval (semantic embedding similarity):**
Embeds both query and documents into a vector space. Retrieves by cosine/dot-product similarity.

*Strengths:* Handles synonymy ("car" retrieves "automobile"), paraphrase, and semantically related concepts. Works well for natural language queries that don't exactly match document vocabulary.
*Fails when:* The query contains exact identifiers — product codes, model numbers, contract IDs, version numbers. Dense retrieval may not rank "SKU-4821-B" above "SKU-4821-A" because they are semantically near-identical. Also fails for rare proper nouns that the embedding model has not seen in training.

**Sparse retrieval (BM25 / TF-IDF):**
Scores documents by exact term frequency, weighted by inverse document frequency.

*Strengths:* Exact keyword matching. "SOC 2 Type II compliance report Q3 2023" will precisely retrieve the document containing those exact terms. Essential for technical, regulatory, and identifier-heavy queries.
*Fails when:* Vocabulary mismatch — the user asks "car maintenance cost" but the document says "vehicle upkeep expense." BM25 finds no overlap and returns zero results.

**Hybrid retrieval:**
Run both dense and sparse retrieval in parallel. Merge results using Reciprocal Rank Fusion (RRF):

```
score_hybrid(doc) = 1/(k + rank_dense(doc)) + 1/(k + rank_sparse(doc))
```
where k=60 is a smoothing constant.

RRF is rank-based (not score-based), so it does not require normalising the raw scores from two different retrieval systems.

*Strengths:* Best of both. Dense handles semantic matching; sparse handles keyword precision. Consistently outperforms either alone on diverse query sets.
*Cost:* Requires maintaining two indexes (vector + inverted index). Two retrieval calls in parallel (adds no latency if parallelised). Most vector databases (Weaviate, Qdrant, Elasticsearch) support hybrid natively.

**Decision rule:**
- Start with hybrid. It is almost always better.
- If corpus has no exact identifiers (pure prose), dense alone may be sufficient.
- If users routinely search by exact IDs or codes, ensure BM25 weight is tuned up.

**Tuning the dense/sparse weight:**
Don't use a fixed weight. Some query types benefit from more dense signal; others from more sparse. Train a lightweight classifier to predict the optimal alpha (dense weight) per query type:

```python
alpha = dense_weight_classifier(query)  # returns 0.0 to 1.0
score = alpha * score_dense + (1 - alpha) * score_sparse
```

---

## Q13. How does your index handle document updates and deletions?

**Answer:**

This is the hardest operational problem in RAG after initial indexing. A document that has been updated but whose old chunks remain indexed will cause the system to retrieve stale information — potentially with higher confidence than the new version if the old version is more topically aligned.

**The three update types and their handling:**

**1. Document update (content change):**
The document exists but its content changed (a policy was amended, a specification was revised).

*Naive approach:* Delete old chunks, re-chunk and re-embed the new version. Works but requires tracking which chunks belong to which document.

*Implementation:* Every chunk stores `{"document_id": "doc_042", "content_hash": "a3f1...", "version": 3}`. On update: fetch all chunk IDs for `document_id=doc_042`, delete them from the vector index, re-ingest the new version. Use a soft delete pattern — mark chunks as `deleted=true` first, verify the new chunks are indexed, then hard delete.

*Consistency window:* Between soft delete and re-indexing, queries may return no results for this document. Design ingestion as an atomic swap: index new chunks before deleting old ones, then atomically mark old chunks as inactive. Trade-off: briefly doubles storage for the document.

**2. Document deletion:**
The document is removed entirely (an outdated policy, a retracted document).

*Risk:* If not handled, deleted document chunks remain in the index indefinitely. A query may retrieve a chunk from a document that no longer exists, and the citation will point to a deleted source.

*Implementation:* Soft delete immediately (mark chunks as `deleted=true`). The retrieval query filters `WHERE deleted = false`. Hard delete during a nightly cleanup job. Index deletes are propagated to the vector store within minutes; no user should see deleted content after the soft delete.

**3. New document addition:**
The simplest case. Ingest, chunk, embed, index. The only risk is concurrency — if two ingestion jobs process the same document simultaneously, you get duplicate chunks.

*Implementation:* Use an idempotency key (the document ID) with a database-level unique constraint. The ingestion job acquires a lock on the document ID before processing.

**Partial re-indexing consistency issues:**
If a corpus update changes 500 documents out of 100,000, the 500 updated documents are in their new state while the 99,500 unchanged documents are in their old state. A query that spans old and new documents may synthesise information from mixed versions. If the update is a policy change that affects cross-references, this creates a coherence gap.

Mitigation: for coordinated updates (policy rollouts, software release notes), process all related documents atomically in a single ingestion transaction. Mark all documents in a "release" with the same version tag and activate them simultaneously.

---

## Q14. Vector indexing strategy as corpus scales — flat vs IVF vs HNSW vs PQ.

**Answer:**

**Flat index (brute-force):**
Compute exact dot-product/cosine similarity against every vector in the index.

*Use when:* Corpus < 100,000 chunks. Exact results, no recall degradation. Latency: O(n). At 100K chunks × 1,536 dimensions, a flat search takes ~50ms on a single CPU core — acceptable for small corpora.
*Breaks when:* Corpus > 500,000 chunks. Search latency becomes unacceptable (> 500ms per query).

**HNSW (Hierarchical Navigable Small World):**
Graph-based ANN. Builds a multi-layer proximity graph at index time.

*Use when:* 100K–50M chunks. Sub-millisecond query latency. Recall@10 of 95%+ with tuned parameters. The right choice for most production RAG deployments.
*Trade-offs:* High memory footprint (HNSW stores the graph structure alongside vectors — 2–3× the raw vector storage). Index build time is O(n log n). Parameters `ef_construction` (build quality, higher = better recall + slower build) and `ef_search` (query quality, higher = better recall + slower query) must be tuned.
*Token cost implication:* HNSW memory usage at scale: 100M chunks × 1,536 dims × 4 bytes/float = ~600GB raw vectors. HNSW overhead adds another ~300GB. This requires a cluster of vector database nodes, which has significant infrastructure cost.

**IVF (Inverted File Index):**
Clusters vectors at index time. At query time, searches only the nearest N clusters.

*Use when:* 10M–1B chunks. Lower memory than HNSW. Good for batch/offline retrieval.
*Trade-offs:* Recall degrades more sharply than HNSW as nprobe (clusters searched) is reduced. Sensitive to clustering quality — poor cluster initialisation hurts recall. Requires periodic re-clustering as the corpus grows.

**Product Quantization (PQ) / Scalar Quantization:**
Compresses vector dimensions to reduce memory and increase throughput.

*Use when:* Memory is the primary constraint. PQ compresses 1,536-dim float32 vectors to ~64–256 bytes (vs 6,144 bytes uncompressed). 24–96× memory reduction.
*Trade-offs:* Recall degrades by 3–8pp compared to exact search. Latency improves due to smaller memory footprint (better cache utilisation).

**Recommended strategy by corpus size:**
| Corpus size | Strategy |
|-------------|----------|
| < 100K chunks | Flat index + exact search |
| 100K–10M chunks | HNSW (ef_construction=200, M=16) |
| 10M–100M chunks | IVF-HNSW hybrid or HNSW with scalar quantization |
| > 100M chunks | IVF-PQ + reranker to compensate for recall loss |

---

## Q15. How do you handle a multilingual corpus?

**Answer:**

**Option A — Multilingual embedding model, single index:**
Use a model trained on multilingual data (multilingual-e5-large, LaBSE, paraphrase-multilingual-mpnet-base-v2). A single embedding space represents all languages. A query in French retrieves relevant chunks in French, English, or Spanish.

*Strengths:* Simple architecture. Works reasonably well for cross-lingual retrieval (query in one language, documents in another).
*Failure modes:*
- Multilingual models sacrifice per-language quality. A English-only model will outperform a multilingual model on English-only queries by 5–15pp recall.
- Low-resource languages are underrepresented in training. Retrieval quality for languages like Thai, Swahili, or Catalan is significantly lower than for English, French, or German.
- Code-switching within a document (multiple languages in one paragraph) can confuse the embedding.

**Option B — Language-specific indexes:**
Maintain a separate vector index per language. Route queries to the correct index based on detected language.

*Strengths:* Best retrieval quality per language. Language-specific models (CamemBERT for French, BERT-base-german for German) outperform multilingual models significantly.
*Failure modes:*
- Language detection errors route a query to the wrong index.
- Cross-lingual queries ("Find the French regulation that corresponds to this English policy") cannot be handled — each index only contains one language.
- Operational complexity: N indexes instead of one. Schema changes must be replicated to all language indexes.

**Option C — Translate-then-retrieve:**
Translate all documents to a single pivot language (usually English) at ingestion time. Embed and index the translated versions. At query time, translate the user's query to English, retrieve, then optionally translate the answer back.

*Strengths:* Best English-language retrieval quality. Simplest single-language pipeline.
*Failure modes:* Translation introduces errors. Legal and technical documents lose precision in translation. Translation adds latency and cost ($0.015–$0.05 per 1,000 tokens for neural MT). Loses the nuance of the original language, which may matter for regulatory or legal contexts.

**Production recommendation:**
For corpora with a dominant language (> 80% English), use a strong English model for the dominant language and a multilingual model for the remainder. Route based on detected language. For genuinely multilingual corpora with balanced language distribution, use multilingual-e5-large as a starting point and measure per-language recall degradation — accept the quality trade-off for operational simplicity, or accept operational complexity for per-language quality.
