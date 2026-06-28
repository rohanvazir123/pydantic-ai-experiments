# Knowledge Base and Ingestion — Answers

## Q37. Design the document ingestion pipeline for 500K documents with continuous updates.

**Answer:**

A production ingestion pipeline for 500K+ documents must handle scale, failure recovery, incremental updates, and heterogeneous document types. It cannot be a simple script.

**Pipeline architecture:**

```
Document Source (S3/SharePoint/GDrive/APIs)
        │
        ▼
Ingestion Queue (SQS / Kafka / Redis Streams)
        │  (one message per document change event)
        ▼
Ingestion Workers (horizontally scalable)
   ├── Document Fetcher     → download from source
   ├── Parser               → extract text per document type
   ├── Chunker              → split into chunks with metadata
   ├── Embedder             → embed each chunk (batched)
   └── Indexer              → upsert into vector store + metadata store
        │
        ▼
Document Registry (PostgreSQL)
   ├── documents (id, source_url, content_hash, version, status, last_indexed)
   └── chunks (id, document_id, content, embedding_id, chunk_index, metadata)
        │
        ▼
Vector Store (Pinecone / pgvector / Qdrant / Weaviate)
```

**Key design decisions:**

*Idempotency:* Every ingestion job must be idempotent. If the worker crashes after embedding but before indexing, the job must be safely retried. Use the content hash as the idempotency key — if a document's content hash matches the currently indexed version, skip re-processing.

*Batched embedding:* Embedding calls are expensive if done one chunk at a time. Batch 100–500 chunks per API call. At 500K documents × 10 chunks average × 512 tokens = 2.56B tokens to embed. At $0.02/1M tokens: $51.20 for initial ingestion. Very cheap — batch to maximise throughput, not to reduce cost.

*Parallelism:* Run 10–20 worker instances concurrently. Each worker pulls from the queue, processes one document, and acknowledges. Queue visibility timeout must exceed the maximum processing time for a single document (typically 2–5 minutes for a 100-page PDF with OCR).

*Dead letter queue:* Documents that fail processing 3 times go to a dead letter queue for manual review. Do not let permanently failing documents block the queue.

*Progress tracking:* Track ingestion status per document in the registry: `pending`, `processing`, `indexed`, `failed`. This enables resumable ingestion — on failure, resume from where it stopped.

**Token cost for initial ingestion at scale:**

| Step | Cost basis | 500K docs cost |
|------|-----------|---------------|
| OCR/parsing (PDF with tables) | $0.001–0.015/page | $500–7,500 |
| Embedding (500 tokens/chunk, 10 chunks/doc) | $0.02/1M tokens | $51 |
| LLM for metadata enrichment (optional) | $0.002/doc | $1,000 |
| Reranker fine-tuning (optional) | Training cost | Variable |
| **Total** | | **$1,550–$8,550** |

Ongoing incremental ingestion cost: proportional to daily change volume. For a corpus where 1% of documents change daily: 5,000 docs/day × $0.002 = $10/day.

---

## Q38. Document updates and deletions without re-indexing the entire corpus.

**Answer:**

**The naive approach and why it fails:**

Some teams re-index the entire corpus nightly. For 500K documents, a full re-index takes hours and costs $1,550–$8,550 per run. If document update latency of several hours is acceptable and cost is not a concern, this is operationally simple. For most production systems, neither is true.

**Incremental update architecture:**

*Change detection:*
Subscribe to document change events from the source system. S3: S3 Event Notifications. SharePoint: Graph API webhooks. Databases: CDC (change data capture). For systems without webhook support: poll the source and compare content hashes to the document registry.

*Chunk-level update:*
When a document changes:
1. Re-parse and re-chunk the new version
2. Compare new chunks to existing chunks by content hash
3. Delete chunks that no longer exist (soft delete first, hard delete after new chunks are indexed)
4. Add new chunks
5. For changed chunks: delete old version, index new version

Only chunks that actually changed need to be re-embedded. For a 50-page document where only page 12 changed, only the 3 chunks from page 12 are re-processed.

**Consistency during partial update:**

Between the time old chunks are deleted and new chunks are indexed, queries may return no results for this document. Mitigation: index new chunks first, then delete old chunks. This creates a brief period of double-indexing (both old and new versions are retrievable) but guarantees continuity.

**Deletion handling:**

When a document is deleted:
1. Immediately mark all its chunks as `active=false` in the document registry
2. The retrieval query filters `WHERE active=true` — deleted chunks are invisible to retrieval within seconds
3. Remove from the vector index asynchronously (hard delete within hours — vector stores don't support immediate consistent deletes in all implementations)
4. Remove the document from the registry

**What breaks during partial re-indexing:**

If you update a policy document that cross-references another document, and the cross-referenced document is not updated simultaneously, queries about the cross-reference may retrieve inconsistent information (old chunk from one, new chunk from the other). For coordinated multi-document updates, use an atomic update group: mark all documents in the group as `update_pending`, process all of them, then activate atomically.

---

## Q39. Document staleness — indexed document diverges from its source.

**Answer:**

Staleness is the gap between what is indexed and what is true. A retrieved document that is 6 months out of date is potentially more dangerous than no retrieved document — it provides confidently wrong information.

**Detection:**

*Content hash comparison:*
Periodically (daily or weekly) re-fetch each source document and compare its content hash to the hash stored in the document registry. A mismatch indicates the document has changed at the source but has not been re-indexed.

*Source-system timestamp comparison:*
If the source system provides `last_modified` metadata, compare it to the `last_indexed` timestamp in the registry. If `source.last_modified > registry.last_indexed`, the document is stale.

*Scheduled staleness audit:*
Run a daily job that:
1. Fetches `last_modified` for all 500K documents from the source
2. Identifies documents where `source.last_modified > registry.last_indexed`
3. Queues them for re-ingestion

For most corpora, < 5% of documents change daily — the audit and re-ingestion workload is manageable.

**User-facing impact of undetected staleness:**

If a policy changed on March 1 and the indexed version is from February, the RAG system provides the old policy until the staleness is detected and corrected. In compliance or legal contexts, this can cause users to act on outdated rules.

**Mitigation:**
- Surface document `last_indexed` date in citations: "According to the HR Handbook (last updated: Feb 15, 2024)"
- For documents with known short shelf-lives (market data, regulatory updates), set a TTL after which the document is flagged for re-validation even without a source change event
- Alert users when a cited document's last_indexed date is more than N days old

---

## Q40. PDFs with scanned images, tables, charts, embedded figures.

**Answer:**

Non-text elements in PDFs are the most common source of information loss in RAG ingestion. A financial report where 40% of the information is in tables and charts cannot be adequately served by text-only extraction.

**Scanned images / scanned PDFs:**

Scanned PDFs are images — there is no text layer. Standard PDF text extractors (PyPDF2, pdfminer) return nothing or garbled text.

*Solution: OCR.* Tesseract (open source), Google Cloud Document AI, AWS Textract, Azure Document Intelligence. Accuracy comparison:
- Tesseract: free, acceptable for clean scans, poor for degraded or complex layouts
- Cloud APIs: 95%+ accuracy on most documents, $0.001–0.015/page, handles complex layouts, tables, and multi-column text

For a 500K document corpus averaging 10 pages, OCR adds $5,000–75,000 to ingestion cost. Budget accordingly.

**Tables:**

Tables are the highest-value information structure in most business documents. Naive text extraction destroys column alignment:

```
# What extraction produces (wrong):
Revenue Q1 Q2 Q3 Total Product A 4.2 4.8 5.1 14.1 Product B 2.1 2.3 2.8 7.2

# What it means:
| Revenue | Q1  | Q2  | Q3  | Total |
|---------|-----|-----|-----|-------|
| Product A | 4.2M | 4.8M | 5.1M | 14.1M |
| Product B | 2.1M | 2.3M | 2.8M | 7.2M  |
```

Use document intelligence APIs that understand table structure (Textract, Azure DI, Docling). Index tables in two formats:
1. Markdown table: preserves structure, embeddable, LLM-readable
2. Structured JSON: enables SQL-like queries on table data

**Charts and figures:**

Charts encode quantitative information visually. Text extraction gets nothing. Options:
1. Use multimodal models (GPT-4o, Claude 3.5 Sonnet) to describe the chart: "This bar chart shows monthly revenue from January to December 2024, with a peak in Q3..."
2. Index chart captions only (cheap but lossy)
3. Store charts as image embeddings (CLIP) alongside text embeddings — enables image-based retrieval

*Cost of multimodal chart processing:*
GPT-4o: ~$0.01 per image. 500K docs × 2 charts average = 1M images → $10,000 for initial ingestion. Significant but often worth it for information-dense corpora.

---

## Q41. Private, access-controlled documents in a multi-tenant corpus.

**Answer:**

Document-level access control is a security requirement, not a product feature. If a user can retrieve content from a document they are not authorised to read, the RAG system is a data leakage vector.

**Where enforcement must happen:**

*At the retrieval layer (not just the application layer):*

```python
def retrieve_chunks(
    query_embedding: np.ndarray,
    user_permissions: set[str],  # document IDs the user can access
    k: int = 10
) -> list[Chunk]:
    # Filter BEFORE similarity search, not after
    return vector_store.search(
        vector=query_embedding,
        filter={"document_id": {"$in": list(user_permissions)}},
        top_k=k
    )
```

The permission filter must be applied inside the vector store query — not as a post-retrieval filter. Post-retrieval filtering means: (1) you retrieve k chunks, (2) some are unauthorised, (3) you filter them out, leaving fewer than k results. Worse: if all k retrieved chunks are unauthorised, you return nothing even though authorised documents exist — the authorised content is never retrieved.

**Permission model:**

```
User → Group → Roles → Document Permissions
```

Each document has an ACL (Access Control List): a list of user IDs, group IDs, or role names that can read it. At query time, resolve the user's effective permissions to a list of document IDs. Pass this list as a filter to the retrieval query.

**Performance:**

A user with access to 50,000 of 500,000 documents passes a 50,000-element filter to every retrieval query. Vector databases handle this efficiently with pre-indexed metadata filters (HNSW with metadata filtering in Qdrant, Weaviate, Pinecone). Latency impact: 10–30ms additional overhead for the filter evaluation.

**Tenant isolation:**

For a multi-tenant SaaS deployment where each tenant is completely isolated (no cross-tenant sharing):
- Maintain a separate vector namespace per tenant
- Each tenant's query only searches their namespace
- A namespace lookup error returns zero results (fails safe)

This is safer than row-level filtering: even if the filter is accidentally omitted, a namespace-scoped search cannot cross tenant boundaries.

**Caching with access control:**

Do NOT cache query results globally. A cached result for User A must not be served to User B if their permissions differ. Cache key must include the user's permission scope hash.
