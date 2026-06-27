# RAG System Design — Deep-Dive Answers

Detailed answers to every topic in [rag_system_design.md](../rag_system_design.md). One file per section. Token cost considerations woven throughout.

## Files

| File | Section | Topics |
|------|---------|--------|
| [00_top20_curated.md](00_top20_curated.md) | Curated Top 20 | Highest-signal topics with selection rationale and quick study map |
| [01_pipeline_architecture.md](01_pipeline_architecture.md) | Pipeline Architecture | End-to-end design, LLM vs retrieval boundary, multi-document queries, advanced vs naive RAG, RAG vs fine-tuning |
| [02_chunking_strategy.md](02_chunking_strategy.md) | Chunking Strategy | Chunk size trade-offs and token cost, semantic vs fixed-size, context boundary problem, heterogeneous docs, document hierarchy |
| [03_embedding_and_indexing.md](03_embedding_and_indexing.md) | Embedding and Indexing | Model selection, dense vs sparse vs hybrid, document updates, index scaling (flat/HNSW/IVF/PQ), multilingual |
| [04_retrieval.md](04_retrieval.md) | Retrieval | Reranking architecture and cost, top-k tuning, query transformation (HyDE/expansion/multi-query), retrieval failure detection, multi-hop reasoning |
| [05_context_assembly.md](05_context_assembly.md) | Context Assembly | Ordering and deduplication, context budget and cost, contradiction handling, structure preservation, lost-in-the-middle |
| [06_generation_hallucination.md](06_generation_hallucination.md) | Generation and Hallucination | Hallucination prevention layers, citation and attribution, insufficient context handling, "I don't know" calibration, parametric vs retrieved knowledge |
| [07_evaluation.md](07_evaluation.md) | Evaluation | Retrieval and generation metrics, ground-truth construction, automated hallucination detection, component-level eval, RAGAS limitations, open-ended evaluation |
| [08_knowledge_base_ingestion.md](08_knowledge_base_ingestion.md) | Knowledge Base and Ingestion | Ingestion pipeline at scale, incremental updates, staleness detection, multimodal document processing, access-controlled multi-tenant corpus |
| [09_latency_performance.md](09_latency_performance.md) | Latency and Performance | Latency components and critical path, caching layers and invalidation, streaming, high concurrency bottlenecks, pre-generation |
| [10_advanced_patterns.md](10_advanced_patterns.md) | Advanced Patterns | HyDE, RAG fusion, agentic RAG, long documents, self-RAG, hybrid structured + unstructured retrieval |
| [11_security_access_control.md](11_security_access_control.md) | Security and Access Control | Document-level ACL, namespace isolation, prompt injection through documents, PII prevention |
