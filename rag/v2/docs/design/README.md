# RAG v2 — Design Deep-Dives

## Table of Contents

- [Ingestion](#ingestion)
- [Redis Streams & Async Workers](#redis-streams--async-workers)
- [Caching](#caching)
- [Retrieval](#retrieval)
- [Reliability & Safety](#reliability--safety)
- [Security & Memory](#security--memory)
- [Deployment](#deployment)
- [Evaluation & Testing](#evaluation--testing)

---

Detailed design docs split out from [../RAGV2_DESIGN.md](../RAGV2_DESIGN.md). Start there for the system overview, architecture diagram, and data flow. Come here for implementation depth on a specific subsystem.

| Document | What it covers |
|----------|---------------|
| [ARCHITECTURE_PROPOSAL.md](ARCHITECTURE_PROPOSAL.md) | Goals and multi-corpus design |
| [SYSTEM_DESIGN_CONSTRAINTS.md](SYSTEM_DESIGN_CONSTRAINTS.md) | Load model, SLAs, token budgets, cost model, circuit breakers |
| [MODULE_LAYOUT.md](MODULE_LAYOUT.md) | Full `knowledge/` package tree with per-subpackage guide |
| [INGESTION.md](INGESTION.md) | Ingestion pipeline, Docling-graph parallel paths, KG extraction, AGE store design, ontology API |
| [REDIS_STREAMS.md](REDIS_STREAMS.md) | Message bus design, async worker lifecycle, DLQ, job status, why search stays sync |
| [CACHING.md](CACHING.md) | L1 in-process LRU, L2 Redis, L3 pgvector semantic cache — TTLs, invalidation, hit rate targets |
| [RETRIEVAL.md](RETRIEVAL.md) | Hybrid search, RRF fusion, CrossEncoder reranking, confidence scoring, confidence-aware pipeline, model tiering |
| [RELIABILITY.md](RELIABILITY.md) | V1–V6 query validation, hook system, guardrail architecture, error handling, retry & resilience |
| [SECURITY.md](SECURITY.md) | JWT RS256 auth, JWE payload encryption, HTTPS/TLS, RBAC, audit log, memory architecture |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Docker Compose local dev, packaging, cloud production deployment, SaaS model, log storage |
| [EVALUATION.md](EVALUATION.md) | Offline eval system, retrieval metrics, faithfulness/relevance scoring, load & chaos testing, implementation phases |
