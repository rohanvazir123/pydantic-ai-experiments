# RAG v2 — Documentation

## Table of Contents

- [REST API Reference](#rest-api-reference)
- [System Design](#system-design)
- [Datastore Reference](#datastore-reference)
- [Agent Prompts](#agent-prompts)
- [Test Plan](#test-plan)
- [Test QA Reference](#test-qa-reference)
- [Implementation TODO](#implementation-todo)

---

| Document | Description |
|----------|-------------|
| [REST_API.md](design/REST_API.md) | REST API reference: all endpoints, request/response shapes, status codes |
| [RAGV2_DESIGN.md](RAGV2_DESIGN.md) | System overview: architecture diagram, data flow, goals, SLAs, module layout |
| [DATASTORE.md](DATASTORE.md) | Complete datastore reference: PostgreSQL/pgvector, AGE, Redis, schema |
| [PROMPTS.md](PROMPTS.md) | All agent system prompts, dynamic instructions, and structured output schemas |
| [LOCAL_LLM_GUIDE.md](LOCAL_LLM_GUIDE.md) | Running Ollama locally and on cloud GPUs (RunPod): VRAM requirements, model tiers |
| [TESTS.md](TESTS.md) | Test plan: categories, requirements, how to run each suite |
| [TEST_QA_REFERENCE.md](TEST_QA_REFERENCE.md) | Metric formulas, thresholds, and acceptance criteria for every test type |
| [TODO_implementation.md](TODO_implementation.md) | Bottom-up build plan: phases, deliverables, and test gates |

### Design deep-dives (`design/`)

| Document | Description |
|----------|-------------|
| [design/INGESTION.md](design/INGESTION.md) | Ingestion pipeline, KG extraction, AGE store, ontology API |
| [design/REDIS_STREAMS.md](design/REDIS_STREAMS.md) | Message bus, async worker lifecycle, DLQ, job status |
| [design/CACHING.md](design/CACHING.md) | L1/L2/L3 cache layers, TTLs, invalidation guide |
| [design/RETRIEVAL.md](design/RETRIEVAL.md) | Hybrid search, RRF, reranking, confidence-aware pipeline, model tiering |
| [design/RELIABILITY.md](design/RELIABILITY.md) | Query validation, guardrails, error handling, retry strategy |
| [design/SECURITY.md](design/SECURITY.md) | JWT/JWE/RBAC, memory architecture |
| [design/DEPLOYMENT.md](design/DEPLOYMENT.md) | Docker Compose, packaging, cloud deployment, SaaS model |
| [design/EVALUATION.md](design/EVALUATION.md) | Eval system, load/chaos testing, implementation phases |
