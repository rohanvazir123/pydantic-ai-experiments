# tests/retrieval/

## Table of Contents

- [What This Is](#what-this-is)
- [Requirements](#requirements)
- [Running](#running)

---

## What This Is

Retrieval quality tests against gold datasets. Computes IR metrics (HitRate, MRR, NDCG, Precision, Recall) and verifies that the retriever meets minimum quality thresholds. Also tests corpus isolation.

For metric definitions and threshold values see `TEST_QA_REFERENCE.md §1 Retrieval Metrics`.

---

## Requirements

- PostgreSQL with NeuralFlow AI corpus ingested (run `make ingest-sample` or `POST /v1/ingest`)
- Ollama running with `nomic-embed-text:latest`

---

## Running

```bash
python -m pytest tests/retrieval/ -v --log-cli-level=INFO
```

Output includes a metrics table logged to INFO level showing HitRate/MRR/NDCG at K=1,3,5 and P95 latency.
