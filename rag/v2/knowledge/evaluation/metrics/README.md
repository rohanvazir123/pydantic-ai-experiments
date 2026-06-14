# knowledge/evaluation/metrics/

## Table of Contents

- [What This Is](#what-this-is)
- [Files](#files)

---

## What This Is

Individual metric implementations. Each file is independent and can be tested in isolation — no DB, no LLM required for the pure metric math (only faithfulness and answer_relevance call a nano model).

For formulas, thresholds, and interpretation guidance see `TEST_QA_REFERENCE.md §1–§4`.

---

## Files

| File | Metrics computed | Requires |
|------|-----------------|----------|
| `retrieval.py` | HitRate@K, MRR@K, NDCG@K, Precision@K, Recall@K, confidence distribution | None (pure math) |
| `faithfulness.py` | Claim decomposition → NLI check → faithfulness score (0–1) | Nano model |
| `answer_relevance.py` | Reverse-question generation → embedding cosine similarity (0–1) | Nano model + embedder |
| `correctness.py` | BLEU-4, ROUGE-1/2/L-F1, METEOR, BERTScore-F, semantic similarity | `nltk`, `rouge-score`, `bert-score` |
| `performance.py` | Latency span recording, token counts, `estimate_cost()` with pricing table | None |
| `pipeline.py` | Abstention rate, false abstention rate, per-layer share, partial answer rate | None (pure math) |
| `online.py` | User feedback aggregation, implicit signal processing | None |
