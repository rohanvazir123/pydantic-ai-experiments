# knowledge/evaluation/

## Table of Contents

- [What This Is](#what-this-is)
- [Files](#files)
- [Running an Eval](#running-an-eval)
- [Gold Datasets](#gold-datasets)

---

## What This Is

Offline evaluation system. Runs the full pipeline (retrieval + generation + judge) against a gold dataset and computes IR metrics, generation quality metrics, and performance metrics. Regression detection blocks merges when quality drops.

For metric definitions, thresholds, and regression tolerances see `TEST_QA_REFERENCE.md`.

---

## Files

| File | Purpose |
|------|---------|
| `harness.py` | `EvaluationHarness`: top-level orchestration |
| `datasets.py` | `GoldDataset`: load/save gold samples from JSONL + DB |
| `runner.py` | Async runner: consumes from `knowledge:eval` Redis stream; inserts results |
| `reporter.py` | Metric aggregation, regression detection, Markdown CI report |
| `schemas.py` | `EvalRun`, `EvalResult`, `GoldSample` Pydantic models |
| `metrics/` | Individual metric implementations — see `metrics/README.md` |
| `data/` | JSONL gold dataset files (one per corpus, version-controlled) |

---

## Running an Eval

```bash
# Via API
curl -X POST http://localhost:8000/api/v1/evaluate/run \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"corpus_id": "default:default", "k": 5, "model_tier": "small"}'

# Via CLI (for CI — exits non-zero on regression)
python -m knowledge.evaluation.runner \
  --corpus-id default:default \
  --baseline-run-id <previous_run_id> \
  --fail-on-regression
```

---

## Gold Datasets

Stored as JSONL files in `evaluation/data/`. Each line is one `GoldSample`:

```json
{"id": "uuid", "corpus_id": "default:default", "query": "What is the PTO policy?",
 "relevant_doc_sources": ["team-handbook"], "difficulty": "easy"}
```

See `TEST_QA_REFERENCE.md §Gold Dataset Format` for the full schema and relevance rules.
