# tests/load/

## Table of Contents

- [What This Is](#what-this-is)
- [Running Load Tests](#running-load-tests)
- [Results](#results)

---

## What This Is

Locust load tests. Validates that SLA targets from `RAGV2_DESIGN.md §System Design Constraints` are real measurements, not assumptions. Run against staging — never against production.

For the full scenario matrix and pass criteria see `TEST_QA_REFERENCE.md §6 Scale Test Plan`.

---

## Running Load Tests

```bash
cd rag/v2

# Baseline — search at 1 RPS for 5 min
locust -f tests/load/locustfile.py \
  --headless --users 1 --spawn-rate 1 --run-time 5m \
  --host https://staging.example.com \
  --csv tests/load/results/baseline-search-$(date +%F)

# Sustained peak — 5 RPS for 30 min
locust -f tests/load/locustfile.py \
  --headless --users 5 --spawn-rate 1 --run-time 30m \
  --host https://staging.example.com \
  --exit-code-on-error 1
```

---

## Results

`results/` stores committed summaries (Markdown). Raw CSVs are git-ignored (too large). After each run, commit a `results/baseline-{date}.md` with the P50/P95/P99 table and error rate.
