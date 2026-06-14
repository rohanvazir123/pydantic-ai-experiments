# tests/load/results/

## Table of Contents

- [What This Is](#what-this-is)
- [File Naming](#file-naming)

---

## What This Is

Committed load test result summaries. Raw Locust CSV files are git-ignored (too large). Only Markdown summary files are committed here.

---

## File Naming

| Pattern | Contents |
|---------|---------|
| `baseline-{date}.md` | P50/P95/P99 + error rate for the standard baseline matrix |
| `grafana-{date}.png` | Screenshot of the Grafana dashboard during the run |
| `chaos-results-{date}.md` | Chaos scenario outcomes and acceptance criteria verdicts |
