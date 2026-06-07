# infra/grafana/dashboards/

## Table of Contents

- [What This Is](#what-this-is)
- [Dashboard](#dashboard)

---

## What This Is

Pre-built Grafana dashboard JSON files. Mounted into the Grafana container at startup via the provisioning config.

---

## Dashboard

| File | Rows |
|------|------|
| `rag_v2.json` | Retrieval Quality · Generation Quality · Answer Correctness · Latency Breakdown · Cost · Online Feedback · Storage |

The dashboard is built after Prometheus is wired — export it from Grafana UI and commit here. Until then this directory is a placeholder.
