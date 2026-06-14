# infra/

## Table of Contents

- [What This Is](#what-this-is)
- [Sub-directories](#sub-directories)

---

## What This Is

Infrastructure configuration files for local development and observability. These are mounted into Docker containers — they are not Python code.

---

## Sub-directories

| Directory | Contents |
|-----------|---------|
| `nginx/` | Nginx reverse proxy config (TLS termination, SSE route settings) |
| `grafana/dashboards/` | Pre-built Grafana dashboard JSON (7-row RAG v2 dashboard) |
