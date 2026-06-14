# infra/grafana/

## Table of Contents

- [What This Is](#what-this-is)
- [Loading Dashboards](#loading-dashboards)

---

## What This Is

Grafana dashboard configuration. Dashboard JSON files are mounted into the Grafana container at startup so the dashboards are pre-loaded without manual import.

---

## Loading Dashboards

Dashboard JSON files in `dashboards/` are automatically provisioned when using the observability Docker Compose profile:

```bash
docker compose -f docker-compose.yml -f docker-compose.observability.yml up grafana
```

To export an updated dashboard from Grafana UI: Dashboard → Share → Export → Save to file → overwrite `dashboards/rag_v2.json`.
