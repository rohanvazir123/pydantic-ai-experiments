# Knowledge Graph REST API

FastAPI server exposing the Apache AGE knowledge graph over HTTP.

## Prerequisites

- Apache AGE container running: `docker-compose up age`
- `AGE_DATABASE_URL` set in `.env` (e.g. `postgresql://age_user:age_pass@localhost:5433/legal_graph`)
- Contracts ingested into the graph (run `build_cuad_kg()` or the extraction pipeline)

## Start the server

```bash
uvicorn kg.app.rest_api.api:app --host 0.0.0.0 --port 8002 --reload
```

Interactive docs available at `http://localhost:8002/docs`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | AGE container connectivity |
| GET | `/v1/stats` | Entity and relationship counts by type |
| POST | `/v1/search` | Case-insensitive entity name search |
| POST | `/v1/context` | LLM-ready relationship context for a query |
| POST | `/v1/related` | Entities connected to a given UUID |
| POST | `/v1/contracts` | Contracts mentioning a named entity |
| POST | `/v1/cypher` | Execute a read-only Cypher MATCH query |
| POST | `/v1/nl_query` | Natural-language → Cypher → results |

## Example requests

```bash
# Stats
curl http://localhost:8002/v1/stats

# Search entities
curl -X POST http://localhost:8002/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Amazon", "entity_type": "Party"}'

# Natural-language query
curl -X POST http://localhost:8002/v1/nl_query \
  -H "Content-Type: application/json" \
  -d '{"question": "Which contracts is Amazon a party to?"}'

# Custom Cypher
curl -X POST http://localhost:8002/v1/cypher \
  -H "Content-Type: application/json" \
  -d '{"cypher": "MATCH (p:Party)-[:PARTY_TO]->(c:Contract) RETURN p.name, c.name LIMIT 10"}'
```
