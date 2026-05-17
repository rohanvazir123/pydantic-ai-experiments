# NL-to-SQL REST API

FastAPI server that translates natural-language questions into SQL and executes them against PostgreSQL via DuckDB.

## Prerequisites

- PostgreSQL running and `DATABASE_URL` set in `.env`
- Ollama running (`ollama serve`) with `llama3.1:8b` pulled
- DuckDB postgres extension (installed automatically on first request)

## Start the server

```bash
uvicorn nl2sql.app.rest_api.api:app --host 0.0.0.0 --port 8001 --reload
```

Interactive docs available at `http://localhost:8001/docs`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | DB and LLM connectivity check |
| POST | `/v1/query` | Translate NL → SQL, execute, return results |
| GET | `/v1/history` | Recent query history for this session |
| GET | `/v1/schema` | Database schema used for SQL generation |

## Example requests

```bash
# Health check
curl http://localhost:8001/health

# Ask a question
curl -X POST http://localhost:8001/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many documents are stored?"}'

# View schema
curl http://localhost:8001/v1/schema

# Query history
curl http://localhost:8001/v1/history
```

## Notes

- Only `SELECT` queries are executed; write/DDL statements are blocked
- Self-correcting: on SQL errors the LLM is re-prompted with the error message (up to 3 attempts)
- Results are capped at 500 rows
