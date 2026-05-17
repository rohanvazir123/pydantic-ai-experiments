# RAG REST API

FastAPI server exposing the RAG agent over HTTP.

## Prerequisites

- PostgreSQL (pgvector) running and `DATABASE_URL` set in `.env`
- Ollama running (`ollama serve`) with `llama3.1:8b` and `nomic-embed-text` pulled
- Documents ingested: `python -m rag.main --ingest --documents rag/documents`

## Start the server

```bash
uvicorn rag.app.rest_api.api:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs available at `http://localhost:8000/docs`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | DB, embedding API, and LLM connectivity |
| POST | `/v1/chat` | Full agent run with tool calls + synthesis |
| POST | `/v1/chat/stream` | SSE-streamed agent response |
| POST | `/v1/retrieve` | Raw retrieval, no LLM synthesis |
| POST | `/v1/ingest` | Trigger document ingestion pipeline |

## Example requests

```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Which contracts have Amazon as a party?"}'

# Streaming chat
curl -X POST http://localhost:8000/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarise the indemnification clauses"}'

# Raw retrieval
curl -X POST http://localhost:8000/v1/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "governing law", "search_type": "hybrid", "match_count": 5}'
```
