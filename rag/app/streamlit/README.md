# RAG Streamlit Apps

Two Streamlit interfaces for the RAG pipeline.

## Prerequisites

- PostgreSQL (pgvector) running and `DATABASE_URL` set in `.env`
- Ollama running (`ollama serve`) with `llama3.1:8b` and `nomic-embed-text` pulled
- Documents ingested: `python -m rag.main --ingest --documents rag/documents`

## Apps

### Legal Contract Assistant (`streamlit_app.py`)

Chat interface that streams tool calls and LLM responses in real time.

```bash
streamlit run rag/app/streamlit/streamlit_app.py
```

Opens at `http://localhost:8501`. Use the sidebar to see current model config and clear the conversation.

**Example queries:**
- Which contracts have Amazon as a party?
- Find all contracts governed by Delaware law
- Summarise the indemnification clauses

### Memory Chat (`streamlit_mem0_app.py`)

Simple chat app that persists user memory across sessions using Mem0 + pgvector.

```bash
streamlit run rag/app/streamlit/streamlit_mem0_app.py
```

Opens at `http://localhost:8501`. Set a User ID in the sidebar to scope memories per user. Use "Show My Memories" to inspect what has been stored.
