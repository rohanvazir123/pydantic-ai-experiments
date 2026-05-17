# RAG CLI

No standalone CLI scripts here — the RAG pipeline CLI lives at the module root.

## Available CLI commands

```bash
# Validate configuration
python -m rag.main --validate

# Ingest documents
python -m rag.main --ingest --documents rag/documents

# Ingest incrementally (keep existing data)
python -m rag.main --ingest --documents rag/documents --no-clean

# Verbose output
python -m rag.main --ingest --documents rag/documents --verbose
```
