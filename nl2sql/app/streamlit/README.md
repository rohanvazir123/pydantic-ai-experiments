# NL-to-SQL Streamlit App

Chat interface for asking plain-English questions about the RAG PostgreSQL database.

## Prerequisites

- PostgreSQL running and `DATABASE_URL` set in `.env`
- Ollama running (`ollama serve`) with `llama3.1:8b` pulled

## Start the app

```bash
streamlit run nl2sql/app/streamlit/streamlit_app.py
```

Opens at `http://localhost:8501`.

## Features

- Type a question in plain English — the LLM generates a SQL `SELECT` and executes it
- Results are shown as a formatted table alongside the generated SQL
- Expand "Database schema" in the sidebar to see all available tables and columns
- Self-correcting: if the SQL fails, the LLM retries with the error (up to 3 attempts)
- Conversation history is preserved within the session

## Example questions

```
How many documents are stored?
What are the 10 most recent chunks?
List all distinct document titles
How many chunks does each document have? (top 20)
What is the average token count per chunk?
Find chunks that mention "governing law"
Which documents have the most chunks?
```
