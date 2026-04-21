"""
MCP server for the RAG knowledge base.

Module: rag.mcp.server
======================

Exposes the RAG system as Model Context Protocol (MCP) tools so that
Claude Desktop, Claude Code, and any other MCP client can call the
knowledge base directly — no HTTP wiring required.

Tools
-----
search(query, user_id?, session_id?)
    Full agentic RAG run: retrieval + LLM synthesis → answer string.
    Wraps ``traced_agent_run`` — same path as POST /v1/chat.

retrieve(query, match_count?, search_type?)
    Raw hybrid retrieval without LLM synthesis → formatted chunk list.
    Useful when the caller wants source passages, not a generated answer.

ingest(documents_folder?, clean?, chunk_size?, max_tokens?)
    Trigger the document ingestion pipeline.
    Same as POST /v1/ingest — runs synchronously within the tool call.

health()
    Check DB, embedding API, and LLM API connectivity.
    Returns a status string: "ok", "degraded", or "unhealthy".

Transport
---------
Runs over stdio (required by Claude Desktop).  The process is launched
by the MCP client — it does not bind to a port.

Usage
-----
    # Run the server (Claude Desktop launches this automatically)
    python -m rag.mcp.server

    # Register with Claude Desktop
    # File: %APPDATA%\\Claude\\claude_desktop_config.json  (Windows)
    #       ~/Library/Application Support/Claude/claude_desktop_config.json  (macOS)
    {
      "mcpServers": {
        "rag": {
          "command": "python",
          "args": ["-m", "rag.mcp.server"],
          "cwd": "C:/Users/rohan/Documents/ai_agents/pydantic-ai-experiments"
        }
      }
    }

    # Register with Claude Code (project-scoped)
    # File: .mcp.json  (in the project root)
    {
      "mcpServers": {
        "rag": {
          "command": "python",
          "args": ["-m", "rag.mcp.server"]
        }
      }
    }
"""

import asyncio
import logging

import httpx
from mcp.server.fastmcp import FastMCP

from rag.agent.rag_agent import RAGState, traced_agent_run
from rag.config.settings import load_settings
from rag.ingestion.pipeline import create_pipeline
from rag.storage.vector_store.postgres import PostgresHybridStore

logger = logging.getLogger(__name__)

mcp = FastMCP("RAG Knowledge Base")


# ---------------------------------------------------------------------------
# search — full agent run (retrieval + LLM synthesis)
# ---------------------------------------------------------------------------


@mcp.tool()
async def search(
    query: str,
    user_id: str | None = None,
    session_id: str | None = None,
) -> str:
    """
    Ask the RAG knowledge base a question.

    Runs the full agentic RAG pipeline: hybrid retrieval → optional rerank
    → LLM synthesis.  Returns the generated answer as a string.

    Args:
        query:      The question to ask.
        user_id:    Optional — enables Mem0 personalisation for this user.
        session_id: Optional — groups Langfuse traces under one session.
    """
    result = await traced_agent_run(
        query=query,
        user_id=user_id,
        session_id=session_id,
    )
    return str(result.output)


# ---------------------------------------------------------------------------
# retrieve — raw retrieval without LLM synthesis
# ---------------------------------------------------------------------------


@mcp.tool()
async def retrieve(
    query: str,
    match_count: int = 5,
    search_type: str = "hybrid",
) -> str:
    """
    Retrieve relevant chunks from the knowledge base without LLM synthesis.

    Returns the raw source passages formatted as context — useful when the
    caller wants to inspect or re-rank the sources rather than get a
    generated answer.

    Args:
        query:       The search query.
        match_count: Number of chunks to return (default: 5).
        search_type: "hybrid" (default), "semantic", or "text".
    """
    state = RAGState()
    try:
        retriever = await state.get_retriever()
        context = await retriever.retrieve_as_context(
            query=query,
            match_count=match_count,
            search_type=search_type,
        )
        return context or "No results found."
    finally:
        await state.close()


# ---------------------------------------------------------------------------
# ingest — trigger the ingestion pipeline
# ---------------------------------------------------------------------------


@mcp.tool()
async def ingest(
    documents_folder: str = "rag/documents",
    clean: bool = True,
    chunk_size: int = 1000,
    max_tokens: int = 512,
) -> str:
    """
    Ingest documents from a folder into the knowledge base.

    Runs the full ingestion pipeline: document conversion → chunking →
    embedding → PostgreSQL storage.  Runs synchronously — may take a while
    for large corpora.

    Args:
        documents_folder: Path to the folder containing documents (default: rag/documents).
        clean:            Truncate existing data before ingesting (default: True).
                          Set to False for incremental ingestion.
        chunk_size:       Target chunk size in characters (default: 1000).
        max_tokens:       Maximum tokens per chunk (default: 512).
    """
    pipeline = create_pipeline(
        documents_folder=documents_folder,
        clean=clean,
        chunk_size=chunk_size,
        max_tokens=max_tokens,
    )
    try:
        await pipeline.initialize()
        results = await pipeline.ingest_documents()
    finally:
        await pipeline.close()

    total_chunks = sum(r.chunks_created for r in results)
    lines = [f"Ingested {len(results)} document(s), {total_chunks} chunk(s) total."]
    for r in results:
        status = f"  • {r.title}: {r.chunks_created} chunks"
        if r.errors:
            status += f" [{len(r.errors)} error(s): {r.errors[0]}]"
        lines.append(status)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# health — connectivity check
# ---------------------------------------------------------------------------


@mcp.tool()
async def health() -> str:
    """
    Check connectivity to the database, embedding API, and LLM API.

    Returns a status string: "ok" (all healthy), "degraded" (some down),
    or "unhealthy" (all down), followed by per-component details.
    """
    settings = load_settings()
    checks: dict[str, bool] = {"db": False, "embedding_api": False, "llm_api": False}

    # DB check
    try:
        store = PostgresHybridStore()
        await asyncio.wait_for(store.initialize(), timeout=5.0)
        await store.close()
        checks["db"] = True
    except Exception as exc:
        logger.warning("Health: DB check failed: %s", exc)

    # Embedding API check
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            base = settings.embedding_base_url.rstrip("/")
            resp = await client.get(
                f"{base}/models",
                headers={"Authorization": f"Bearer {settings.embedding_api_key}"},
            )
            checks["embedding_api"] = resp.status_code < 500
    except Exception as exc:
        logger.warning("Health: Embedding API check failed: %s", exc)

    # LLM API check
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            base = settings.llm_base_url.rstrip("/")
            resp = await client.get(
                f"{base}/models",
                headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            )
            checks["llm_api"] = resp.status_code < 500
    except Exception as exc:
        logger.warning("Health: LLM API check failed: %s", exc)

    all_ok = all(checks.values())
    any_ok = any(checks.values())
    status = "ok" if all_ok else ("degraded" if any_ok else "unhealthy")

    lines = [f"status: {status}"]
    for component, ok in checks.items():
        lines.append(f"  {component}: {'✓' if ok else '✗'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
