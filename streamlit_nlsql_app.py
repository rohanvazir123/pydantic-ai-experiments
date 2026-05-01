"""
Natural-language SQL explorer — Streamlit app.

Wraps ConversationManager from nlp2sql/nlp_sql_postgres_v2.py.
Connects to the RAG PostgreSQL database (DATABASE_URL from .env) via DuckDB's
postgres scanner, generates a schema, then lets users ask questions in plain
English.  Generated SQL and results are displayed alongside the answer.

Usage:
    streamlit run streamlit_nlsql_app.py
"""

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import streamlit as st
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from nlp2sql.nlp_sql_postgres_v2 import ConversationManager, QueryResult
from rag.config.settings import load_settings

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NL-to-SQL Explorer",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

_SYSTEM_PROMPT = (
    "You are a SQL expert. Given a database schema and a natural-language question, "
    "return ONLY a valid SQL SELECT statement with no explanation, no markdown fences. "
    "Use table prefixes shown in the schema (e.g. rag_db.main.chunks). "
    "Always include a LIMIT clause (max 500 rows)."
)


# ---------------------------------------------------------------------------
# Cached resources — created once per server process
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Connecting to database…")
def _build_manager() -> tuple[ConversationManager, str]:
    """
    Set up DuckDB, attach the RAG PostgreSQL database, generate schema,
    build a Pydantic AI agent, and return a ConversationManager + schema text.
    """
    settings = load_settings()

    conn = duckdb.connect(database=":memory:")
    conn.execute("INSTALL postgres; LOAD postgres;")
    conn.execute(
        f"ATTACH '{settings.database_url}' AS rag_db (TYPE postgres, READ_ONLY)"
    )

    # Generate schema from attached DB
    lines: list[str] = ["=== rag_db tables (prefix: rag_db.main.<table>) ==="]
    tables = conn.execute(
        "SELECT table_name FROM rag_db.information_schema.tables "
        "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' "
        "ORDER BY table_name"
    ).fetchall()
    for (tbl,) in tables:
        cols = conn.execute(
            f"SELECT column_name, data_type "
            f"FROM rag_db.information_schema.columns "
            f"WHERE table_schema = 'public' AND table_name = '{tbl}' "
            f"ORDER BY ordinal_position"
        ).fetchall()
        lines.append(f"Table: rag_db.main.{tbl}")
        for col_name, col_type in cols:
            lines.append(f"  - {col_name} ({col_type})")
        lines.append("")
    schema_text = "\n".join(lines).strip()

    # Build agent using the same LLM settings as the RAG agent
    llm = OpenAIModel(
        settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
    )
    pydantic_agent = Agent(model=llm, result_type=str, system_prompt=_SYSTEM_PROMPT)

    manager = ConversationManager(
        conn=conn,
        agent=pydantic_agent,
        schema_text=schema_text,
        cache_size=30,
        max_retries=3,
        max_result_rows=500,
        query_timeout=30.0,
    )
    return manager, schema_text


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_state() -> None:
    if "nl_messages" not in st.session_state:
        st.session_state.nl_messages: list[dict] = []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _render_sidebar(schema_text: str) -> None:
    with st.sidebar:
        st.title("🗄️ NL-to-SQL Explorer")
        st.caption("Ask questions about the RAG database in plain English.")
        st.divider()

        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.nl_messages = []
            st.rerun()

        st.divider()
        with st.expander("📋 Database schema"):
            st.code(schema_text, language="sql")

        st.divider()
        with st.expander("ℹ️ Example queries"):
            st.markdown(
                """
- How many documents are stored?
- What are the 10 most recent chunks?
- List all distinct document titles
- How many chunks does each document have? (top 20)
- What is the average token count per chunk?
- Find chunks that mention "governing law"
- Which documents have the most chunks?
            """
            )


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------
def _render_result(qr: QueryResult) -> str:
    """Format a QueryResult as markdown for display in the chat."""
    if not qr.success:
        return f"**Error after {qr.attempts} attempt(s):** {qr.error}"

    lines: list[str] = []
    if qr.cached:
        lines.append("*(cached)*")

    # Generated SQL
    lines.append(f"```sql\n{qr.sql}\n```")

    if not qr.rows:
        lines.append("*No rows returned.*")
        return "\n".join(lines)

    # Results as markdown table
    header = " | ".join(qr.columns)
    sep = " | ".join("---" for _ in qr.columns)
    lines.append(f"| {header} |")
    lines.append(f"| {sep} |")
    display_rows = qr.rows[:50]
    for row in display_rows:
        cells = " | ".join(str(v) if v is not None else "" for v in row)
        lines.append(f"| {cells} |")
    if len(qr.rows) > 50:
        lines.append(f"*… {len(qr.rows) - 50} more rows not shown.*")
    lines.append(f"\n*{len(qr.rows)} row(s) — {qr.attempts} LLM attempt(s)*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    _init_state()

    try:
        manager, schema_text = _build_manager()
    except Exception as exc:
        st.error(f"Failed to connect to database: {exc}")
        st.info("Make sure DATABASE_URL is set in .env and the database is reachable.")
        return

    _render_sidebar(schema_text)

    st.title("💬 Ask your database anything")

    # Display history
    for msg in st.session_state.nl_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("e.g. How many documents are stored?"):
        st.session_state.nl_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating SQL…"):
                qr: QueryResult = asyncio.run(manager.run_query(prompt))
            response = _render_result(qr)
            st.markdown(response)

        st.session_state.nl_messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
