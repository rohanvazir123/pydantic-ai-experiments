"""
Knowledge Graph Explorer — Streamlit UI.

Browse and query the Apache AGE knowledge graph built from ingested documents.
Supports entity search, relationship traversal, graph statistics, and custom Cypher.

Usage:
    streamlit run apps/kg/streamlit_app.py

Requirements:
    - Apache AGE container running: docker-compose up age
    - AGE_DATABASE_URL set in .env (postgresql://age_user:age_pass@localhost:5433/legal_graph)
    - Documents ingested with KG_BACKEND=age

API:
    uvicorn apps.kg.api:app --port 8002
"""

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv

from kg.age_graph_store import AgeGraphStore

load_dotenv(override=True)

st.set_page_config(
    page_title="Knowledge Graph Explorer",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached store — initialized once per server process
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Connecting to Apache AGE…")
def _get_store() -> AgeGraphStore:
    store = AgeGraphStore()
    asyncio.run(store.initialize())
    return store


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> str:
    with st.sidebar:
        st.title("🕸️ KG Explorer")
        st.caption("Browse the Apache AGE knowledge graph.")
        st.divider()

        mode = st.radio(
            "Mode",
            ["📊 Graph Stats", "🔍 Search Entities", "🔗 Related Entities", "🧪 Custom Cypher"],
            label_visibility="collapsed",
        )

        st.divider()
        with st.expander("ℹ️ About"):
            st.markdown(
                """
**Backend:** Apache AGE (PostgreSQL extension)

**Vertex labels:** Entity, Party, Contract, Jurisdiction,
LicenseClause, Clause, TerminationClause, …

**Edge types:** PARTY_TO, GOVERNED_BY_LAW, HAS_CLAUSE,
RELATED_TO, LICENSED_TO, …

**Ingest:** Documents processed via KG pipeline create
typed vertices and relationships from named entities
and clause structure.

**API:** `uvicorn apps.kg.api:app --port 8002`
            """
            )

    return mode


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def _show_stats(store: AgeGraphStore) -> None:
    st.subheader("Graph Statistics")
    with st.spinner("Fetching graph stats…"):
        stats = asyncio.run(store.get_graph_stats())

    col1, col2 = st.columns(2)
    col1.metric("Total Entities", stats.get("total_entities", 0))
    col2.metric("Total Relationships", stats.get("total_relationships", 0))

    st.divider()
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Entities by Type")
        entities_by_type = stats.get("entities_by_type", {})
        if entities_by_type:
            st.bar_chart(entities_by_type)
        else:
            st.info("No entities found.")

    with col4:
        st.subheader("Relationships by Type")
        rels_by_type = stats.get("relationships_by_type", {})
        if rels_by_type:
            st.bar_chart(rels_by_type)
        else:
            st.info("No relationships found.")


def _show_search(store: AgeGraphStore) -> None:
    st.subheader("Search Entities")

    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        query = st.text_input("Search query", placeholder="e.g. Amazon, Delaware, indemnification")
    with col2:
        entity_type = st.selectbox(
            "Entity type (optional)",
            ["", "Party", "Contract", "Jurisdiction", "Clause", "LicenseClause",
             "TerminationClause", "IndemnificationClause"],
        )
    with col3:
        limit = st.number_input("Limit", min_value=1, max_value=100, value=20)

    if st.button("Search", type="primary") and query:
        with st.spinner("Searching…"):
            results = asyncio.run(
                store.search_entities(
                    query=query,
                    entity_type=entity_type or None,
                    limit=limit,
                )
            )
        if not results:
            st.info("No entities found matching that query.")
        else:
            st.success(f"Found {len(results)} entit{'y' if len(results) == 1 else 'ies'}")
            st.dataframe(
                results,
                use_container_width=True,
                column_config={
                    "id": st.column_config.TextColumn("UUID", width="medium"),
                    "name": st.column_config.TextColumn("Name", width="large"),
                    "entity_type": st.column_config.TextColumn("Type", width="medium"),
                    "document_id": st.column_config.TextColumn("Document ID", width="medium"),
                },
            )

    st.divider()
    st.subheader("Context Lookup")
    st.caption("Retrieve entity relationships formatted as LLM-ready context.")
    context_query = st.text_input("Context query", placeholder="e.g. Google licensee")
    if st.button("Get Context") and context_query:
        with st.spinner("Building context…"):
            context = asyncio.run(store.search_as_context(query=context_query))
        st.code(context, language="markdown")


def _show_related(store: AgeGraphStore) -> None:
    st.subheader("Related Entities")
    st.caption("Enter an entity UUID to find what it connects to.")

    col1, col2 = st.columns([3, 2])
    with col1:
        entity_id = st.text_input("Entity UUID", placeholder="e.g. 3a7c1f2e-…")
    with col2:
        rel_type = st.text_input("Relationship type (optional)", placeholder="e.g. PARTY_TO")

    limit = st.slider("Max results", min_value=1, max_value=100, value=20)

    if st.button("Find Related", type="primary") and entity_id:
        with st.spinner("Traversing graph…"):
            results = asyncio.run(
                store.get_related_entities(
                    entity_id=entity_id.strip(),
                    relationship_type=rel_type.strip() or None,
                    limit=limit,
                )
            )
        if not results:
            st.info("No related entities found for that UUID.")
        else:
            st.success(f"Found {len(results)} related entities")
            st.dataframe(results, use_container_width=True)

    st.divider()
    st.subheader("Contracts by Entity")
    st.caption("Find contracts that mention a named entity.")
    entity_name = st.text_input("Entity name", placeholder="e.g. Amazon")
    if st.button("Find Contracts") and entity_name:
        with st.spinner("Searching contracts…"):
            contracts = asyncio.run(
                store.find_contracts_by_entity(entity_name=entity_name.strip())
            )
        if not contracts:
            st.info("No contracts found for that entity.")
        else:
            st.success(f"Found {len(contracts)} contract(s)")
            st.dataframe(contracts, use_container_width=True)


def _show_cypher(store: AgeGraphStore) -> None:
    st.subheader("Custom Cypher Query")
    st.caption("Read-only MATCH queries only. CREATE/MERGE/SET/DELETE are blocked.")

    default_query = (
        "MATCH (e:Party)-[r]->(c:Contract)\n"
        "RETURN e.name, type(r), c.name\n"
        "LIMIT 20"
    )
    cypher = st.text_area("Cypher query", value=default_query, height=120)

    with st.expander("📖 Query examples"):
        st.markdown(
            """
```cypher
-- All Party → Contract relationships
MATCH (p:Party)-[r]->(c:Contract)
RETURN p.name, type(r), c.name
LIMIT 20

-- Contracts governed by Delaware
MATCH (c:Contract)-[:GOVERNED_BY_LAW]->(j:Jurisdiction)
WHERE toLower(j.name) CONTAINS 'delaware'
RETURN c.name, j.name
LIMIT 20

-- Party co-occurrence count
MATCH (p:Party)-[]->(c:Contract)<-[]-(p2:Party)
WHERE p.uuid <> p2.uuid
RETURN p.name, p2.name, count(c) AS shared_contracts
ORDER BY shared_contracts DESC
LIMIT 15

-- Top clause types per contract
MATCH (c:Contract)<-[:HAS_CLAUSE]-(cl)
RETURN c.name, cl.label, count(cl) AS clause_count
ORDER BY clause_count DESC
LIMIT 20
```
        """
        )

    if st.button("Run Query", type="primary"):
        if not cypher.strip():
            st.warning("Enter a Cypher query first.")
        else:
            with st.spinner("Running Cypher…"):
                result = asyncio.run(store.run_cypher_query(cypher))
            if result.startswith("Error") or result.startswith("Cypher error"):
                st.error(result)
            else:
                st.code(result, language="text")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        store = _get_store()
    except Exception as exc:
        st.error(f"Failed to connect to Apache AGE: {exc}")
        st.info(
            "Make sure the AGE container is running (`docker-compose up age`) "
            "and AGE_DATABASE_URL is set in .env."
        )
        return

    mode = _render_sidebar()
    st.title("🕸️ Knowledge Graph Explorer")

    if mode == "📊 Graph Stats":
        _show_stats(store)
    elif mode == "🔍 Search Entities":
        _show_search(store)
    elif mode == "🔗 Related Entities":
        _show_related(store)
    elif mode == "🧪 Custom Cypher":
        _show_cypher(store)


if __name__ == "__main__":
    main()
