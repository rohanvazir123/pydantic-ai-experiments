# Knowledge Graph Streamlit App

Interactive browser for the Apache AGE knowledge graph.

## Prerequisites

- Apache AGE container running: `docker-compose up age`
- `AGE_DATABASE_URL` set in `.env` (e.g. `postgresql://age_user:age_pass@localhost:5433/legal_graph`)
- Contracts ingested into the graph

## Start the app

```bash
streamlit run kg/app/streamlit/streamlit_app.py
```

Opens at `http://localhost:8501`.

## Modes

Use the sidebar radio to switch between:

| Mode | Description |
|------|-------------|
| Graph Stats | Entity and relationship counts, bar charts by type |
| Search Entities | Substring search with optional type filter; context lookup |
| Related Entities | Traverse edges from a given entity UUID; find contracts by entity name |
| Custom Cypher | Run any read-only `MATCH` query (CREATE/MERGE/SET/DELETE are blocked) |

## Example Cypher queries

```cypher
-- All Party → Contract relationships
MATCH (p:Party)-[r]->(c:Contract)
RETURN p.name, type(r), c.name LIMIT 20

-- Contracts governed by Delaware
MATCH (c:Contract)-[:GOVERNED_BY_LAW]->(j:Jurisdiction)
WHERE toLower(j.name) CONTAINS 'delaware'
RETURN c.name, j.name LIMIT 20

-- Party co-occurrence
MATCH (p:Party)-[]->(c:Contract)<-[]-(p2:Party)
WHERE p.uuid <> p2.uuid
RETURN p.name, p2.name, count(c) AS shared_contracts
ORDER BY shared_contracts DESC LIMIT 15
```
