# Graph Viewer — Setup & Queries

Apache AGE Viewer is the graph visualisation UI for the `legal_graph`. It renders
Cypher query results as interactive node-edge diagrams.

All four logical graphs (hierarchy, semantic, lineage, risk) live in the **same**
`legal_graph` AGE graph — they are distinguished by vertex labels and edge types,
not by separate graph objects.

---

## 1. Start the stack

```powershell
# Start AGE + AGE Viewer together
docker compose up -d age age-viewer

# Or start everything (includes pgvector for the RAG store)
docker compose up -d
```

AGE Viewer is now at **http://localhost:3001**

AGE (PostgreSQL) is at **localhost:5433**

---

## 2. Connect AGE Viewer to the database

Open **http://localhost:3001**, click **Connect**, and enter:

| Field      | Value           |
|------------|-----------------|
| Host       | `age`           |
| Port       | `5432`          |
| Database   | `legal_graph`   |
| User       | `age_user`      |
| Password   | `age_pass`      |
| Graph Path | `legal_graph`   |

> **Why `age` and `5432`, not `localhost:5433`?**
> The viewer runs inside Docker and connects to the `age` container over the
> internal Compose network. Inside Docker, the AGE container listens on `5432`.
> Port `5433` is only exposed to your host machine (for DBeaver, psql, etc.).
>
> **Graph Path** is the AGE graph name, set separately from the PostgreSQL
> database name. Both happen to be `legal_graph` in this project.

---

## 3. Viewing the four graphs

Paste each Cypher block into the AGE Viewer query editor and click **Run**.
Always `RETURN` node and edge variables (not scalar properties like `.name`) so
the viewer can render them as a diagram.

---

### 3.1 Document Hierarchy Graph

Shows the structural decomposition of each contract: Contract → Section → Clause → Chunk.

```cypher
MATCH (c:Contract)-[:HAS_SECTION]->(s:Section)-[:HAS_CLAUSE]->(cl:Clause)
RETURN c, s, cl
LIMIT 80
```

Drill into one contract (replace with a real contract name):

```cypher
MATCH (c:Contract)-[:HAS_SECTION]->(s:Section)
WHERE c.name CONTAINS 'LIGHTBRIDGE'
OPTIONAL MATCH (s)-[:HAS_CLAUSE]->(cl:Clause)
RETURN c, s, cl
LIMIT 100
```

---

### 3.2 Legal Semantic Graph

Shows entity relationships within and across contracts: parties, jurisdictions,
indemnification chains, liability clauses, payment terms, confidentiality, and
obligations.

All relationships from a contract:

```cypher
MATCH (c:Contract)-[r]->(n)
RETURN c, r, n
LIMIT 60
```

Party-centric view — who signs what, who indemnifies whom:

```cypher
MATCH (p:Party)-[r:SIGNED_BY|INDEMNIFIES|DISCLOSES_TO]-(n)
RETURN p, r, n
LIMIT 50
```

Full semantic neighbourhood of one contract:

```cypher
MATCH (c:Contract)-[r]-(n)
WHERE c.name CONTAINS 'Strategic Alliance'
RETURN c, r, n
LIMIT 80
```

Key clause types:

```cypher
MATCH (c:Contract)-[r]->(clause)
WHERE clause:LiabilityClause
   OR clause:TerminationClause
   OR clause:IndemnityClause
   OR clause:ConfidentialityClause
   OR clause:GoverningLawClause
RETURN c, r, clause
LIMIT 60
```

---

### 3.3 Cross-Contract Lineage Graph

Shows how contracts relate across time: amendments, supersessions, replacements,
incorporated-by-reference documents, and attachments.

All lineage edges:

```cypher
MATCH (c1:Contract)-[r:AMENDS|SUPERCEDES|REPLACES|REFERENCES|INCORPORATES_BY_REFERENCE|ATTACHES]->(c2)
RETURN c1, r, c2
LIMIT 60
```

Amendment chains only:

```cypher
MATCH path = (c1:Contract)-[:AMENDS*1..3]->(c2:Contract)
RETURN path
LIMIT 40
```

Supersession chains:

```cypher
MATCH (c1:Contract)-[r:SUPERCEDES]->(c2:Contract)
RETURN c1, r, c2
LIMIT 50
```

Incorporated and attached documents:

```cypher
MATCH (c:Contract)-[r:INCORPORATES_BY_REFERENCE|ATTACHES]->(ref:ReferenceDocument)
RETURN c, r, ref
LIMIT 50
```

---

### 3.4 Risk Dependency Graph

Shows risk factors, their severity, and which parties they affect. Also shows
risk-causes-risk chains.

All risks and the parties they affect:

```cypher
MATCH (r:Risk)-[rel:INCREASES_RISK_FOR]->(p:Party)
RETURN r, rel, p
LIMIT 60
```

Risk cascades (what causes what):

```cypher
MATCH (r1:Risk)-[rel:CAUSES]->(r2:Risk)
RETURN r1, rel, r2
LIMIT 50
```

High-severity risks only:

```cypher
MATCH (r:Risk)-[rel:INCREASES_RISK_FOR]->(p:Party)
WHERE r.severity = 'HIGH'
RETURN r, rel, p
LIMIT 40
```

Full risk subgraph (risks + parties + causation):

```cypher
MATCH (r:Risk)-[rel]->(n)
RETURN r, rel, n
LIMIT 60
```

---

## 4. Tips

- **Visual rendering requires node/edge objects.** `RETURN c, r, n` renders a
  graph. `RETURN c.name, n.name` renders a table — useful for debugging but
  produces no diagram.

- **Start small with LIMIT.** Rendering 500+ nodes is slow. Use `LIMIT 50–100`
  for exploration, then filter by name once you know what to look for.

- **Click a node** in the viewer to see all its properties in the side panel.

- **Filter by label** to isolate one graph type:
  ```cypher
  -- Show only Party and Jurisdiction nodes
  MATCH (p:Party)-[r:GOVERNED_BY]->(j:Jurisdiction)
  RETURN p, r, j LIMIT 50
  ```

- **Graph must be populated first.** If you see "No results", run the
  ingestion pipeline:
  ```powershell
  python -m kg.cuad_kg_ingest        # CUAD annotation ingest (fast, no LLM)
  # or
  python -m kg.extraction_pipeline   # LLM extraction (slow, higher quality)
  ```

---

## 5. Troubleshooting

| Symptom | Fix |
|---------|-----|
| AGE Viewer shows "Connection refused" | `docker compose up -d age` — the AGE container must be healthy first |
| "No results" on all queries | Graph not yet populated — run the ingestion pipeline |
| Node labels show as `vertex` not `Party` | Use `RETURN p, r, n` not `RETURN p.properties, ...`; ensure AGE Viewer version ≥ 0.1.0 |
| Can't reach `host.docker.internal` on Linux | Use `--network host` or replace with the host's LAN IP |
| Port 3001 already in use | Change the host port in `docker-compose.yml`: `"3002:3001"` |
