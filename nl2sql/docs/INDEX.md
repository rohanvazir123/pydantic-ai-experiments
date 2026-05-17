# NL-to-SQL — Documentation Index

All documentation for the NL-to-SQL / NL-to-Cypher system.

---

## Documents

| File | What it covers |
|------|---------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System diagrams, data flow, prompt examples, agent orchestration patterns |
| [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) | Full pipeline design + implementation status + sample prompts + gaps & roadmap |
| [CALL_GRAPH.md](CALL_GRAPH.md) | Method-level call graphs for v1, v2, and the SQL Discovery agent |
| [FAQ.md](FAQ.md) | Q&A covering end-to-end flow, guardrails, caching, schema discovery, and NL→Cypher |
| [LOCAL_LLM_GUIDE.md](LOCAL_LLM_GUIDE.md) | Best local LLMs for SQL/Cypher, token limits, VRAM requirements, Ollama config, RunPod sizing |

---

## ARCHITECTURE.md

1. [Detailed Architecture Diagram](ARCHITECTURE.md#detailed-architecture-diagram)
2. [Overview](ARCHITECTURE.md#overview)
3. [Stack](ARCHITECTURE.md#stack)
4. [Components](ARCHITECTURE.md#components)
5. [Data Flow — NL Query](ARCHITECTURE.md#data-flow--nl-query)
6. [Sample LLM Prompts — NL→SQL](ARCHITECTURE.md#sample-llm-prompts--nlsql)
   - System prompt
   - First-attempt user prompt
   - Self-correction prompt
7. [Sample LLM Prompts — NL→Cypher](ARCHITECTURE.md#sample-llm-prompts--nlcypher)
   - System prompt
   - User prompt with graph schema context
   - Prompt vs SQL differences
8. [Agent Orchestration: Single Prompt vs Tool Calling](ARCHITECTURE.md#agent-orchestration-single-prompt-vs-tool-calling)
   - v1 — Single-prompt (`nlp_sql_postgres_v2.py`)
   - v2 — Tool-calling (`sql_discovery.py`)
   - Comparison table
9. [DuckDB ↔ PostgreSQL Bridge](ARCHITECTURE.md#duckdb--postgresql-bridge)
10. [Guardrails](ARCHITECTURE.md#guardrails)
11. [Caching](ARCHITECTURE.md#caching)
12. [Key Configuration](ARCHITECTURE.md#key-configuration-env)
13. [API Endpoints](ARCHITECTURE.md#api-endpoints)
14. [Running](ARCHITECTURE.md#running)

---

## SYSTEM_DESIGN.md

1. [Requirements](SYSTEM_DESIGN.md#1-requirements)
   - High Level
   - Low Level
2. [LLM Model](SYSTEM_DESIGN.md#2-llm-model)
3. [Pipeline Overview](SYSTEM_DESIGN.md#3-pipeline-overview)
4. [Caching Strategy](SYSTEM_DESIGN.md#4-caching-strategy)
5. [Schema Discovery Service](SYSTEM_DESIGN.md#5-schema-discovery-service)
   - Process
   - Schema Chunk Format
   - Schema Retrieval
6. [Prompt Generation Pipeline](SYSTEM_DESIGN.md#6-prompt-generation-pipeline)
   - Stage 1 — Normalization
   - Stage 2 — Context Assembly
   - Stage 3 — Output Format
   - Stage 4 — Cache Update
7. [SQL Generation Pipeline](SYSTEM_DESIGN.md#7-sql-generation-pipeline)
8. [SQL Validation Pipeline](SYSTEM_DESIGN.md#8-sql-validation-pipeline)
   - Check 1 — Static Guardrails
   - Check 2 — Schema Validation (SQLGlot)
   - Check 3 — RBAC Policy
   - Repair Loop
9. [SQL Executor Pipeline](SYSTEM_DESIGN.md#9-sql-executor-pipeline)
   - Router
   - Connection Pooling
   - Execution
   - Output Adapters
   - Observability
   - Index Feedback
10. [SQL Best Practices (prompt guardrails)](SYSTEM_DESIGN.md#10-sql-best-practices-prompt-guardrails)
11. [Graph Query Generation — NL→Cypher (Apache AGE)](SYSTEM_DESIGN.md#11-graph-query-generation--nlcypher-apache-age)
    - How it fits
    - Graph Schema Discovery
    - Prompt differences vs SQL
    - Cypher Validation
    - AGE Execution
    - SQL vs Cypher pipeline comparison
12. [Sample LLM Prompts](SYSTEM_DESIGN.md#12-sample-llm-prompts)
    - NL→SQL system prompt
    - NL→SQL first-attempt user prompt
    - NL→SQL self-correction prompt
    - NL→SQL planned `<thinking>`+`<query>` format
    - NL→Cypher system prompt
    - NL→Cypher user prompt
13. [Implementation Gaps & What Should Be Built](SYSTEM_DESIGN.md#13-implementation-gaps--what-should-be-built)
    - Gap 1 — SQLGlot schema validation
    - Gap 2 — `<thinking>` reasoning extraction
    - Gap 3 — Semantic NL cache (pgvector)
    - Gap 4 — N-candidate generation + confidence scoring
    - Gap 5 — RBAC policy check
    - Gap 6 — Query router (SQL vs Cypher)
    - Gap 7 — LLM-based NL→Cypher fallback
    - Gap 8 — Cursor-based pagination
    - Gap 9 — Structured observability
    - Priority order

---

## CALL_GRAPH.md

1. [NL-to-SQL v1 — Sync MVP](CALL_GRAPH.md#1-nl-to-sql-v1--sync-mvp)
2. [NL-to-SQL v2 — Async + Retry + Guardrails](CALL_GRAPH.md#2-nl-to-sql-v2--async--retry--guardrails)
   - Full call graph
   - Streamlit chat path
   - FastAPI path
   - Self-correction loop
   - Cache hit paths
3. [SQL Discovery Agent](CALL_GRAPH.md#3-sql-discovery-agent)
4. [Key Files](CALL_GRAPH.md#4-key-files)

---

## FAQ.md

### NLP-to-SQL System

| Q | Question |
|---|---------|
| [Q210](FAQ.md#q210) | Walk me through the end-to-end flow |
| [Q211](FAQ.md#q211) | How does schema discovery work? |
| [Q212](FAQ.md#q212) | Why DuckDB over Spark / Trino / pg_parquet? |
| [Q213](FAQ.md#q213) | How do cross-source JOINs work? |
| [Q214](FAQ.md#q214) | What are the limitations of this architecture? |
| [Q215](FAQ.md#q215) | How is the model prompted to generate correct SQL? |
| [Q216](FAQ.md#q216) | How are hallucinated table or column names handled? |
| [Q217](FAQ.md#q217) | What happens with semantically valid but semantically wrong SQL? |
| [Q218](FAQ.md#q218) | How is ambiguous natural language handled? |
| [Q219](FAQ.md#q219) | How does ConversationManager maintain context across follow-ups? |
| [Q220](FAQ.md#q220) | How is GCS authentication handled in DuckDB? |
| [Q221](FAQ.md#q221) | What did v2 improve over v1? |
| [Q222](FAQ.md#q222) | What guardrails are built into the NLP-to-SQL pipeline? |
| [Q223](FAQ.md#q223) | How does the SELECT-only guardrail work? |
| [Q224](FAQ.md#q224) | How does the result row cap work? |
| [Q225](FAQ.md#q225) | How does the query timeout work? |

### System Design

| Section | Topic |
|---------|-------|
| [Schema Discovery](FAQ.md#system-design--schema-discovery) | How tables and columns are discovered and serialised |
| [Prompt Engineering](FAQ.md#system-design--prompt-engineering) | Prompt structure, history context, output format |
| [SQL Generation](FAQ.md#system-design--sql-generation) | Generation pipeline, model choices, LLM SQL capability |
| [SQL Validation](FAQ.md#system-design--sql-validation) | Static guardrails, SQLGlot AST checks, RBAC |
| [Execution & Caching](FAQ.md#system-design--execution--caching) | DuckDB execution, NL cache, SQL hash cache |
| [NL→Cypher (Apache AGE)](FAQ.md#graph-query-generation--nlcypher-apache-age) | Graph query pipeline, `<cypher>`/`<columns>` format, AGE execution, LLM Cypher capability |
