# NL2SQL Pipeline — Text Diagrams

## Table of Contents

- [End-to-End Pipeline](#end-to-end-pipeline)
- [Query Normalization Detail](#query-normalization-detail)
- [Schema Retrieval Detail](#schema-retrieval-detail)
- [SQL Validation Detail](#sql-validation-detail)
- [Multi-Turn Context Flow](#multi-turn-context-flow)
- [Ambiguity Detection Gate](#ambiguity-detection-gate)

---

## End-to-End Pipeline

```
User Query (raw text)
        │
        ▼
┌───────────────────────────────────┐
│         QUERY NORMALIZATION       │
│  lowercase · abbreviation expand  │
│  synonym map · typo correct       │
│  pronoun resolve (multi-turn)     │
└─────────────────┬─────────────────┘
                  │
                  ▼
┌───────────────────────────────────┐
│     INTENT & SCOPE DETECTION      │  ──── CONVERSATIONAL ──→  NL response
│  ANALYTICAL / OOS / AMBIGUOUS     │  ──── OUT_OF_SCOPE   ──→  "Can't answer"
│  Ambiguity score per dimension:   │  ──── AMBIGUOUS      ──→  Clarify first
│  metric · time · region · product │
└─────────────────┬─────────────────┘
                  │ ANALYTICAL + clear
                  ▼
┌───────────────────────────────────┐
│           CACHE LOOKUP            │  ──── HIT ──→  Return cached SQL (< 50ms)
│  key = normalize(query)           │
│       + schema_version_hash       │
│       + auth_scope_hash           │
└─────────────────┬─────────────────┘
                  │ MISS
                  ▼
┌───────────────────────────────────┐
│         SCHEMA RETRIEVAL          │
│  1. Embed normalized query        │
│  2. ANN search  (top-40)          │  ← enriched embeddings:
│  3. Cross-encoder rerank (top-15) │    table desc · column desc
│  4. Join graph expansion          │    glossary · join paths
│  5. Column-level filtering        │    sample values · synonyms
└─────────────────┬─────────────────┘
                  │
                  ▼
┌───────────────────────────────────┐
│         SQL GENERATION            │
│  Input:  system prompt            │
│        + enriched schema (top-15) │  ← token budget: ~4,000–5,000 tokens
│        + 3 few-shot examples      │  ← dynamically selected by similarity
│        + normalized query         │
│  Output: raw SQL string           │
└─────────────────┬─────────────────┘
                  │
                  ▼
┌───────────────────────────────────┐
│         SQL VALIDATION            │
│  Pass 1: parse (AST) — < 1ms     │  ──── FAIL ──→  Retry (max 2×)
│  Pass 2: schema check — < 5ms    │  hallucinated table/col? → re-retrieve
│  Pass 3: security check — < 5ms  │  DML / injection? → reject
│  Pass 4: EXPLAIN cost — 50–500ms │  too expensive? → warn user
└─────────────────┬─────────────────┘
                  │ valid
                  ▼
┌───────────────────────────────────┐
│         SQL EXECUTION             │
│  Read-only service account        │
│  Row-level security enforced      │
│  Timeout: configurable (30–120s)  │
│  Result: rows / error / timeout   │
└─────────────────┬─────────────────┘
                  │
                  ▼
┌───────────────────────────────────┐
│    RESULT FORMATTING + CITATION   │
│  Confidence score (0–1)           │
│  NL explanation (optional)        │
│  Assumption annotations           │
│  Cache write (SQL + schema hash)  │
└─────────────────┬─────────────────┘
                  │
                  ▼
               User
```

---

## Query Normalization Detail

```
Raw query: "What's the YTD ARR by Reg?"
                  │
                  ▼
    ┌─────────────────────────┐
    │  1. Lowercase + strip   │  →  "what's the ytd arr by reg?"
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │  2. Contraction expand  │  →  "what is the ytd arr by reg?"
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │  3. Punctuation strip   │  →  "what is the ytd arr by reg"
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │  4. Typo correction     │  →  no changes (all words valid)
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │  5. Abbreviation expand │  →  "what is the year to date
    │  (tenant dict)          │      annual recurring revenue by reg"
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │  6. Synonym normalise   │  →  "what is the year to date
    │  (tenant dict)          │      annual recurring revenue by region"
    └────────────┬────────────┘
                 │
                 ▼
Normalized: "what is the year to date annual recurring revenue by region"
Cache key:  hash("what is the year to date annual recurring revenue by region"
                  + schema_v14 + auth_scope_abc)
```

---

## Schema Retrieval Detail

```
Normalized query embedding
           │
           ▼
  ┌────────────────────────────────────────┐
  │   VECTOR STORE  (enriched schema)      │
  │                                        │
  │   Table embeddings built from:         │
  │   ┌──────────────────────────────┐     │
  │   │ table name (expanded)        │     │
  │   │ table description (LLM-gen)  │     │
  │   │ column names + descriptions  │     │
  │   │ sample values (non-PII)      │     │
  │   │ business glossary synonyms   │     │
  │   │ historical query patterns    │     │
  │   │ join relationship summary    │     │
  │   └──────────────────────────────┘     │
  │                                        │
  │   ANN search → top-40 candidates       │
  └────────────────────┬───────────────────┘
                       │
                       ▼
  ┌────────────────────────────────────────┐
  │   CROSS-ENCODER RERANKER               │
  │   Input: (query, table_description)   │
  │   Output: relevance score 0–1          │
  │   top-40 → top-15                      │
  │   +5–15pp precision vs ANN alone       │
  └────────────────────┬───────────────────┘
                       │
                       ▼
  ┌────────────────────────────────────────┐
  │   JOIN GRAPH EXPANSION                 │
  │   For each top-15 table:               │
  │     traverse FK graph 1 hop out        │
  │     score bridge tables by             │
  │     historical co-occurrence           │
  │   Surfaces semantically invisible      │
  │   bridge / junction tables             │
  └────────────────────┬───────────────────┘
                       │
                       ▼
  ┌────────────────────────────────────────┐
  │   COLUMN-LEVEL FILTERING               │
  │   Top 2 tables: full DDL               │
  │   Tables 3–8:   column names + types   │
  │   Tables 9–15:  name + description     │
  │   → 40–60% token reduction             │
  └────────────────────┬───────────────────┘
                       │
                       ▼
           Enriched schema context
           (ready for LLM prompt)
```

---

## SQL Validation Detail

```
Generated SQL string
        │
        ▼
┌────────────────────────────────────┐
│  PASS 1: SYNTAX PARSE (< 1ms)      │
│  sqlglot.parse(sql, dialect=...)   │
│  → AST or ParseError               │
└───────────────┬────────────────────┘
                │ valid AST
                ▼
┌────────────────────────────────────┐
│  PASS 2: SCHEMA VALIDATION (< 5ms) │
│  Walk AST — every table + column   │
│  referenced must exist in the      │
│  retrieved schema context          │
│  Unknown name → HALLUCINATION flag │
└───────────────┬────────────────────┘
                │ no hallucinations
                ▼
┌────────────────────────────────────┐
│  PASS 3: SECURITY CHECK (< 5ms)    │
│  Statement type == SELECT only     │
│  No DML: INSERT / UPDATE / DELETE  │
│  No DDL: CREATE / DROP / ALTER     │
│  No comment sequences: -- or /*    │
│  No UNION to unauthorised tables   │
└───────────────┬────────────────────┘
                │ secure
                ▼
┌────────────────────────────────────┐
│  PASS 4: COST CHECK (50–500ms)     │
│  EXPLAIN plan → estimated rows     │
│  Large table with no WHERE? → warn │
│  Cartesian product? → reject       │
│  Estimated cost > cap? → block     │
└───────────────┬────────────────────┘
                │ within budget
                ▼
         SQL approved for execution
```

---

## Multi-Turn Context Flow

```
Turn 1: "Show me Q3 revenue by region"
  │
  ├── Normalize → generate SQL → execute → cache
  └── Store in conversation state:
        { sql: "...", tables: ["orders"], filters: ["Q3 2024"],
          grouping: ["region"], result_schema: [...] }

Turn 2: "Now filter to EMEA only"
  │
  ├── Co-reference resolution:
  │     "now" → continue from prior query
  │     "filter to EMEA" → add WHERE region = 'EMEA'
  │
  ├── Reconstruct full intent:
  │     "Show Q3 revenue by region WHERE region = 'EMEA'"
  │
  ├── Regenerate SQL from reconstructed intent
  │   (do NOT patch the previous SQL — regenerate fresh)
  │
  └── Update conversation state

Turn 3: "Break it down by month"
  │
  ├── Intent: add monthly granularity to existing context
  │
  ├── Detect: "it" → prior query result
  │   "break down by month" → change grouping to month + region
  │
  ├── Reconstructed: "Show Q3 revenue by region and month WHERE region = 'EMEA'"
  │
  └── Semantic similarity check:
        if new query similarity to accumulated context < 0.4
          → TOPIC SHIFT DETECTED → reset context
          → surface: "Starting a new question — is that right?"
```

---

## Ambiguity Detection Gate

```
Normalized analytical query
           │
           ▼
  Score each dimension (0.0 = clear, 1.0 = unresolved):

  ┌──────────────┬──────────────────────────────────────────┐
  │ METRIC       │ "revenue"→ multiple cols: 0.9 × 2.0 = 1.8│
  │              │ "net_revenue" → one col:  0.0 × 2.0 = 0.0│
  ├──────────────┼──────────────────────────────────────────┤
  │ TIME PERIOD  │ no time expr:       0.7 × 1.5 = 1.05     │
  │              │ "Q3 2024":          0.0 × 1.5 = 0.0      │
  │              │ "last year" fiscal: 0.8 × 1.5 = 1.2      │
  ├──────────────┼──────────────────────────────────────────┤
  │ REGION       │ not mentioned, schema has region: 0.5×1.2│
  │              │ "EMEA" → exact match: 0.0 × 1.2 = 0.0   │
  ├──────────────┼──────────────────────────────────────────┤
  │ PRODUCT      │ not mentioned: 0.4 × 0.8 = 0.32          │
  ├──────────────┼──────────────────────────────────────────┤
  │ GROUPING     │ "by region" → clear: 0.0 × 0.8 = 0.0    │
  │              │ no breakdown:        0.5 × 0.8 = 0.4     │
  ├──────────────┼──────────────────────────────────────────┤
  │ GRANULARITY  │ "monthly" → clear:   0.0 × 0.5 = 0.0    │
  │              │ not mentioned:       0.5 × 0.5 = 0.25    │
  └──────────────┴──────────────────────────────────────────┘

  Weighted total → routing decision:

  < 1.0  ─────────────────────→  Proceed with defaults
                                  Annotate assumptions in result

  1.0 – 2.0  ─────────────────→  Ask ONE clarifying question
                                  (highest-weight unresolved dim)
                                  Default + annotate the rest

  > 2.0  ─────────────────────→  Must clarify before generating
                                  Offer schema-derived options
```
