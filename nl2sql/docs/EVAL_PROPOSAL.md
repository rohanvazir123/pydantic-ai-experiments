# NL2SQL Evaluation — Proposal

## Problem

The current test suite (`test_nlp_sql_postgres_v2.py`) mocks the Pydantic AI agent and asserts on internal mechanics — cache logic, retry counting, guardrail firing. It never measures whether the system actually produces correct SQL. There is no benchmark, no regression gate, and no way to know if a prompt change made things better or worse.

---

## What we build

Three things in order:

1. **Gold dataset** — handwritten question/SQL pairs covering the DuckDB sales schema already used in tests
2. **Eval runner** — executes both generated and gold SQL against a real DuckDB fixture, computes metrics from `QueryResult` fields
3. **GEval layer** — LLM-judged schema fidelity for CI smoke checks that run without a database

---

## Primary metrics

### Execution Accuracy (EA) — the one that matters

Run the generated SQL and the gold SQL against the same DuckDB fixture. Compare result sets row-by-row (order-insensitive for non-ORDER-BY queries).

```
EA = count(result_sets_match) / count(total_samples)
```

Two queries can look completely different and both be correct. EA is the only metric that catches this.

### Exact Match (EM) — fast regression proxy

After normalising whitespace, aliases, and keyword casing, check `normalised(generated) == normalised(gold)`.

```
EM = count(normalised_match) / count(total_samples)
```

Runs in milliseconds with no database. Use in CI on every PR. Low EM with high EA is acceptable — it means the model found an equivalent but differently-phrased query.

---

## Secondary metrics (free from `QueryResult`)

No LLM judge needed — all fields already exist on the `QueryResult` dataclass.

| Metric | Source field | What it measures |
|--------|-------------|-----------------|
| Valid SQL Rate | `error is None` on attempt 1 | Syntax correctness before retries |
| Retry Rate | `attempts > 1` | How often self-correction fires |
| Mean Attempts | `mean(attempts)` | Correction loop health; target ≤ 1.3 |
| Write-Guard Rate | guardrail_fired tag | Coverage against adversarial inputs |
| Timeout Rate | `"timed out" in error` | Query complexity vs. timeout budget |

---

## Gold dataset design

File: `nl2sql/evals/gold.jsonl`

```json
{
  "id": "0001",
  "question": "What is the total revenue per product?",
  "gold_sql": "SELECT product, SUM(revenue) AS total_revenue FROM sales GROUP BY product ORDER BY total_revenue DESC",
  "difficulty": "easy",
  "tags": ["aggregation", "group-by"],
  "expected_error": null
}
```

Set `gold_sql: null` and `expected_error: "readonly"` for write-guard test cases — the pass condition is that the guardrail fires, not that a result set matches.

### Coverage targets for v1 (20 rows)

| Category | Count | Examples |
|----------|-------|---------|
| Simple SELECT | 4 | total revenue, count rows, max quantity |
| Aggregation | 5 | GROUP BY, HAVING, window functions |
| Filtering | 3 | WHERE with date range, string match |
| Subquery / multi-step | 3 | top-N per group, EXISTS |
| Edge cases | 3 | empty result, single row, all NULLs |
| Write-guard | 2 | DELETE, DROP TABLE |

---

## Runner architecture

The runner reuses the existing DuckDB fixture from `conftest.py` and calls `ConversationManager.run_query()` directly — no HTTP, no mocking.

```
nl2sql/evals/
├── gold.jsonl          # handwritten gold rows
├── run_eval.py         # runner — loads gold, runs queries, computes metrics, writes report
├── fixtures.py         # DuckDB in-memory setup (extracted from conftest.py)
└── results/
    ├── latest.md       # always overwritten
    └── YYYY-MM-DD.md   # timestamped archive
```

```python
# run_eval.py — skeleton
async def evaluate(gold_path: Path, judge_model: str | None = None) -> Report:
    conn = build_fixture()          # same sales table as conftest.py
    agent = build_agent()           # real LLM call — Ollama or OpenAI from .env
    manager = ConversationManager(conn, agent, schema_text=SCHEMA)

    results = []
    for row in load_gold(gold_path):
        qr = await manager.run_query(row["question"])
        ea = execution_match(qr, row, conn)     # run gold_sql, compare result sets
        em = exact_match(qr.sql, row["gold_sql"])
        results.append(EvalResult(row=row, qr=qr, ea=ea, em=em))

    return Report(results)
```

`ConversationManager` is stateful (it has a history). Each gold row gets a fresh instance so prior turns don't leak.

---

## GEval layer

Use GEval for two things that execution accuracy doesn't catch:

**1. Schema fidelity** — catches hallucinated column names. Runs without a database.

```python
GEval(
    name="Schema Fidelity",
    criteria=(
        "The SQL must only reference columns that exist in the schema. "
        "Using 'total_sales' when only 'revenue' exists is a failure. "
        "Using 'unit_price' when the schema has no such column is a failure."
    ),
    evaluation_params=[INPUT, ACTUAL_OUTPUT],
    threshold=0.8,
    model=ollama_judge,
)
```

**2. Semantic alignment** — catches correct-syntax-wrong-intent bugs.

```python
GEval(
    name="Semantic Alignment",
    criteria=(
        "The SQL must answer the exact question. "
        "If the question asks for revenue per product, grouping by user_id is a failure. "
        "Adding unrequested WHERE filters is a failure."
    ),
    evaluation_params=[INPUT, ACTUAL_OUTPUT, EXPECTED_OUTPUT],
    threshold=0.7,
    model=ollama_judge,
)
```

GEval runs in CI on every PR (no DB needed). EA runs nightly against the live DuckDB fixture.

---

## Target baselines (sales schema, 20 gold rows)

| Metric | Target |
|--------|--------|
| Execution Accuracy | ≥ 0.80 |
| Exact Match | ≥ 0.55 |
| Valid SQL Rate | ≥ 0.95 |
| Mean Attempts | ≤ 1.30 |
| Write-Guard Rate | 1.00 |
| GEval Schema Fidelity | ≥ 0.80 |

EM target is intentionally lower than EA — the model will often produce equivalent but differently-aliased queries.

---

## Phases

### Phase 1 — Gold dataset + offline runner (no LLM, DuckDB only)

- Write 20 gold rows covering the 6 categories above
- Build `fixtures.py` (extract DuckDB setup from `conftest.py`)
- Build `run_eval.py` with EA + EM + secondary metrics
- Wire `--gold-only` mode: skip LLM, just verify gold SQL executes and matches itself (sanity check)

### Phase 2 — Live LLM eval

- Connect runner to real `ConversationManager` with Ollama (`llama3.2:3b`)
- Establish baseline scores, commit to `evals/results/baseline.md`
- Block regression: if EA drops > 5 pp from baseline, exit non-zero

### Phase 3 — GEval + CI

- Add schema fidelity + semantic alignment GEval metrics (Ollama judge)
- Add `make eval` target to `Makefile`
- GitHub Actions: GEval on every PR (fast, no DB); full EA nightly on main

### Phase 4 — Expand gold dataset

- Add PostgreSQL gold rows (same questions, different catalog prefixes)
- Add multi-turn conversation gold rows (question 2 references question 1 result)
- Add GCS Parquet gold rows when the fixture supports it

---

## What this is NOT

- Not a replacement for the existing unit tests — those test internal mechanics (cache, retries, guardrails). This tests end-to-end SQL quality.
- Not a semantic similarity metric — cosine similarity on SQL strings is meaningless. Two queries can be character-identical and semantically different.
- Not a continuous online monitor — that requires logging every production query and a feedback mechanism. That's a Phase 5 if needed.
