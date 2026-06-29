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

### All 20 gold rows (`nl2sql/evals/gold.jsonl`)

All SQL verified against the DuckDB sales fixture (`Laptop`/`Monitor`, 4 rows, `product`, `user_id`, `quantity`, `revenue`).

```jsonl
{"id": "0001", "question": "What is the total revenue per product?", "gold_sql": "SELECT product, SUM(revenue) AS total_revenue FROM sales GROUP BY product ORDER BY total_revenue DESC", "difficulty": "easy", "tags": ["aggregation", "group-by"], "expected_error": null}
{"id": "0002", "question": "How many rows are in the sales table?", "gold_sql": "SELECT COUNT(*) AS row_count FROM sales", "difficulty": "easy", "tags": ["count"], "expected_error": null}
{"id": "0003", "question": "What is the maximum quantity sold in a single transaction?", "gold_sql": "SELECT MAX(quantity) AS max_quantity FROM sales", "difficulty": "easy", "tags": ["aggregation"], "expected_error": null}
{"id": "0004", "question": "List all distinct products", "gold_sql": "SELECT DISTINCT product FROM sales ORDER BY product", "difficulty": "easy", "tags": ["distinct"], "expected_error": null}
{"id": "0005", "question": "What is the average revenue per transaction?", "gold_sql": "SELECT AVG(revenue) AS avg_revenue FROM sales", "difficulty": "easy", "tags": ["aggregation"], "expected_error": null}
{"id": "0006", "question": "Which products have total revenue above 2000?", "gold_sql": "SELECT product, SUM(revenue) AS total_revenue FROM sales GROUP BY product HAVING SUM(revenue) > 2000 ORDER BY total_revenue DESC", "difficulty": "medium", "tags": ["aggregation", "having"], "expected_error": null}
{"id": "0007", "question": "What is the total quantity sold per user?", "gold_sql": "SELECT user_id, SUM(quantity) AS total_quantity FROM sales GROUP BY user_id ORDER BY total_quantity DESC", "difficulty": "easy", "tags": ["aggregation", "group-by"], "expected_error": null}
{"id": "0008", "question": "What percentage of total revenue does each product contribute?", "gold_sql": "SELECT product, ROUND(SUM(revenue) * 100.0 / (SELECT SUM(revenue) FROM sales), 2) AS revenue_pct FROM sales GROUP BY product ORDER BY revenue_pct DESC", "difficulty": "medium", "tags": ["aggregation", "subquery", "window"], "expected_error": null}
{"id": "0009", "question": "Show total revenue and total quantity sold across all records", "gold_sql": "SELECT SUM(revenue) AS total_revenue, SUM(quantity) AS total_quantity FROM sales", "difficulty": "easy", "tags": ["aggregation", "multi-column"], "expected_error": null}
{"id": "0010", "question": "Show all sales for Laptop", "gold_sql": "SELECT * FROM sales WHERE product = 'Laptop'", "difficulty": "easy", "tags": ["filter"], "expected_error": null}
{"id": "0011", "question": "Which transactions had revenue greater than 1000 and quantity greater than 1?", "gold_sql": "SELECT * FROM sales WHERE revenue > 1000 AND quantity > 1", "difficulty": "easy", "tags": ["filter", "multi-condition"], "expected_error": null}
{"id": "0012", "question": "Which users bought both Laptop and Monitor?", "gold_sql": "SELECT DISTINCT user_id FROM sales WHERE product = 'Laptop' INTERSECT SELECT DISTINCT user_id FROM sales WHERE product = 'Monitor'", "difficulty": "medium", "tags": ["set-operation", "intersect"], "expected_error": null}
{"id": "0013", "question": "Which user spent the most in total?", "gold_sql": "SELECT user_id, SUM(revenue) AS total FROM sales GROUP BY user_id ORDER BY total DESC LIMIT 1", "difficulty": "medium", "tags": ["top-n", "aggregation"], "expected_error": null}
{"id": "0014", "question": "How many distinct products were sold by users who bought more than 3 items in total?", "gold_sql": "SELECT COUNT(DISTINCT product) AS product_count FROM sales WHERE user_id IN (SELECT user_id FROM sales GROUP BY user_id HAVING SUM(quantity) > 3)", "difficulty": "hard", "tags": ["subquery", "having", "distinct"], "expected_error": null}
{"id": "0015", "question": "What is the total revenue from users who have purchased more than one product type?", "gold_sql": "SELECT SUM(revenue) AS total_revenue FROM sales WHERE user_id IN (SELECT user_id FROM sales GROUP BY user_id HAVING COUNT(DISTINCT product) > 1)", "difficulty": "hard", "tags": ["subquery", "having"], "expected_error": null}
{"id": "0016", "question": "Show sales for Tablet", "gold_sql": "SELECT * FROM sales WHERE product = 'Tablet'", "difficulty": "easy", "tags": ["edge-case", "empty-result"], "expected_error": null}
{"id": "0017", "question": "What is the total revenue from user 3?", "gold_sql": "SELECT SUM(revenue) AS total_revenue FROM sales WHERE user_id = 3", "difficulty": "easy", "tags": ["filter", "aggregation"], "expected_error": null}
{"id": "0018", "question": "Which transaction had the lowest revenue?", "gold_sql": "SELECT * FROM sales ORDER BY revenue ASC LIMIT 1", "difficulty": "easy", "tags": ["sort", "top-n"], "expected_error": null}
{"id": "0019", "question": "Delete all Laptop records", "gold_sql": null, "difficulty": "easy", "tags": ["write-guard"], "expected_error": "readonly"}
{"id": "0020", "question": "Drop the sales table", "gold_sql": null, "difficulty": "easy", "tags": ["write-guard"], "expected_error": "readonly"}
```

Coverage summary:

| Category | IDs | Count |
|----------|-----|-------|
| Simple SELECT / single agg | 0002–0005, 0009 | 5 |
| GROUP BY / HAVING | 0001, 0006, 0007, 0008 | 4 |
| Filtering | 0010, 0011, 0017 | 3 |
| Subquery / set ops | 0012, 0013, 0014, 0015 | 4 |
| Edge cases | 0016, 0018 | 2 |
| Write-guard | 0019, 0020 | 2 |

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
