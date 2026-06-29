# NL2SQL — Evaluation Design

NL2SQL evaluation is different from RAG evaluation: the output is executable code, so the ground truth is unambiguous — either the query runs and returns the right rows, or it doesn't. LLM-judged metrics play a secondary role here.

---

## Two ground-truth metrics (do these first)

### 1. Execution Accuracy (EA)

Run both the generated SQL and the gold SQL against the real database. Compare result sets regardless of query form.

```
EA = count(results_match) / count(total_samples)
```

This is the primary metric. Two queries can look completely different and both be correct.

### 2. Exact Match (EM)

After normalising whitespace, case, and alias names, check whether generated SQL == gold SQL character-for-character.

```
EM = count(normalised_gen == normalised_gold) / count(total_samples)
```

EM is a proxy — useful for spotting regressions fast (no DB needed), but a low EM with high EA is fine.

---

## Gold row format

Gold rows live in `nl2sql/evals/gold.jsonl`. Each line:

```json
{
  "id": "0001",
  "question": "What is the total revenue per product?",
  "gold_sql": "SELECT product, SUM(revenue) AS total_revenue FROM sales GROUP BY product ORDER BY total_revenue DESC",
  "db": "duckdb",
  "schema": "sales",
  "difficulty": "easy",
  "tags": ["aggregation", "group-by"]
}
```

`db` is `"duckdb"` or `"postgres"`. `schema` names the fixture to spin up for execution.

### Sample rows

```jsonl
{"id": "0001", "question": "What is the total revenue per product?", "gold_sql": "SELECT product, SUM(revenue) AS total_revenue FROM sales GROUP BY product ORDER BY total_revenue DESC", "db": "duckdb", "schema": "sales", "difficulty": "easy", "tags": ["aggregation"]}
{"id": "0002", "question": "Which user spent the most?", "gold_sql": "SELECT user_id, SUM(revenue) AS total FROM sales GROUP BY user_id ORDER BY total DESC LIMIT 1", "db": "duckdb", "schema": "sales", "difficulty": "easy", "tags": ["top-n"]}
{"id": "0003", "question": "How many distinct products were sold by users who bought more than one item?", "gold_sql": "SELECT COUNT(DISTINCT product) FROM sales WHERE user_id IN (SELECT user_id FROM sales GROUP BY user_id HAVING SUM(quantity) > 1)", "db": "duckdb", "schema": "sales", "difficulty": "hard", "tags": ["subquery", "filter"]}
{"id": "0004", "question": "Show me all sales", "gold_sql": "SELECT * FROM sales LIMIT 100", "db": "duckdb", "schema": "sales", "difficulty": "easy", "tags": ["select-all"]}
{"id": "0005", "question": "Delete all laptop records", "gold_sql": null, "db": "duckdb", "schema": "sales", "difficulty": "easy", "tags": ["write-guard"], "expected_error": "readonly"}
```

Row 5 has no `gold_sql` — it tests that the write-guard fires and the query is rejected, not executed.

---

## Secondary metrics

| Metric | Formula | What it catches |
|--------|---------|----------------|
| **Valid SQL Rate** | `count(no parse error) / total` | Syntax failures before execution |
| **Retry Rate** | `count(attempts > 1) / total` | How often self-correction is needed |
| **Mean Attempts** | `mean(attempts)` | Correction loop health; should stay near 1.0 |
| **Write-Guard Rate** | `count(write blocked) / write_samples` | Guardrail coverage |
| **Timeout Rate** | `count(timed out) / total` | Query complexity vs. timeout budget |

These come free from `QueryResult` fields (`attempts`, `error`, `cached`) — no LLM judge needed.

---

## GEval — when execution isn't enough

Execution accuracy requires a running database. GEval covers cases where you can't execute but still want a quality signal, or where the SQL is technically correct but semantically wrong.

### Example: schema fidelity

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

schema_fidelity = GEval(
    name="Schema Fidelity",
    criteria=(
        "The generated SQL must only reference tables and columns that exist in the schema. "
        "Hallucinating a column name (e.g. 'total_sales' when only 'revenue' exists) is a failure. "
        "Using a valid alias for a non-existent column is also a failure."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,           # the NL question
        LLMTestCaseParams.ACTUAL_OUTPUT,   # the generated SQL
    ],
    threshold=0.8,
    model=judge,
)

case = LLMTestCase(
    input="What is the total revenue per product?",
    actual_output="SELECT product, SUM(total_sales) AS rev FROM sales GROUP BY product",
    # 'total_sales' doesn't exist — only 'revenue' does
)

schema_fidelity.measure(case)
# Fails — judge flags the hallucinated column name
```

### Example: semantic alignment

```python
semantic_alignment = GEval(
    name="Semantic Alignment",
    criteria=(
        "The generated SQL must answer the exact question asked. "
        "If the question asks for revenue per product, grouping by user_id is a failure. "
        "Adding unrequested filters (e.g. WHERE quantity > 1 when not asked) is a failure."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,  # gold SQL for reference
    ],
    threshold=0.7,
    model=judge,
)
```

Use GEval for CI smoke checks on PRs (no DB needed), then run execution accuracy nightly against a live fixture.

---

## Running evals

```bash
# Execution accuracy — needs DuckDB fixture (no services)
cd nl2sql
uv run python evals/run_eval.py --db duckdb

# GEval schema fidelity — needs Ollama
uv run python evals/run_eval.py --metrics geval --judge llama3.2:3b
```

### Target baselines (NeuralFlow sales schema)

| Metric | Target |
|--------|--------|
| Execution Accuracy | ≥ 0.80 |
| Exact Match | ≥ 0.55 |
| Valid SQL Rate | ≥ 0.95 |
| Mean Attempts | ≤ 1.3 |
| GEval Schema Fidelity | ≥ 0.80 |

---

## What's not built yet

- `nl2sql/evals/` directory and runner script
- `gold.jsonl` populated beyond the 5 sample rows above
- Schema fixtures for eval (currently only `conftest.py` has DuckDB setup)
- GEval integration with `OllamaJudge` (same pattern as `rag/v2/scripts/run_deepeval.py`)
