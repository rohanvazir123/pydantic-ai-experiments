# Spider2-lite — Setup, Integration, and Analysis

Complete guide to using Spider2-lite as an evaluation framework for your NL2SQL pipeline.

Repo: https://github.com/xlang-ai/Spider2/tree/main/spider2-lite

## Table of Contents

- [What Spider2-lite Tests That Internal Evals Don't](#what-spider2-lite-tests-that-internal-evals-dont)
- [One-Time Setup](#one-time-setup)
- [Wiring Your Pipeline to Generate Predictions](#wiring-your-pipeline-to-generate-predictions)
- [Handling External Knowledge](#handling-external-knowledge)
- [Dialect-Specific Generation](#dialect-specific-generation)
- [Running Evaluation and Reading Results](#running-evaluation-and-reading-results)
- [Segmenting Failure Analysis](#segmenting-failure-analysis)
- [Execution-Based Evaluation — What It Catches and Misses](#execution-based-evaluation--what-it-catches-and-misses)
- [Using Spider2-lite as a Regression Gate in CI](#using-spider2-lite-as-a-regression-gate-in-ci)

---

## What Spider2-lite Tests That Internal Evals Don't

Most teams build internal eval sets from their own schema and their own users' queries. These are essential but systematically biased: they cover the schemas the team is familiar with, the query patterns the team thought to include, and the dialects the current system already handles.

Spider2-lite adds:

- **Real-world query complexity:** multi-step aggregations, window functions, CTEs, nested subqueries, semi-structured data (BigQuery ARRAY/STRUCT), Snowflake-specific functions
- **Three dialects in one benchmark:** BigQuery Standard SQL, Snowflake SQL, SQLite — forces dialect-aware generation or exposes dialect blind spots
- **External knowledge dependency:** 107 of 547 instances require reading a markdown doc to understand the schema, simulating real-world corpora where column names alone are not enough
- **Verified execution results:** gold answers are actual query outputs, not hand-written SQL — removes exact-match bias

Running Spider2-lite tells you whether your pipeline generalises beyond your internal schema. A system that scores 85% internally and 40% on Spider2-lite has a generalisation problem.

---

## One-Time Setup

### Repository

```bash
git clone https://github.com/xlang-ai/Spider2.git
cd Spider2/spider2-lite

# Install evaluation dependencies
pip install pandas google-cloud-bigquery snowflake-connector-python tqdm
```

### SQLite databases (160 instances, no credentials)

```bash
# Download from Google Drive link in the Spider2 README
# Unzip and place all .sqlite files here:
mkdir -p resource/databases/spider2-localdb
# e.g.: resource/databases/spider2-localdb/formula_1.sqlite
#        resource/databases/spider2-localdb/california_schools.sqlite
```

### BigQuery (180 instances)

```bash
# 1. Create a GCP project and enable the BigQuery API
# 2. Create a service account with BigQuery Data Viewer + BigQuery Job User roles
# 3. Download the JSON key file
cp ~/Downloads/my-project-key.json evaluation_suite/bigquery_credential.json

# Test:
export GOOGLE_APPLICATION_CREDENTIALS=evaluation_suite/bigquery_credential.json
python -c "from google.cloud import bigquery; c = bigquery.Client(); print('OK')"
```

Note: BigQuery queries cost money. Running all 180 BigQuery instances in Spider2-lite processes roughly 5–15 GB depending on your predictions — at BigQuery on-demand pricing ($6.25/TB), this is $0.03–$0.09 per full evaluation run. Negligible, but track it.

### Snowflake (207 instances)

Fill out the [Spider2 Snowflake Access form](https://docs.google.com/forms/d/e/1FAIpQLScbVIYcBkADVr-NcYm9fLMhlxR7zBAzg-jaew1VNRj6B8yD3Q/viewform). They email credentials within ~24 hours.

```json
// evaluation_suite/snowflake_credential.json
{
  "account": "your_account_id",
  "user": "your_username",
  "password": "your_password",
  "warehouse": "COMPUTE_WH",
  "role": "SPIDER2_ROLE"
}
```

---

## Wiring Your Pipeline to Generate Predictions

The prediction format is simple: one `.sql` file per instance, named by `instance_id`.

```python
# generate_predictions.py
import json
from pathlib import Path
from your_nl2sql_pipeline import NL2SQLPipeline

pipeline = NL2SQLPipeline(
    schema_retriever=...,
    sql_generator=...,
    dialect_router=...,
)

# Load all instances
instances = []
with open("spider2-lite/spider2-lite.jsonl") as f:
    for line in f:
        instances.append(json.loads(line))

# Load external knowledge docs
EXTERNAL_KNOWLEDGE_DIR = Path("spider2-lite/resource/documentation")

output_dir = Path("my_predictions")
output_dir.mkdir(exist_ok=True)

for instance in instances:
    instance_id  = instance["instance_id"]
    db           = instance["db"]
    question     = instance["question"]
    ext_knowledge = instance.get("external_knowledge")  # filename or None

    # Skip if already generated (for resumable runs)
    out_file = output_dir / f"{instance_id}.sql"
    if out_file.exists():
        continue

    # Load external knowledge if present
    context_docs = []
    if ext_knowledge:
        doc_path = EXTERNAL_KNOWLEDGE_DIR / ext_knowledge
        if doc_path.exists():
            context_docs.append(doc_path.read_text())

    # Detect dialect from instance_id prefix
    if instance_id.startswith("bq"):
        dialect = "bigquery"
    elif instance_id.startswith("sf"):
        dialect = "snowflake"
    else:
        dialect = "sqlite"

    try:
        sql = pipeline.generate(
            question=question,
            db_name=db,
            dialect=dialect,
            external_knowledge=context_docs,
        )
    except Exception as e:
        print(f"[{instance_id}] ERROR: {e}")
        sql = "SELECT 1"  # placeholder — will score 0, but doesn't crash the eval

    out_file.write_text(sql)
    print(f"[{instance_id}] done")
```

**Parallelise generation** — 547 instances at 2–5 seconds each is 18–45 minutes sequentially. With 10 concurrent workers: 2–5 minutes.

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(generate_one, instance) for instance in instances]
    for f in futures:
        f.result()
```

**Resume-safe:** the `if out_file.exists(): continue` guard makes the script resumable. If it crashes at instance 300, re-run and it picks up from 301.

---

## Handling External Knowledge

107 instances include an `external_knowledge` field pointing to a markdown file in `spider2-lite/resource/documentation/`. These files contain:

- Table schema descriptions in natural language
- Column value explanations (what the codes mean)
- BigQuery-specific function documentation
- Domain context the model cannot infer from table names alone

**Example external knowledge file (excerpt):**
```markdown
# ga4_obfuscated_sample_ecommerce.events

This table contains Google Analytics 4 event data for an obfuscated e-commerce website.

## Key Fields
- `user_pseudo_id`: Pseudo-anonymous user identifier. Unique per device.
- `event_timestamp`: Microseconds since epoch (NOT milliseconds). Divide by 1,000,000 for seconds.
- `engagement_time_msec`: Milliseconds of user engagement. A value of 0 means no engagement.

## Important Notes
- The table is partitioned by `event_date` (format: YYYYMMDD as STRING, not DATE).
- Filter on `event_date` for performance: `WHERE event_date BETWEEN '20210101' AND '20210107'`
```

**Naive approach (dump the whole doc into the prompt):**
```python
system_prompt = f"""You are a SQL expert. Generate {dialect} SQL.

Schema documentation:
{external_knowledge_text}

Question: {question}
"""
```

Works but can be expensive — some docs are 2,000–4,000 tokens. For GPT-4o at $2.50/1M input tokens, adding 3,000 tokens per instance costs an extra $0.0075 per query. Across 107 instances: $0.80. Acceptable.

**Better approach — retrieve relevant sections:**
Embed the question and the external knowledge document. Retrieve only the sections with high cosine similarity to the question. Reduces prompt size by 50–70% for large docs.

```python
def get_relevant_knowledge(question: str, doc_text: str, max_tokens: int = 1000) -> str:
    # Split doc into paragraphs
    paragraphs = [p.strip() for p in doc_text.split("\n\n") if p.strip()]
    if not paragraphs:
        return doc_text

    # Embed question and paragraphs
    q_emb = embedder.embed(question)
    p_embs = embedder.embed_batch(paragraphs)

    # Score and select top paragraphs within token budget
    scored = sorted(
        zip(paragraphs, cosine_similarities(q_emb, p_embs)),
        key=lambda x: x[1], reverse=True
    )

    selected, token_count = [], 0
    for para, score in scored:
        para_tokens = len(para.split()) * 1.3  # rough estimate
        if token_count + para_tokens > max_tokens:
            break
        selected.append(para)
        token_count += para_tokens

    return "\n\n".join(selected)
```

**Always include the external knowledge for instances that have it** — the evaluation framework was designed with the expectation that models use it. Omitting it artificially lowers your score on those 107 instances.

---

## Dialect-Specific Generation

Spider2-lite spans three dialects. Getting dialect right is worth 10–20pp on the benchmark.

**Common dialect pitfalls:**

| Feature | BigQuery | Snowflake | SQLite |
|---------|----------|-----------|--------|
| Row limit | `LIMIT n` | `LIMIT n` | `LIMIT n` |
| Date truncation | `DATE_TRUNC(date, MONTH)` | `DATE_TRUNC('month', date)` | `strftime('%Y-%m', date)` |
| String concat | `CONCAT(a, b)` or `a \|\| b` | `a \|\| b` or `CONCAT` | `a \|\| b` |
| Current date | `CURRENT_DATE()` | `CURRENT_DATE()` | `date('now')` |
| Array aggregation | `ARRAY_AGG(x)` | `ARRAY_AGG(x)` | Not native |
| JSON access | `JSON_VALUE(col, '$.key')` | `col:key` (colon syntax) | `json_extract(col, '$.key')` |
| Regex | `REGEXP_CONTAINS(col, r'...')` | `REGEXP_LIKE(col, '...')` | `col REGEXP '...'` |
| Window functions | Full support | Full support | Limited (SQLite 3.25+) |
| Timestamp microseconds | `TIMESTAMP_MICROS(n)` | `TO_TIMESTAMP(n/1e6)` | `datetime(n/1e6, 'unixepoch')` |

**Implementation options:**

**Option A — Single model, dialect in system prompt:**
```python
dialect_instruction = {
    "bigquery":  "Generate Google BigQuery Standard SQL. Use ARRAY_AGG, DATE_TRUNC(date, PERIOD), REGEXP_CONTAINS, TIMESTAMP_MICROS as appropriate.",
    "snowflake": "Generate Snowflake SQL. Use DATE_TRUNC('period', date), REGEXP_LIKE, semi-structured colon syntax for JSON (col:key::type) as appropriate.",
    "sqlite":    "Generate SQLite SQL. Use strftime() for date formatting, || for string concatenation, json_extract() for JSON. No ARRAY_AGG.",
}
```

**Option B — Transpile to target dialect after generation:**
```python
import sqlglot

def transpile_to_dialect(sql: str, target: str) -> str:
    dialect_map = {"bigquery": "bigquery", "snowflake": "snowflake", "sqlite": "sqlite"}
    return sqlglot.transpile(sql, write=dialect_map[target], pretty=True)[0]
```

sqlglot handles most syntax differences automatically. Fails on dialect-specific functions with no equivalent (e.g. `ARRAY_AGG` has no SQLite equivalent). Use transpilation as a safety net after dialect-instructed generation, not as the primary mechanism.

**Option C — Route to dialect-specific fine-tuned models:**
Most expensive to maintain, highest accuracy for dialect-specific features. Justified only if benchmark results show significant dialect-specific accuracy gaps.

---

## Running Evaluation and Reading Results

```bash
cd Spider2/spider2-lite/evaluation_suite

# Evaluate SQL predictions (runs each SQL, compares output to gold)
python evaluate.py \
    --result_dir ../../my_predictions \
    --mode sql \
    --max_workers 20 \
    --timeout 60

# Evaluate pre-executed CSV results (faster, no database connections needed)
python evaluate.py \
    --result_dir ../../my_predictions_csv \
    --mode exec_result \
    --max_workers 20
```

**Output:**
```
{bq011: 1, bq012: 0, bq015: 1, sf001: 0, ...}
Final score: 0.347, Correct examples: 190, Total examples: 547
Real score: 0.347, Correct examples: 190, Total examples: 547
TOTAL_GB_PROCESSED: 3.24150 GB
```

- `Final score` = correct / evaluated (useful during partial runs)
- `Real score` = correct / 547 (the official metric — denominator is always 547)
- The script saves a `correct_ids.csv` in your predictions folder listing which instances passed

**Partial evaluation (SQLite only, no credentials needed):**
```bash
# Filter to only SQLite instances
python -c "
import json
instances = [json.loads(l) for l in open('../spider2-lite.jsonl')]
sqlite_ids = {i['instance_id'] for i in instances if not i['instance_id'].startswith(('bq','sf'))}
print(len(sqlite_ids), 'SQLite instances')
"
# 160 instances — evaluate these first while BigQuery/Snowflake setup is in progress
```

---

## Segmenting Failure Analysis

A single score of 34% tells you nothing about where to improve. Segment by every available dimension.

```python
import json, csv
from pathlib import Path
from collections import defaultdict

# Load evaluation results
correct_ids = set()
with open("my_predictions/correct_ids.csv") as f:
    for row in csv.reader(f):
        correct_ids.add(row[0])

# Load instance metadata
instances = {}
with open("spider2-lite/spider2-lite.jsonl") as f:
    for line in f:
        inst = json.loads(line)
        instances[inst["instance_id"]] = inst

# Load eval standard (has difficulty, condition_cols etc.)
eval_std = {}
with open("spider2-lite/evaluation_suite/gold/spider2lite_eval.jsonl") as f:
    for line in f:
        item = json.loads(line)
        eval_std[item["instance_id"]] = item

# Segment results
segments = defaultdict(lambda: {"correct": 0, "total": 0})

for iid, inst in instances.items():
    # By dialect
    if iid.startswith("bq"):   dialect = "bigquery"
    elif iid.startswith("sf"): dialect = "snowflake"
    else:                      dialect = "sqlite"

    # By external knowledge
    has_ext = "with_ext_knowledge" if inst.get("external_knowledge") else "no_ext_knowledge"

    # By order sensitivity (from eval standard)
    ignore_order = eval_std.get(iid, {}).get("ignore_order", True)
    order_type = "order_insensitive" if ignore_order else "order_sensitive"

    correct = 1 if iid in correct_ids else 0

    for key in [dialect, has_ext, order_type]:
        segments[key]["correct"] += correct
        segments[key]["total"] += 1

print("Segment Analysis:")
print(f"{'Segment':<25} {'Accuracy':>10} {'Correct':>10} {'Total':>8}")
print("-" * 55)
for seg, counts in sorted(segments.items()):
    acc = counts["correct"] / counts["total"]
    print(f"{seg:<25} {acc:>10.1%} {counts['correct']:>10} {counts['total']:>8}")
```

**Example output:**
```
Segment Analysis:
Segment                   Accuracy    Correct    Total
-------------------------------------------------------
bigquery                     29.4%         53      180
snowflake                    38.6%         80      207
sqlite                       35.6%         57      160
with_ext_knowledge           18.7%         20      107
no_ext_knowledge             38.5%        170      440
order_insensitive            36.4%        176      483
order_sensitive              21.9%         14       64
```

**What this tells you:**
- BigQuery is weakest (29.4%) → dialect-specific generation or date/timestamp handling
- External knowledge is the biggest gap (18.7% vs 38.5%) → knowledge injection is not working
- Order-sensitive queries are weak (21.9%) → ORDER BY handling or result ordering logic

**Prioritise fixes by:**
```
impact = (target_accuracy - current_accuracy) × instance_count
```
External knowledge: (38.5% - 18.7%) × 107 = 21.2 additional correct instances if fixed
BigQuery dialect:   (38.5% - 29.4%) × 180 = 16.4 additional correct instances if fixed
```

Fix external knowledge first.

---

## Execution-Based Evaluation — What It Catches and Misses

### What it catches that exact-match misses

**Semantically equivalent SQL:**
```sql
-- Both are correct; exact match fails on the second
SELECT customer_id FROM orders GROUP BY customer_id HAVING COUNT(*) > 3
SELECT DISTINCT customer_id FROM (SELECT customer_id, COUNT(*) as cnt FROM orders GROUP BY customer_id WHERE cnt > 3)
```
Execution comparison: both return the same result set → both score 1.

**Column ordering differences:**
```sql
SELECT name, revenue FROM ...   -- gold
SELECT revenue, name FROM ...   -- predicted
```
Spider2-lite's `compare_pandas_table` checks column vectors independently, not ordered left-to-right, so this scores 1.

**Equivalent filter conditions:**
```sql
WHERE status = 'active'         -- gold
WHERE status != 'inactive'      -- predicted (same result on this dataset)
```
Scores 1 if the dataset has only 'active' and 'inactive' values — the results match.

### What it still misses

**Wrong SQL that happens to return the same result:**
If the gold SQL returns `(42)` and your SQL also returns `(42)` via a completely different (and generally wrong) query path, you score 1. Execution comparison cannot detect this — it is the same limitation as all execution-based benchmarks.

**Dataset-specific correctness:**
A query with an off-by-one date boundary may produce the same result on Spider2-lite's specific dataset because no records fall on the boundary date. The SQL is wrong in general but scores 1.

**Performance correctness:**
A correct but O(n²) query that times out on the production dataset scores 0 (timeout) on Spider2-lite. This is actually useful — a 60-second timeout (`--timeout 60`) exposes inefficient queries.

**Partial credit:**
Spider2-lite gives 0 or 1 per instance. No partial credit for a query that returns the right columns with a wrong aggregation, or the right rows with the wrong column values. This makes the metric harsh but unambiguous.

---

## Using Spider2-lite as a Regression Gate in CI

Once you have a baseline score, use Spider2-lite (SQLite subset, no credentials) as an automated regression check:

```yaml
# .github/workflows/nl2sql-eval.yml
jobs:
  spider2-lite-regression:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Spider2-lite
        run: |
          git clone --depth 1 https://github.com/xlang-ai/Spider2.git /tmp/spider2
          pip install pandas tqdm

      - name: Generate SQLite predictions
        run: |
          python scripts/generate_spider2_predictions.py \
            --subset sqlite \
            --output /tmp/predictions

      - name: Run evaluation
        run: |
          cd /tmp/spider2/spider2-lite/evaluation_suite
          python evaluate.py \
            --result_dir /tmp/predictions \
            --mode sql \
            --max_workers 4 \
            --timeout 30 \
            2>&1 | tee eval_output.txt

          # Extract score and compare to baseline
          SCORE=$(grep "Real score:" eval_output.txt | awk '{print $3}' | tr -d ',')
          BASELINE=0.35   # set from your last known good score
          python -c "
          score = float('$SCORE')
          baseline = float('$BASELINE')
          if score < baseline - 0.03:
              print(f'REGRESSION: {score:.3f} < {baseline:.3f} - 0.03')
              exit(1)
          print(f'PASS: {score:.3f} (baseline {baseline:.3f})')
          "
```

**What this catches:**
- Prompt changes that improve one query class but regress SQLite queries
- Model routing changes that accidentally send complex queries to the smaller model
- Schema retrieval changes that hurt recall on novel schemas

**Evaluation time for SQLite subset:** ~3–8 minutes with 4 workers (160 instances, SQLite is fast). Acceptable for a CI gate.

**Update the baseline** whenever you intentionally improve the system: commit the new baseline score alongside the code change so the diff is auditable.
