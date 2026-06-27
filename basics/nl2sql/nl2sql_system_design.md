# NL2SQL System Design Basics

Key design areas, trade-offs, and deep-dive topics for building production NL2SQL systems.

## Table of Contents

- [Pipeline Architecture](#pipeline-architecture)
- [Schema Representation and Context](#schema-representation-and-context)
- [Accuracy, Evaluation, and Trust](#accuracy-evaluation-and-trust)
- [Multi-Turn and Context](#multi-turn-and-context)
- [Execution, Safety, and Security](#execution-safety-and-security)
- [Latency and Performance](#latency-and-performance)
- [Model and Prompt Strategy](#model-and-prompt-strategy)
- [Feedback Loops and Continuous Improvement](#feedback-loops-and-continuous-improvement)
- [Hard Trade-offs](#hard-trade-offs)
- [Latency SLAs](#latency-slas)
- [Correctness Around Vague Queries](#correctness-around-vague-queries)
- [Schema Changes and Drift](#schema-changes-and-drift)
- [Passing the Right Subset of Schemas to the LLM](#passing-the-right-subset-of-schemas-to-the-llm)
- [Benchmarking with Spider2-lite](#benchmarking-with-spider2-lite)

---

## Pipeline Architecture

**1.** Walk me through the end-to-end architecture of a production NL2SQL system. Where are the failure modes, and how do you detect them at each stage without a human in the loop?

**2.** How do you decide where to draw the boundary between what the LLM handles and what deterministic code handles? What are the consequences of getting that boundary wrong in each direction?

**3.** Schema linking is often cited as the hardest sub-problem in NL2SQL. Describe your approach — how do you map a natural language phrase like "top performing reps last quarter" to the correct tables, columns, and time logic in a schema you've never seen?

**4.** How does your pipeline behave when the user's question is genuinely ambiguous — the kind where two different SQL queries are both valid interpretations? Do you pick one, ask for clarification, or return both? What are the product trade-offs of each?

---

## Schema Representation and Context

**5.** A schema has 400 tables, 6,000 columns, extensive foreign key relationships, and inconsistent naming conventions built up over 15 years. You can't fit it all in context. How do you decide what to include, and what do you do when the relevant tables aren't retrieved?

**6.** Column names like `amt`, `flg`, `cd` are meaningless to an LLM. How do you enrich schema metadata without requiring the data team to manually annotate 6,000 columns, and how do you keep that enrichment fresh as the schema evolves?

**7.** How do you represent foreign key relationships and join paths to the model? Embedding them as prose vs. structured DDL vs. a graph — what are the retrieval and reasoning trade-offs?

**8.** How does your schema retrieval strategy change when the database is a data warehouse (star schema, denormalized) versus an OLTP system (normalized, 3NF)? Does the same NL2SQL stack work for both?

---

## Accuracy, Evaluation, and Trust

**9.** How do you build a ground-truth evaluation set for a domain you're deploying into for the first time, before you have any user queries? What biases does that evaluation set carry, and how do they affect what your accuracy numbers actually mean?

**10.** Execution accuracy and exact match accuracy both have serious flaws as metrics. Describe a metric or evaluation framework you'd actually trust for a production system — and be specific about what it still misses.

**11.** A query runs successfully and returns results, but the results are semantically wrong — the SQL is syntactically valid, executes cleanly, but answers a different question than the user asked. How do you detect this class of error? Can you detect it at all without ground truth?

**12.** Your system has 85% execution accuracy on your benchmark. Your PM says that's good enough to ship. What do you tell them, and why might 85% on a benchmark be misleading in both directions?

---

## Multi-Turn and Context

**13.** A user says "now filter that by region" as a follow-up. How do you resolve "that" to the previous query's context? What breaks in your approach when the conversation is 10 turns deep and the user has changed topics in the middle?

**14.** How do you handle pronoun and entity co-reference across turns — for example, "show me their top customers" where "their" refers to a company mentioned three turns ago? What does your state management look like?

**15.** When should your system refuse to carry context forward and instead ask the user to re-state their question? How do you make that decision programmatically?

---

## Execution, Safety, and Security

**16.** What is your full defence against prompt injection in an NL2SQL system? Be specific — a malicious user crafts a question designed to exfiltrate data or drop tables. Walk me through every layer.

**17.** How do you enforce row-level security and column-level access controls when the LLM generates the SQL? What happens if the LLM generates a query that technically bypasses the application-level permissions you've defined?

**18.** Before executing LLM-generated SQL, what validation do you run? What can static analysis catch, what requires a dry run or explain plan, and what can only be caught post-execution?

**19.** Your system generates a query with a missing `WHERE` clause on a 2 billion row table. The warehouse query costs $400 and takes 45 minutes. How do you prevent this at the system level, and how do you balance query guardrails against rejecting legitimate analytical questions?

---

## Latency and Performance

**20.** A user expects a response in under 2 seconds. Your LLM call alone takes 1.5 seconds. Walk me through every latency lever you have — what can you parallelize, what can you cache, and what are the correctness risks of each optimization?

**21.** You want to cache generated SQL for repeated questions. What is your cache key, and what are the invalidation conditions? What goes wrong if a schema migration happens and your cache isn't invalidated correctly?

**22.** How does streaming change the user experience and the system architecture for NL2SQL? Are there cases where streaming the SQL generation is harmful rather than helpful?

---

## Model and Prompt Strategy

**23.** Fine-tuning versus few-shot prompting versus RAG over a query library — lay out the trade-offs across accuracy, maintainability, schema adaptability, and operational cost. When does each approach win?

**24.** How do you construct few-shot examples for a schema the model has never seen? How many examples do you need, how do you select them, and what is the risk of examples that closely match the user's question but produce subtly wrong SQL?

**25.** Your prompt is 8,000 tokens on average. The model's context window is 128k but cost and latency scale with input length. How do you decide what to cut, and how do you measure the accuracy impact of each cut?

**26.** How do you handle SQL dialects — Postgres, BigQuery, Spark SQL, T-SQL all have meaningfully different syntax. Do you run one model per dialect, detect and branch, or prompt the model to handle it? What are the failure modes of each?

---

## Feedback Loops and Continuous Improvement

**27.** How do you build a feedback loop when most users won't explicitly tell you the result was wrong — they'll just re-phrase the question or give up? What implicit signals can you collect, and how do you avoid training on noisy labels?

**28.** A user edits the generated SQL before running it. How do you use that edit signal to improve the system? How do you distinguish "the SQL was wrong" from "the user changed their mind about what they wanted"?

**29.** How do you safely deploy a model update that improves accuracy on new query patterns but regresses on a subset of existing ones? What does your rollout strategy look like?

---

## Hard Trade-offs

**30.** You can invest in making the system more accurate or more explainable — but not both equally. A more accurate system uses a larger model that is harder to introspect; a more explainable system uses chain-of-thought that exposes reasoning but adds latency and sometimes decreases accuracy. How do you make that call, and for what user populations does the answer change?

**31.** Should your system tell the user when it is uncertain, or silently return its best guess? What is the product cost of each in a financial reporting context versus a casual analytics context?

**32.** Your NL2SQL system is being used by both analysts who write SQL fluently and executives who cannot. The optimal prompting strategy, output format, and error messages are different for each persona. How do you handle this without maintaining two separate systems?

---

## Latency SLAs

**33.** You have a p99 latency SLA of 3 seconds end-to-end. Your LLM call is 1.8s, schema retrieval is 400ms, SQL validation and execution is 600ms — you're already at 2.8s with zero margin. Walk me through every architectural decision you make to hold that SLA under production load, and what you drop when you can't.

**34.** How do you set a latency SLA for NL2SQL in the first place? The query execution time is non-deterministic — a simple question might hit a cached result in 10ms or trigger a full table scan taking minutes. Where do you draw the SLA boundary, and what do you communicate to the user when SQL execution blows past it?

**35.** Your schema retrieval step has a p50 of 80ms but a p99 of 900ms. That long tail is killing your overall SLA. What are the causes of that tail, and do you solve it with infrastructure, algorithmic changes, or by changing what the retrieval step does?

**36.** You want to offer a "fast mode" that sacrifices some accuracy for lower latency. What exactly do you cut — fewer schema examples, a smaller model, skip validation, skip reranking? For each cut, quantify the accuracy risk and describe how you'd measure it before shipping.

---

## Correctness Around Vague Queries

**37.** "Show me the sales numbers" — vague in at least five dimensions: which sales metric, which time period, which region, which product line, which granularity. How does your system decide whether to ask a clarifying question, make a default assumption, or return multiple interpretations? What signals drive that decision?

**38.** When your system makes an assumption to resolve a vague query — say it defaults to the current quarter — how and where do you surface that assumption to the user? What is the risk of surfacing it too prominently versus too subtly?

**39.** How do you define a correctness SLA for vague queries? You cannot have a single ground-truth SQL for "show me performance trends" — multiple queries are defensibly correct. How do you measure whether your system is doing a good job on this class of input?

**40.** A user asks "who are our best customers?" — best by revenue, by margin, by order frequency, by recency, or by some weighted combination your company defines internally as a KPI. How does your system know which definition to use, where does that business logic live, and what happens when that definition changes?

**41.** At what point do you reject a query as too vague to answer rather than guess? What is the threshold, how is it computed, and what do you return to the user — an error, a clarifying question, or a best-effort result with a confidence warning?

---

## Schema Changes and Drift

**42.** A table is renamed overnight, a column is deprecated and replaced with two new columns, and a new fact table is added — all in the same migration. Your NL2SQL system has no idea any of this happened. Walk me through every place this breaks and your strategy for detecting and recovering from schema drift without requiring a manual re-deployment.

**43.** How do you version your schema representation? If a user asks about a query they ran last month and the schema has since changed, what does your system do — re-generate against the new schema, retrieve the old schema version, or something else?

**44.** Your schema cache has a TTL of 24 hours. A critical table is renamed at 9am and your cache doesn't refresh until midnight. Analysts are getting wrong SQL all day. How do you design the invalidation strategy so this cannot happen, and what are the operational trade-offs of more aggressive invalidation?

**45.** Some schema changes are backwards compatible — adding a nullable column — and some are breaking — renaming a column or changing its type. How does your system distinguish between the two, and does your handling differ?

**46.** A column `revenue` existed in the schema six months ago and has now been split into `gross_revenue` and `net_revenue`. Users are still asking about "revenue." How does your system handle this ambiguity, and how do you prevent it from silently generating SQL against a column that no longer exists?

**47.** Your schema has 40 tables added per quarter. How does your schema retrieval quality degrade as the schema grows, and how do you detect that degradation before users notice it?

---

## Passing the Right Subset of Schemas to the LLM

**48.** You have 500 tables. You can fit 15 in context. Your retrieval step picks the wrong 15 — the correct table is ranked 18th. The model generates plausible-looking SQL against the wrong table, it executes, and returns wrong data. How do you detect this failure, and how do you fix the retrieval step without just making the context window larger?

**49.** Walk me through your complete schema retrieval pipeline — embedding strategy, similarity metric, reranking approach, and final selection. Where does each component fail, and what is your fallback when the top-k retrieved schemas are insufficient?

**50.** How do you embed a table schema for retrieval? Column names and types alone are low-signal. Do you include sample values, column descriptions, query history, foreign keys, table-level documentation? For each addition, what is the retrieval quality gain and the embedding cost?

**51.** A user asks a question that requires a 4-way join across tables that are conceptually related but have no direct foreign key relationship — they're linked through two intermediate bridge tables. Your retrieval step has no signal that these bridge tables are needed. How do you surface them?

**52.** How do you decide the right value of k — the number of tables to include in context? Too few and you miss the right table; too many and you dilute the signal, increase cost, and push the relevant schema toward the edge of the context window where attention degrades. How do you tune this, and does k vary per query?

**53.** Some queries require only one table. Some require eight. A fixed k wastes context on simple queries and truncates complex ones. How do you build a system that dynamically determines how many tables to include, and how do you validate that the dynamic selection is correct?

**54.** Your schema retrieval uses semantic similarity — embedding the user question and matching against embedded table descriptions. But "show me the churn rate by cohort" doesn't semantically resemble any column name or table description; it requires domain knowledge that the schema text doesn't contain. How do you bridge that gap?

**55.** How do you handle multi-tenancy in schema retrieval? Tenant A has a `payments` table with 12 columns; Tenant B has a `payments` table with 40 columns and completely different semantics. Your retrieval and prompting must be tenant-scoped with zero cross-contamination. Walk me through the architecture.

---

## Benchmarking with Spider2-lite

Spider2-lite is an open benchmark for evaluating NL2SQL systems across real-world databases and SQL dialects. Repo: [xlang-ai/Spider2](https://github.com/xlang-ai/Spider2/tree/main/spider2-lite)

### What It Contains

547 NL→SQL instances across three backends:

| Backend | Instances | Setup required |
|---------|-----------|---------------|
| BigQuery | 180 | GCP project + `bigquery_credential.json` |
| Snowflake | 207 | Request access via their form (free) |
| SQLite | 160 | Download one `.sqlite` archive (~1.4 GB) |

107 instances include `external_knowledge` — markdown files with schema context the model needs to generate correct SQL (table descriptions, function references, dialect-specific docs).

Each instance has four fields:
```json
{
  "instance_id": "bq011",
  "db": "ga4",
  "question": "How many distinct pseudo users had positive engagement time in the 7-day period ending on January 7, 2021 at 23:59:59, but had no positive engagement time in the 2-day period ending on the same date?",
  "external_knowledge": "ga4_obfuscated_sample_ecommerce.events.md"
}
```

### Prediction Format

One `.sql` file per instance, named by `instance_id`:
```
your_predictions/
├── bq011.sql
├── sf001.sql
├── local001.sql
└── ...
```

### Running Evaluation

```bash
# Clone the repo
git clone https://github.com/xlang-ai/Spider2.git
cd Spider2/spider2-lite/evaluation_suite

# Install dependencies
pip install pandas google-cloud-bigquery tqdm

# SQLite only (no cloud credentials needed)
python evaluate.py --result_dir your_predictions/ --mode sql

# Or if you've pre-executed the SQL and have CSV result files
python evaluate.py --result_dir your_predictions_csv/ --mode exec_result
```

### Setup (one-time per backend)

**SQLite (fastest to start):**
```bash
# Download and unzip the local databases
# Place all .sqlite files into:
spider2-lite/resource/databases/spider2-localdb/
```

**BigQuery:**
- Create a GCP project, enable BigQuery API
- Download a service account JSON key
- Place it at `evaluation_suite/bigquery_credential.json`

**Snowflake:**
- Fill out the [Spider2 Snowflake Access form](https://docs.google.com/forms/d/e/1FAIpQLScbVIYcBkADVr-NcYm9fLMhlxR7zBAzg-jaew1VNRj6B8yD3Q/viewform)
- They email you credentials within a day or two
- Place them at `evaluation_suite/snowflake_credential.json`

### Metric

Execution-based accuracy: your SQL result set is compared to the gold result set column by column with 1e-2 float tolerance and optional order-insensitive comparison. No partial credit.

```
Final score = correct_instances / 547
```

The script also prints `correct / len(evaluated)` for the subset you have predictions for, so you can evaluate incrementally before generating all 547.

### Integrating with Your Pipeline

**56.** How do you wire your NL2SQL pipeline into Spider2-lite to generate predictions for all 547 instances automatically?

**57.** Spider2-lite includes instances with `external_knowledge` — markdown documentation files the model needs to answer correctly. How do you incorporate external knowledge into your prompt without exceeding the token budget?

**58.** Spider2-lite spans BigQuery, Snowflake, and SQLite — three different SQL dialects with meaningfully different syntax (date functions, LIMIT vs TOP, array handling, semi-structured data). How does your pipeline handle dialect-specific generation, and how does dialect confusion affect your score?

**59.** Your pipeline scores 60% on Spider2-lite but you need to understand where it fails. How do you segment the failure analysis — by database backend, by query complexity, by whether external knowledge was required — to prioritise what to fix first?

**60.** Spider2-lite uses execution-based evaluation — your SQL must produce the same result set as the gold SQL, not just look similar. What classes of error does this catch that exact-match evaluation misses, and what classes does it still miss?

Full setup and integration walkthrough: [.answers/18_spider2_lite_eval.md](.answers/18_spider2_lite_eval.md)
