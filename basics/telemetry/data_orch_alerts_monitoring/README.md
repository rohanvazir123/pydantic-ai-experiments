# Data Orchestration, Observability & Operational Reliability

A field guide covering three things production systems live or die on: how
work gets **scheduled and orchestrated**, how failures get **seen before a
user reports them**, and how a team **responds once something breaks
anyway**. Each section compares the industry-standard tools with their real
limitations, not just their marketing, and closes with concrete callouts to
where this exact repo already builds smaller versions of these same ideas.

## Table of Contents

- [Data Orchestration Tools](#data-orchestration-tools)
  - [Comparison](#orchestration-comparison)
  - [Orchestration in this repo](#orchestration-in-this-repo)
- [Observability: Metrics, Logs, Traces, Alerting](#observability-metrics-logs-traces-alerting)
  - [Comparison](#observability-comparison)
  - [Observability in this repo](#observability-in-this-repo)
- [System Reliability & Incident Response](#system-reliability--incident-response)
  - [Reliability patterns](#reliability-patterns)
  - [Incident response](#incident-response)
  - [Reliability patterns in this repo](#reliability-patterns-in-this-repo)
- [Designing Robust Production Systems](#designing-robust-production-systems)
- [Status](#status)

## Data Orchestration Tools

| Tool | Model | Advantages | Limitations / issues |
|---|---|---|---|
| **Apache Airflow** | Static Python DAGs, task-centric | Huge ecosystem, mature, most-hired-for skill, good UI for DAG state | Scheduler struggles at high DAG/task counts; dynamic DAGs are a workaround, not a first-class feature; XCom is a poor fit for passing real data (meant for small metadata); retries are task-level, not data-aware (a retried task re-runs, it doesn't know *why* it failed); backfills are notoriously fiddly to get exactly right |
| **Dagster** | Software-defined **assets**, not just tasks | Asset lineage and typing built in; strong local testing story; observability (materialization history) is native, not bolted on | Asset-centric mental model is a real shift from Airflow's task-centric one — migration cost is conceptual, not just mechanical; smaller hiring pool than Airflow |
| **Prefect** | Dynamic, Python-native flows | Easiest local dev loop of the three; hybrid execution (your infra runs tasks, Prefect Cloud/Server only orchestrates) avoids full vendor lock-in | Prefect 1→2 was a breaking rewrite that burned some adopters' trust; smaller plugin/ecosystem surface than Airflow; Prefect Cloud cost scales with usage |
| **Temporal** (and Cadence) | Durable **workflow-as-code**, not a DAG scheduler | Strong execution guarantees for long-running, stateful workflows (survives worker crashes mid-workflow); genuinely good for orchestrating *microservices*, not just batch data jobs; human-in-the-loop / signal support | Operating the Temporal server cluster is its own project; steep learning curve (workflow/activity/signal model is unfamiliar to most data engineers); overkill for "run this batch job nightly" |
| **dbt** | SQL-centric transform layer, not a scheduler | Best-in-class for the transform step specifically: testing, docs, lineage, modularity; became the de facto standard for the "T" in ELT | Not a general orchestrator — still needs Airflow/Dagster/Prefect/cron to actually trigger `dbt run`; Jinja macros get unreadable at scale; encourages "everything is SQL" even where it isn't the right tool |
| **Argo Workflows** | Kubernetes-native DAGs | GitOps-friendly, no separate scheduler process to run, scales with your k8s cluster | YAML-heavy authoring; steep for teams not already deep in Kubernetes; less mature Python-native ergonomics than Airflow/Dagster/Prefect |
| **Cloud-managed** (Step Functions, Cloud Composer, Azure Data Factory) | Managed versions of the above | No infra to operate; integrates natively with the rest of that cloud | Vendor lock-in; local dev/testing is worse than self-hosted; cost model can surprise at scale (per-transition pricing on Step Functions, for example) |

### Orchestration comparison

The real axis isn't "which is best" — it's **what unit of work are you
scheduling**. Batch data pipelines (extract → transform → load, nightly or
hourly) fit Airflow/Dagster/Prefect well. Long-running stateful business
processes (a multi-day approval workflow, a saga across microservices) fit
Temporal, not a DAG scheduler — forcing that shape into Airflow means
fighting its execution model the whole way. dbt isn't a competitor to any of
these; it's a specialist tool that needs one of them (or plain cron) sitting
above it.

### Orchestration in this repo

- `rag/v2/knowledge/scheduler/` is a real APScheduler-based job runner with
  its own job store (`schema/007_scheduler.sql`) — the same "durable job
  table, claim with a transaction" idea `job_sched/README.md`'s Approach C
  formalizes as a design, citing that exact module as prior art.
- `job_sched/README.md` itself is a from-scratch evaluation of three ways to
  build a priority scheduler (in-process heap, Redis ZSETs, Postgres
  `FOR UPDATE SKIP LOCKED`) — the same durability/latency/coordination
  trade-offs that separate Airflow-style schedulers from Temporal-style
  durable execution, worked out at smaller scale.

## Observability: Metrics, Logs, Traces, Alerting

| Category | Tool(s) | Advantages | Limitations / issues |
|---|---|---|---|
| **Metrics** | Prometheus + Grafana | Pull-based, powerful query language (PromQL), the open-source default; huge dashboard ecosystem | Local disk storage doesn't scale to long retention/high cardinality without a remote-write backend (Thanos, Cortex, Mimir) — that's a second system to operate; high-cardinality labels (e.g. per-user metrics) silently blow up memory |
| **Metrics (managed)** | Datadog, CloudWatch, New Relic | No infra to run; fast time-to-value; unified with logs/traces in one UI | Cost scales fast with hosts/custom metrics; vendor lock-in on dashboards and alert definitions |
| **Logs** | ELK / OpenSearch | Full-text search, mature, flexible | Expensive to run at scale (indexing everything is costly in both storage and CPU); operational burden of running Elasticsearch/OpenSearch clusters is nontrivial |
| **Logs** | Grafana Loki | Cheap — indexes only labels, not full log text; pairs naturally with Prometheus/Grafana | Query language (LogQL) is less powerful for ad-hoc full-text search than Elasticsearch; label cardinality mistakes hurt the same way as in Prometheus |
| **Traces** | OpenTelemetry (instrumentation) + Jaeger/Tempo (storage/UI) | Vendor-neutral instrumentation standard — instrument once, send anywhere; avoids re-instrumenting if the backend changes later | OTel itself is just the SDK/protocol — still need to run and pay for a backend; distributed tracing is only as useful as instrumentation coverage, and partial coverage leaves blind spots exactly where an incident needs them |
| **Traces / wide events** | Honeycomb, Datadog APM | High-cardinality, arbitrary-dimension queries on live production data — genuinely better than pre-aggregated metrics for "why is *this specific* request slow" debugging | Cost and vendor lock-in (Honeycomb); the "three pillars" framing itself has real critics (Honeycomb's own position: wide structured events subsume metrics+logs+traces, and splitting them loses the ability to correlate) |
| **Alert routing** | Prometheus Alertmanager | Free, integrates directly with Prometheus, handles dedup/grouping/silencing | Configuration is YAML-and-regex heavy; no native on-call scheduling — needs PagerDuty/Opsgenie on top for rotations |
| **On-call / paging** | PagerDuty, Opsgenie, VictorOps | Escalation policies, rotations, mobile paging, incident timelines | Cost per seat; yet another system whose config can drift from the alerting rules that feed it |

### Observability comparison

The dangerous failure mode isn't "no monitoring" — it's **alerts that fire
on symptoms nobody acts on**, which trains the on-call rotation to ignore
pages (alert fatigue is a reliability bug, not a tooling gap). The fix is
alerting on **SLO burn rate**, not raw thresholds: "error rate > 1%" pages
constantly on a noisy-but-fine service and never pages early enough on a
slow-burning outage; "we'll exhaust 2% of this month's error budget in the
next hour at the current rate" pages exactly when it matters, at a
severity proportional to how fast the budget is burning.

### Observability in this repo

- `rag/v2` already runs the open-source stack directly:
  `docker-compose.observability.yml` (Prometheus + Grafana),
  `knowledge/observability/` (metrics + alert helpers), and
  `infra/` holding the actual Grafana dashboards and Prometheus config — not
  a toy, the real thing, scoped to one service.
- `worker_queues/base.py`'s `_status`/`_cancelled` map is a crude,
  in-memory analog of an observability surface: "can I answer `GET
  /jobs/{id}` for any job, from any worker, without knowing which one ran
  it" is the same question a metrics/tracing system answers for a whole
  fleet — just answered here with a single shared dict instead of a real
  time-series store, and the parent README names the exact bug that comes
  with skipping the real thing (`_status` grows without bound; nobody
  sweeps it).

## System Reliability & Incident Response

### Reliability patterns

| Pattern | What it solves | Where it can go wrong |
|---|---|---|
| **Circuit breaker** | Stops hammering a dependency that's already failing; fails fast instead of piling up timeouts | Wrong thresholds either trip too eagerly (false positives take down a healthy dependency) or too late (the point of the pattern); half-open recovery logic is easy to get subtly wrong |
| **Retry with backoff + jitter** | Absorbs transient failures without a thundering herd | Retrying a non-idempotent operation causes duplicate side effects; backoff without jitter synchronizes retries across clients into a new thundering herd |
| **Bulkheads** | One overloaded dependency doesn't starve resources needed by unrelated work | Over-partitioning resources leaves each bulkhead under-provisioned for its own peak load |
| **Backpressure / bounded queues** | Bounds memory *and* surfaces overload immediately instead of hiding it in an unbounded buffer | A bound with no drop/shed policy just turns unbounded memory growth into an unbounded stall — needs a paired decision (this repo's own "two bounds" rule: `worker_queues`/`rate_limiter`) |
| **Idempotency** | Makes retries and at-least-once delivery safe | Requires a stable idempotency key end-to-end (client-generated request ID, or a natural dedup key) — bolting it on after the fact is much harder than designing for it |
| **Graceful degradation** | A non-critical dependency failing degrades the response instead of failing the whole request | Requires deciding, ahead of the incident, which features are "load-bearing" and which are droppable — a decision teams often haven't made until the outage forces it |

### Incident response

- **SLIs/SLOs/error budgets** (Google SRE book): define what "working" means
  quantitatively *before* an incident, not during one. The error budget is
  what turns "should we ship this risky change" into a data question
  instead of a debate.
- **Severity levels + incident commander**: a named IC owns coordination
  (not necessarily the fix) so engineers fixing the problem aren't also
  managing stakeholder updates — the two jobs compete for the same
  attention under pressure.
- **Runbooks**: written *before* the incident, for the failure modes anyone
  can predict — the value is speed under pressure, not novelty.
- **Blameless postmortems**: the goal is "what let this happen" (process,
  missing alert, missing runbook), not "who broke it" — blame teaches
  people to hide problems, not prevent them.
- **Chaos engineering** (Chaos Monkey, Gremlin, Litmus): proactively
  triggering the failure modes a design claims to handle, on purpose, in a
  controlled way — the only way to know a circuit breaker/retry/failover
  path actually works is to have watched it trigger.
- **Load testing** (Locust, k6, Gatling): finds the capacity ceiling and the
  failure mode at that ceiling *before* real traffic does. `rag/v2/tests/load/`
  already uses Locust for exactly this in this repo.

### Reliability patterns in this repo

- `rate_limiter/` is a real distributed token bucket — the one place in
  this repo's telemetry exercises where state is genuinely shared (not
  partitionable), so it pays the full coordination price
  (`WATCH`/`MULTI`/`EXEC`) rather than getting to dissolve the problem via
  ownership. Backpressure/shedding, worked out concretely.
- `rag/v2/knowledge/bus/` has a real circuit breaker in front of Redis
  Streams — this is the design doc's own cited prior art for "an external
  dependency in a hot path must never block, must be capacity-bounded, must
  fail closed" (see `anomaly/README.md`'s weather-enrichment section, which
  explicitly reuses this reasoning).
- `resequencer.py` names its own liveness gap in writing rather than hiding
  it: `max_delay` never fires for a quiet vehicle because nothing re-checks
  time without a new frame arriving. That's the reliability-patterns table
  above applied to one file — a bound that's real on paper but has an edge
  case, written down instead of assumed away.
- `worker_queues/base.py`'s `WorkerPool.start()` is a template method
  precisely so failure during setup (`_init_workers` spawning processes)
  can't leave a half-built pool escaping into use — construction-is-inert,
  `start()`-runs as a reliability property, not just a style choice.

## Designing Robust Production Systems

Synthesizing the three sections above into a short checklist, roughly in
the order these decisions actually need to get made:

1. **Define the SLO before writing the pipeline**, not after it's in
   production — "99.9% of events processed within 5 minutes of ingestion"
   is a design constraint (it tells you whether you need Kafka-scale
   durability or a cron job is fine), not a monitoring afterthought.
2. **Design for failure, not just for the happy path** — every external
   call needs a named answer to "what happens when this times out, returns
   garbage, or is simply gone," the way `anomaly/README.md`'s weather
   enrichment section explicitly names "enrich with nulls, never block" as
   its failure-mode answer rather than leaving it implicit.
3. **Make retries safe before making them automatic** — idempotency keys
   end-to-end, or don't retry non-idempotent operations at all.
4. **Alert on budget burn, not on raw thresholds** — see Observability
   comparison above; this is the single highest-leverage change most teams
   can make to their existing Prometheus/Datadog setup without adopting any
   new tool.
5. **Bound every buffer with two numbers, not one** — a size bound and a
   time bound, because memory and latency are different failure modes and
   either alone leaves the other unbounded (this repo's own recurring rule,
   `basics/telemetry/README.md` Question 2).
6. **Write the runbook and run the chaos test before the real incident**,
   not after — a reliability pattern nobody has watched trigger under
   controlled conditions is a hypothesis, not a guarantee.
7. **Blameless postmortems, tracked action items** — a postmortem that ends
   at "we understand what happened" without a tracked follow-up is a story,
   not a fix.

## Status

**Reference document — not a design doc for code in this repo.** No
implementation is planned from this folder; it exists to evaluate tooling
and reliability thinking against concrete prior art already built elsewhere
in this repo.
