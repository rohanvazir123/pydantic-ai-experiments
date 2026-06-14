# tests/chaos/

## Table of Contents

- [What This Is](#what-this-is)
- [Running Chaos Tests](#running-chaos-tests)

---

## What This Is

Fault injection tests. Each test kills one external service while load continues at 3 RPS, then verifies graceful degradation and recovery. Run against staging only — never production.

For the full scenario matrix and acceptance criteria see `TEST_QA_REFERENCE.md §7 Chaos and Resilience Test Plan`.

---

## Running Chaos Tests

Via Makefile (local Docker Compose):

```bash
cd rag/v2
make chaos-kill-redis     # stops Redis 60s, verifies no_cache mode, restarts
make chaos-kill-ollama    # stops Ollama 120s, verifies search_only + circuit open
make chaos-kill-postgres  # stops PostgreSQL 30s, verifies 503 + job queue integrity
```

**Non-negotiable acceptance criteria for every scenario:**
1. No HTTP 500s — all errors return structured error codes
2. `X-Degraded-Mode` header present on every response during failure
3. No data corruption — tenant isolation holds
4. Alert email sent to `rohan.vazirani@gmail.com` within 60s of circuit opening
5. DLQ depth = 0 after all services restored
