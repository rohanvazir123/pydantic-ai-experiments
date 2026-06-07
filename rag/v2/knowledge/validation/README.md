# knowledge/validation/

## Table of Contents

- [What This Is](#what-this-is)
- [Files](#files)
- [Validation Chain](#validation-chain)

---

## What This Is

The input validation pipeline. Runs before any DB query or LLM call. Cheap checks first (regex, length) — expensive checks (nano model) last. Reject fast; pay compute only on clean requests.

---

## Files

| File | Purpose |
|------|---------|
| `pipeline.py` | `ValidationPipeline`: runs V1–V6 in order; returns first rejection or passes through |

---

## Validation Chain

| Step | Check | Cost | Rejection code |
|------|-------|------|----------------|
| V1 | Pydantic schema validation | < 1ms | `400 Bad Request` |
| V2 | Length guard (`len(query) > MAX_QUERY_CHARS`) | < 1ms | `422` |
| V3 | Language detection (optional) | < 5ms | `422` |
| V4 | Prompt injection detector (regex + embedding-sim) | < 10ms | `422` |
| V5 | Content policy classifier (nano model) | ~50ms | `422` (off-topic) / `400` (inappropriate) |
| V6 | Corpus access RBAC (JWT roles) | < 2ms (cached) | `403 Forbidden` |

V5 is the only step with an LLM call. It uses the nano model tier (`qwen2.5:0.5b`) and returns `ContentPolicyResult(verdict, confidence, reason)`. The `reason` is logged but never returned to the client.

V6 runs after V5 so a forbidden corpus request doesn't reveal which content policy check the query passed.
