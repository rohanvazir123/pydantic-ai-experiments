# Session Context

## What we're working on

System design interview prep. Two problems being designed and deep-dived:

1. **Multi-Step Risk-Aware Loan Application Router** — `LoanApproval/Design.md`
2. **Automated Transaction Dispute Pipeline** — not yet started (next)

---

## Current state of LoanApproval/Design.md

Complete, including a deep eval pipeline (previously the thin spot). Sections done:
- Problem statement, functional + non-functional requirements
- Capacity estimation (1M applications/day, ~12 avg req/s, 60–120 peak)
- Core entities including `LoanApplicationDetails` (raw applicant-submitted PII)
- API design (REST, PUT over PATCH, idempotency header)
- Infrastructure choices: PostgreSQL, Kafka, Redis, Temporal, token bucket rate limiter, circuit breaker per dependency
- High-level architecture diagram
- Data flow (Temporal workflow with parallel activities + HIL signal/timer pattern)
- Agentic AI components (L1 for verification steps, L1 LLM-as-judge for explanation only, code for routing)
- Data model with SQL (including `loan_application_details` with encrypted SSN)
- Test & eval framework — now split explicitly into decision-quality (fair-lending) vs explanation-quality (LLM-as-judge) tracks
- Observability
- Deep dives: idempotency for credit pulls, regulatory rules engine, worker pool scaling, Temporal workflow design (HIL + SLA timer), **eval pipeline architecture (Deep Dive 5, new)**
- Fault analysis — added hallucinated-explanation and silent-decision-boundary-shift rows
- Tradeoffs summary — added reject inference, shadow deployment, fair-lending CI gate rows

### Deep Dive 5 — eval pipeline (this session's focus, was the weak spot)

Built around three time horizons, since loan performance ground truth is delayed
(months to years) unlike typical same-day-label ML eval:
- **Pre-deploy**: golden-set replay (historical outcomes + underwriter-proxy labels +
  synthetic adversarial cases + stratified adverse-impact sample via BISG) gates CI.
- **Near-real-time**: explanation faithfulness sampling, underwriter override rate,
  tier-distribution drift via PSI (not a flat % threshold) gates alerting.
- **Long-horizon**: quarterly reconciliation against actual loan performance feeds
  back into the golden set.

Key concepts to be ready to explain out loud:
- Why decision-quality eval (fair lending / adverse impact ratio, 4/5 rule) and
  explanation-quality eval (LLM-as-judge faithfulness/hallucination check) are two
  separate tracks, not one "AI eval" blob.
- **Reject inference** — denied applicants' true performance is never observed;
  extrapolate from the accepted population's score-to-outcome curve.
- **Shadow deployment**, not canary, for LLM upgrades — safe specifically because
  the explanation is advisory-only (L1 judge, ties back to the original L1-vs-L3
  tradeoff already in the doc).
- Judge calibration is a distinct failure mode from explanation quality — measure
  the automated judge's agreement with human raters (Cohen's kappa) separately.

**Tooling: DeepEval for the explanation track only.** Decision quality (tier
routing) is deterministic code output — plain pytest over `(features,
expected_tier)` + a stats pass for PSI/adverse-impact, not an LLM-eval framework.
Explanation quality maps cleanly onto DeepEval's `LLMTestCase`:
- `input` = serialized RiskSignal prompt, `actual_output` = generated explanation,
  `context` = RiskSignal facts restated as strings (NOT `retrieval_context` — no
  RAG step here).
- `HallucinationMetric(context=...)` = the faithfulness/groundedness check.
- `GEval(criteria=...)` = coherence/actionability rubric (calibrated against
  1–5 human ratings) AND a separate narrower GEval for the ECOA prohibited-basis
  scan.
- `BiasMetric` = general demographic-bias language check.
- PII restatement check is a plain regex/classifier pass, not a DeepEval metric —
  no reason to pay judge-call cost for a bounded pattern match.
- `EvaluationDataset` + `deepeval test run` in CI satisfies the append-only golden
  set + regression-gate requirement, for the explanation track specifically.
- `LatencyMetric(max_latency=...)` — not LLM-judged, just asserts a `latency`
  value you measured yourself (`LLMTestCase(..., latency=measured_seconds)`)
  against a threshold. Pre-deploy check (did this PR slow down the judge call on
  the golden set) — complements, doesn't replace, the production p95/p99 in
  Observability (real traffic distribution vs. per-case CI assertion). Budget it
  as a slice of the overall 90s p95 end-to-end target.

**Section 7 — edge cases in the eval pipeline itself (staff+ framing).** Distinct
from the system's own Fault Analysis table: these are second-order failures of the
*evaluator*, where the eval reports green while the real system degrades. Be ready
to name unprompted:
- Survivorship bias / circularity — historical golden set only contains outcomes
  for what the *old* policy approved; a new policy is scored on resemblance to old
  blind spots, not ground truth. Real fix: a small randomized test-and-learn cohort.
- Goodhart's law — golden set gets taught to once engineers can see it. Split
  visible dev set vs. held-out set the tuning team doesn't control.
- Proxy-metric decay — underwriter override rate looks great when underwriters are
  actually just rubber-stamping (automation bias). Audit independent of override.
- Judge checks claim *keys* not *values* — groundedness check can pass a
  transposed number if it merely verifies the field was referenced.
- Multiple-comparisons inflation — many segments × many metrics per PR means
  something fails by chance most of the time; pre-register primary hard gates,
  minimum-N before a segment can trigger a block.
- Silent vendor-side model drift bypasses the CI trigger entirely (no internal PR
  to catch it) — pin exact model versions, monitor continuously in production too.
- Adversarial gaming of a known/discoverable threshold — randomize secondary
  review sampling even inside the "safe" band; boundary-clustering is itself a
  fraud signal.

## Key design decisions made

- **Temporal** chosen as workflow engine — multi-step durable execution, HIL waits via signals, SLA timers that survive worker restarts
- **L2 outer loop** — code controls flow/routing; model handles interpretation only (regulatory requirement)
- **No partial results** — if any verification step fails after retries, route to underwriter, never auto-decide
- **LoanApplicationDetails** — raw stated data (SSN encrypted AES-256-GCM, stated income, employer) separate from verification results; risk agent compares stated vs verified
- **PUT over PATCH** everywhere retries matter
- **Circuit breaker per external dependency** (bureau, ID provider, doc processor)

## Next session agenda

User will drill on both designs — questions about failure handling, evaluation,
model upgrade safety, latency, regulatory edge cases. Then we go through together.

Transaction Dispute Pipeline design still to be started — use `sd_template.md` as
the base, same approach as LoanApproval.

## Files to know about

- `.system_design_practice/LoanApproval/Design.md` — loan approval system design
- `.system_design_practice/LoanApproval/agentic_banking_pipelines.md` — original blueprint for both problems
- `.system_design_practice/sd_template.md` — standard template for all future designs
- `basics/workflows/temporal_python_sdk.md` — Temporal Python SDK reference
