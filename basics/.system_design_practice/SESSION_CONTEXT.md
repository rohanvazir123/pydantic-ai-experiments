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
- `PIILeakageMetric` = built-in metric for PII in the explanation (LLM-judge —
  extracts statements, classifies each for PII; NOT a plain regex under the
  hood). A custom `BaseMetric` regex pre-filter for strictly-formatted PII
  (SSN, DOB) still earns its place alongside it — cheap, catches fixed-format
  cases before paying for a judge call.
- `EvaluationDataset` + `deepeval test run` in CI satisfies the append-only golden
  set + regression-gate requirement, for the explanation track specifically.
- **Latency is NOT a DeepEval concern** — corrected mid-session after checking
  the primary source (see verification note below). Measure with plain
  `time.perf_counter()` + a `pytest` threshold, not an LLM-eval-framework metric.

**Verification note (self-correction):** I initially cited a `LatencyMetric` for
DeepEval based on blog posts, without checking the primary source — wrong. GitHub
code search on confident-ai/deepeval found zero hits for `LatencyMetric` in the
actual metrics package (`deepeval/metrics/`, ~40 metric folders — no `latency`
one; the only repo hit was a stale 2024 changelog entry). Lesson: verify library
claims against the source repo/official docs, not secondary blog posts, before
they go in an interview-prep doc. `PIILeakageMetric`, by contrast, WAS verified
this way and is real.

**DeepEval's three metric tiers — pick per check, don't default to one.** Be ready
to name all three unprompted:
1. **Built-in default** (no config beyond a threshold) — `HallucinationMetric`,
   `BiasMetric`, `PIILeakageMetric`.
2. **`GEval`** (DeepEval's own definition of "custom metric") — natural-language
   rubric, auto-generated CoT, LLM-judged. Used for coherence/actionability —
   genuinely subjective, free-form rubric is the right tool.
3. **`DAGMetric`** (decision-tree LLM-as-judge, deterministic branching) — used
   for the prohibited-basis scan instead of GEval, on purpose: a compliance-
   critical check should be auditable/reproducible (see exactly which branch
   fired), not subject to free-form-rubric variance. Objective/checklist-like
   criteria → DAGMetric; subjective criteria → GEval.
4. **`BaseMetric`** (fully custom, subclassed, no LLM required) — used for a
   deterministic SSN/DOB regex pre-filter alongside `PIILeakageMetric`, and
   generally the right tier whenever a check doesn't need a judge call at all
   (non-LLM scorers, bounded pattern matches). Subclassing rather than a side
   script means it still runs inside the same `evaluate()` call and CI report as
   everything else.

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
