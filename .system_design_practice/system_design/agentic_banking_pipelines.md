# Agentic Banking Pipelines — Basic Design

Two system design problems for an agentic AI platform in banking/financial services.
Neither requires full autonomy — the model earns its place at specific bounded
decision points. Code controls outer flow, thresholds, and compliance gates.

---

## Design 1: Multi-Step Risk-Aware Loan Application Router

### Core principle
Most of the flow is known in advance (credit check, income verification, document
validation, compliance rules) → L2 code-controlled DAG for the outer loop. The model
handles interpretation at specific points, not the routing decision itself.

### Architecture

```
Application intake
  │
  ├─ [parallel, code-dispatched]
  │   ├─ Identity verification  (L1 — structured output against ID docs)
  │   ├─ Credit pull            (L1 — score + tradeline summary)
  │   └─ Document extraction    (L1 — income/employment from uploaded docs)
  │
  ├─ Risk synthesis agent       (L3 — tool-calling)
  │   tools: get_dti_ratio, check_regulatory_rules(state, loan_type),
  │           flag_anomalies, lookup_fraud_indicators
  │   output: RiskDecision { score, tier, flags, explanation }
  │
  └─ Router                     (code)
      ├─ score ≥ threshold → auto-approve lane
      ├─ score in gray zone → human underwriter queue
      └─ hard fail (fraud flag, sanctions hit) → deny + compliance log
```

### Key tradeoffs (surface proactively)

- **Determinism vs flexibility.** Regulators need auditable, reproducible decisions.
  Routing thresholds live in code, not the model. Model handles interpretation only
  (self-employed income, non-standard employment structures).
- **Parallel vs sequential checks.** Identity + credit + document extraction are
  genuinely independent — run in parallel. Risk synthesis waits on all three.
  Real latency win because independence is real, not assumed.
- **Idempotency.** Retry must not trigger a second credit hard inquiry.
  Key = `application_id + check_type`.
- **Human-in-the-loop gate.** Gray-zone cases go to a human underwriter with the
  model's risk explanation pre-populated — not a blank queue item.

---

## Design 2: Automated Transaction Dispute Pipeline

### Core principle
Closer to L3/L4 than the loan router — dispute investigation is less predictable.
The agent may need to look at transaction history, merchant patterns, and fraud
signals before knowing what evidence matters. But the tool set is still fixed and
bounded; the model sequences investigation, not invents it.

### Architecture

```
Dispute filed
  │
  ├─ Eligibility check          (L1 — is this dispute type/window valid?)
  │
  ├─ Investigation agent        (L3 — tool-calling)
  │   tools: get_transaction_detail, get_merchant_history,
  │           check_prior_disputes(customer), flag_fraud_pattern,
  │           lookup_dispute_policy(transaction_type)
  │   output: DisputeFindings { evidence, confidence, recommended_action }
  │
  ├─ Resolution decision        (code + threshold)
  │   ├─ high confidence + policy clear → auto-resolve
  │   ├─ low confidence / high value → human review queue
  │   └─ fraud signal → freeze + escalate
  │
  └─ Provisional credit         (immediate, before investigation closes)
      + customer notification
```

### Key tradeoffs (surface proactively)

- **Regulatory timing constraints.** Provisional credit within 5 business days;
  full resolution within 45/90 days depending on case type. Pipeline must track
  deadlines per case, not just run to completion.
- **Bounded investigation.** Fixed tool set — the model cannot invent new tools or
  access systems outside the defined set. Intentional blast-radius control.
- **Confidence threshold calibration.** Auto-resolve only if confidence scores are
  validated against historical outcomes. Needs an offline eval loop.
- **Idempotency on credit issuance.** Double credit is worse than a delay.
  Key = `dispute_id + action_type`.

---

## The thread connecting both designs

> The value of the model is interpretation and synthesis at specific bounded points —
> not controlling the outer flow. Code controls routing, thresholds, and compliance
> gates. The model handles the parts where rules alone are insufficient.

Neither design needs full autonomy. Raising this proactively is the engineering
judgment that matters.

---

## Context for next session — planned deep dives

The user will drill first, then we go through together. Areas to be ready for:

- **Failure modes** — what breaks in each pipeline and why
- **Evaluation & monitoring** — how you validate model confidence scores against
  real outcomes; offline eval loop design; what metrics matter
- **Model upgrade safety** — how to upgrade the underlying model without breaking
  compliance properties or shifting decision boundaries silently
- **Latency optimization** — beyond parallel checks; where the bottlenecks actually are
- **Regulatory edge cases** — how the pipeline handles ambiguous cases that sit
  on the boundary of automated vs human decision
- **Observability** — what you instrument, what you alert on, what a production
  incident looks like in these pipelines
