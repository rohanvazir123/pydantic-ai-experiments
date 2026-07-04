# Test Output — Incident Response Workflow

**Runtime:** Pydantic AI 1.107.0 · Temporal 1.28.0 · Python 3.13.14  
**Mode:** `WorkflowEnvironment.start_time_skipping()` — no real Temporal server  
**LLM:** `FunctionModel`-style monkey-patch over `TestModel` — no real LLM calls

```
============================= test session starts ==============================
platform darwin -- Python 3.13.14, pytest-9.1.0, pluggy-1.6.0
asyncio: mode=Mode.STRICT

collected 5 items

tests/test_incident_workflow.py::test_happy_path_restart_resolves              PASSED
tests/test_incident_workflow.py::test_first_action_fails_second_resolves       PASSED
tests/test_incident_workflow.py::test_compensation_scale_up_worsens_then_clear_cache_resolves PASSED
tests/test_incident_workflow.py::test_escalation_after_all_actions_fail        PASSED
tests/test_incident_workflow.py::test_llm_reroutes_to_rollback                 PASSED

============================== 5 passed in 1.14s ===============================
```

---

## Scenario coverage

| Test | Scenario | Outcome |
|------|----------|---------|
| `test_happy_path_restart_resolves` | `restart_service` succeeds; LLM says resolved | `resolved=True`, 1 action, no compensations |
| `test_first_action_fails_second_resolves` | `scale_up` activity exhausts retries (ActivityError); LLM redirects to `clear_cache` | `resolved=True`, 2 actions (`scale_up` failed + `clear_cache` succeeded) |
| `test_compensation_scale_up_worsens_then_clear_cache_resolves` | `scale_up` succeeds but worsens error rate (0.45→0.60 > 0.45×1.2); compensation `scale_down` fires automatically; then `clear_cache` resolves | `resolved=True`, `compensations=["scale_down (compensating scale_up)"]` |
| `test_escalation_after_all_actions_fail` | All three actions fail; LLM eventually says escalate; `page_oncall` spy records the alert ID | `resolved=False`, `escalated=True`, `final_status="escalated"`, on-call paged |
| `test_llm_reroutes_to_rollback` | LLM triage suggests `restart_service`; after it barely helps, assessment redirects to `rollback_deployment` which resolves | `resolved=True`, both actions in `actions_taken`, `compensations=[]` |

---

## Key implementation notes

- `@workflow.defn(sandboxed=False)` — required because `beartype` (a pydantic-ai
  transitive dependency) triggers a circular import inside Temporal's sandbox runner.
  Marking the workflow unsandboxed is the standard fix when trusted third-party
  libraries conflict with the sandbox.
- Activity inputs are JSON strings throughout — clean Temporal serialization, easy
  to log and inspect.
- Multi-arg activities use dataclass wrappers (`AssessInput`, `PageInput`) as
  Temporal expects a single argument.
- Compensation is structural — workflow detects metric regression after `scale_up`
  and fires `scale_down` before proceeding to the next LLM assessment.
