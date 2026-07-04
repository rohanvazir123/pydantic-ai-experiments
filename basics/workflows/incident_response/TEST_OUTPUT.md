# Incident Response Workflow — Temporal + Pydantic AI

## Running the examples

### Tests (no Temporal server, no LLM required)

```bash
# from repo root
uv run python -m pytest basics/workflows/incident_response/tests/ basics/workflows/deployment_saga/tests/ -v
```

Uses `WorkflowEnvironment.start_time_skipping()` (in-process Temporal) and
`TestModel` (no real LLM). Completes in ~1–2s.

### Live run (Temporal server + Ollama required)

```bash
# 1. Start Temporal dev server (separate terminal)
temporal server start-dev

# 2. Ensure Ollama is running with the model pulled
ollama serve
ollama pull qwen2.5:14b

# 3. Run the workflow (worker + execution in one process)
uv run python -m basics.workflows.incident_response.run_live

# Run with a different model
AGENT_LARGE_MODEL=qwen2.5:7b uv run python -m basics.workflows.incident_response.run_live

# Run worker only (then trigger via Temporal UI or CLI in another terminal)
uv run python -m basics.workflows.incident_response.run_live --worker-only
```

---

## Test Output

**Runtime:** Pydantic AI 1.107.0 · Temporal 1.28.0 · Python 3.13.14  
**Mode:** `WorkflowEnvironment.start_time_skipping()` — no real Temporal server  
**LLM:** `FunctionModel`-style monkey-patch over `TestModel` — no real LLM calls

```
============================= test session starts ==============================
platform darwin -- Python 3.13.14, pytest-9.1.0, pluggy-1.6.0
asyncio: mode=Mode.STRICT

collected 18 items

basics/workflows/deployment_saga/tests/test_deployment_saga.py::test_happy_path_all_stages_succeed PASSED
basics/workflows/deployment_saga/tests/test_deployment_saga.py::test_llm_nogo_rollbacks_staging_and_resources PASSED
basics/workflows/deployment_saga/tests/test_deployment_saga.py::test_staging_fails_rollbacks_only_resources PASSED
basics/workflows/deployment_saga/tests/test_deployment_saga.py::test_production_fails_rollbacks_staging_and_resources PASSED
basics/workflows/deployment_saga/tests/test_deployment_saga.py::test_dns_fails_full_three_stage_rollback PASSED
basics/workflows/deployment_saga/tests/test_deployment_saga.py::test_provision_fails_no_compensations PASSED
basics/workflows/deployment_saga/tests/test_deployment_saga.py::test_dns_fails_rollback_order_is_exactly_reversed PASSED
basics/workflows/incident_response/tests/test_incident_workflow.py::test_happy_path_restart_resolves PASSED
basics/workflows/incident_response/tests/test_incident_workflow.py::test_first_action_fails_second_resolves PASSED
basics/workflows/incident_response/tests/test_incident_workflow.py::test_compensation_scale_up_worsens_then_clear_cache_resolves PASSED
basics/workflows/incident_response/tests/test_incident_workflow.py::test_escalation_after_all_actions_fail PASSED
basics/workflows/incident_response/tests/test_incident_workflow.py::test_llm_reroutes_to_rollback PASSED
basics/workflows/incident_response/tests/test_incident_workflow.py::test_saga_chain_fires_on_llm_escalation PASSED
basics/workflows/incident_response/tests/test_incident_workflow.py::test_saga_chain_fires_on_queue_exhaustion PASSED
basics/workflows/incident_response/tests/test_incident_workflow.py::test_triage_agent_has_investigation_tools PASSED
basics/workflows/incident_response/tests/test_incident_workflow.py::test_self_heal_sequence_triggers_after_action_failure PASSED
basics/workflows/incident_response/tests/test_incident_workflow.py::test_self_heal_only_fires_once PASSED
basics/workflows/incident_response/tests/test_incident_workflow.py::test_self_heal_not_triggered_for_heal_sequence_actions PASSED

============================== 18 passed in 0.87s ==============================
```

---

## Scenario coverage

### Incident Response (11 tests)

| Test | Scenario | Outcome |
|------|----------|---------|
| `test_happy_path_restart_resolves` | `restart_service` succeeds; LLM says resolved | `resolved=True`, 1 action, no compensations |
| `test_first_action_fails_second_resolves` | `scale_up` activity exhausts retries (ActivityError); LLM redirects to `clear_cache` | `resolved=True`, 2 actions (`scale_up` failed + `clear_cache` succeeded) |
| `test_compensation_scale_up_worsens_then_clear_cache_resolves` | `scale_up` succeeds but worsens error rate (0.45→0.60 > 0.45×1.2); compensation `scale_down` fires automatically; then `clear_cache` resolves | `resolved=True`, `compensations=["scale_down (compensating scale_up)"]` |
| `test_escalation_after_all_actions_fail` | All three actions fail; LLM eventually says escalate; `page_oncall` spy records the alert ID | `resolved=False`, `escalated=True`, `final_status="escalated"`, on-call paged |
| `test_llm_reroutes_to_rollback` | LLM triage suggests `restart_service`; after it barely helps, assessment redirects to `rollback_deployment` which resolves | `resolved=True`, both actions in `actions_taken`, `compensations=[]` |
| `test_saga_chain_fires_on_llm_escalation` | `scale_up` enters saga chain; LLM escalates on first assessment → `scale_down` compensation fires before `page_oncall` | `escalated=True`, `compensations=["scale_down (compensating scale_up)"]` |
| `test_saga_chain_fires_on_queue_exhaustion` | `scale_up` enters saga chain; action queue empties after assessment (no next_action) → loop exits → compensation fires | `final_status="escalated_max_actions"`, `compensations=["scale_down (compensating scale_up)"]` |
| `test_triage_agent_has_investigation_tools` | Unit check — triage agent wired with ≥4 tools (runbook + metrics + deployments + deps); assess agent wired with ≥1 (current metrics) | Tool registration verified; no Temporal needed |
| `test_self_heal_sequence_triggers_after_action_failure` | `scale_up` fails → self-heal prepends `[clear_cache, restart_service]`; `clear_cache` resolves | `resolved=True`, `escalated=False`; self-heal prevented escalation |
| `test_self_heal_only_fires_once` | `scale_up` fails (heal fires); `clear_cache` (from heal) also fails (no second heal); `restart_service` (from heal) succeeds | `resolved=True`; exactly 1 `clear_cache` in `actions_taken` |
| `test_self_heal_not_triggered_for_heal_sequence_actions` | `clear_cache` (in `SELF_HEAL_SEQUENCE`) fails → no heal triggered (avoids loop); LLM escalates | `escalated=True`; exactly 1 `clear_cache` in `actions_taken` |

### Deployment Saga (7 tests)

| Test | Scenario | Outcome |
|------|----------|---------|
| `test_happy_path_all_stages_succeed` | All 5 stages succeed; LLM says proceed | `succeeded=True`, 5 completed stages, no compensations |
| `test_llm_nogo_rollbacks_staging_and_resources` | LLM says no-go after integration tests | `aborted_at="evaluate_test_results"`, compensations: `[undeploy_staging, deprovision_resources]` |
| `test_staging_fails_rollbacks_only_resources` | `deploy_to_staging` raises | `aborted_at="deploy_to_staging"`, compensations: `[deprovision_resources]` |
| `test_production_fails_rollbacks_staging_and_resources` | `deploy_to_production` raises | compensations: `[undeploy_staging, deprovision_resources]` |
| `test_dns_fails_full_three_stage_rollback` | `update_dns` raises after all prior stages succeed | compensations: `[undeploy_production, undeploy_staging, deprovision_resources]` |
| `test_provision_fails_no_compensations` | First stage fails immediately | `completed_stages=[]`, `compensations_run=[]` |
| `test_dns_fails_rollback_order_is_exactly_reversed` | `update_dns` fails; spy activities record call order | call_order verified == `[undeploy_production, undeploy_staging, deprovision_resources]` |

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
- `COMPENSATIONS` dict + `_run_compensation_chain` — the saga pattern: each successful
  state-changing action is pushed onto a chain as `(action, compensation)`. On abort,
  the chain runs in exact reverse order. Actions with `comp=None` are stateless/safe
  and never enter the chain (e.g. `clear_cache`, `run_integration_tests`).
- Module-scoped `temporal_env` fixture — one ephemeral Temporal server per test module
  avoids port-conflict races when tests start servers back-to-back. Each test still
  creates its own `Worker` with specific activities.
- `DeploymentSagaWorkflow` (`deployment_saga/`) — a second example demonstrating the
  same saga pattern in a sequential 5-stage deployment pipeline with an LLM go/no-go
  gate after integration tests.
