# app/temporal/

The Temporal orchestration layer.

## Table of Contents

- [Purpose](#purpose)
- [Files](#files)
- [Determinism boundary](#determinism-boundary)

## Purpose

Durable, crash-safe order processing: a workflow orchestrates the steps and
suspends for human approval; activities perform all I/O; the worker hosts both.

## Files

| File | Role |
|------|------|
| `workflow.py` | `OrderWorkflow` — deterministic orchestration; validate → HIL gate (signal + 2-day SLA timer) → confirm/reject. |
| `activities.py` | `OrderActivities` — I/O methods (`load_order`, `mark_status`) with an injected repository. |
| `client.py` | `connect()` + `TemporalWorkflowStarter` (starts workflows, sends approval signals). |
| `worker.py` | Worker entrypoint wiring Postgres + activities + workflow. |

## Determinism boundary

Workflow code is replayed from history, so it must be deterministic — no I/O, no
clocks, no randomness. Everything with a side effect is an **activity**. Pure
`domain` functions are safe to call directly from the workflow.
