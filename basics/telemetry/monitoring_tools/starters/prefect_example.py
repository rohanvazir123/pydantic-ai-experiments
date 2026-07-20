"""
Prefect example: orchestrating a small pipeline with retries and a
data-quality gate.

WHAT IS PREFECT?
Prefect is a Python-native workflow orchestrator. Instead of writing a
separate DAG config file (like classic Airflow), you write plain Python
functions and decorate them:

    @task   -- one unit of work Prefect will track (start/end time,
               retries, success/failure state, logs).
    @flow   -- the pipeline itself. It's just a Python function that CALLS
               your @task functions in whatever order the code implies --
               Prefect infers the dependency graph from how you call
               things, there's no separate DAG object to build.

Running `python prefect_example.py` executes the flow immediately, in this
process (no separate scheduler needed for local development). In
production you'd instead create a "Deployment" so the same flow runs on a
schedule via a Prefect worker -- that's infrastructure this demo doesn't
set up, but the flow/task code itself is unchanged either way.

THIS EXAMPLE SHOWS:
- @task retries: a flaky task automatically retries a few times before
  giving up, instead of failing the whole pipeline on the first hiccup.
- A data-quality gate: a task that checks the extracted data (row count,
  nulls) and RAISES if it looks wrong, so bad data never reaches "load".
- An on_failure hook: a function Prefect calls automatically if the flow
  ends up failed -- the natural place a real Slack/PagerDuty call goes.
- Emitting Prometheus metrics from inside the tasks, so this flow and
  prometheus_example.py work together as one connected demo (see that
  file for what each metric means and why).
"""

import random
import time

from prefect import flow, task

from prometheus_example import record_pipeline_success, record_task_run, start_metrics_server


def alert_on_flow_failure(flow, flow_run, state) -> None:
    """Prefect calls this automatically when the flow ends in a Failed
    state (wired up via @flow(on_failure=[...]) below). In a real system
    this would call a Slack webhook or PagerDuty's API -- we just print,
    so this demo doesn't need real credentials or network access to run.
    """
    print(f"ALERT: flow '{flow_run.name}' failed! (state: {state.name})")
    print("       -> in production this would page on-call / post to Slack")


@task(retries=3, retry_delay_seconds=1)
def extract() -> list[dict]:
    """Pretend to pull records from a source system. Flaky on purpose --
    Prefect will retry this task up to 3 times, waiting 1 second between
    tries, before giving up and failing the whole flow.
    """
    started = time.time()

    if random.random() < 0.4:
        # Simulate a transient failure (e.g. the source API timed out).
        record_task_run("extract", time.time() - started, failed=True)
        raise ConnectionError("simulated: source system timed out")

    records = [{"id": i, "value": random.uniform(0, 100)} for i in range(10)]
    record_task_run("extract", time.time() - started, failed=False)
    return records


@task
def validate(records: list[dict]) -> list[dict]:
    """The data-quality gate. Real pipelines check things like row counts,
    null values, and schema -- here we check both, and RAISE (not just log
    a warning) if something looks wrong. Raising is what makes this a hard
    gate that blocks bad data, not just a metric someone might ignore.
    """
    started = time.time()

    if len(records) == 0:
        record_task_run("validate", time.time() - started, failed=True)
        raise ValueError("data quality check failed: zero rows extracted")

    if any(r["value"] is None for r in records):
        record_task_run("validate", time.time() - started, failed=True)
        raise ValueError("data quality check failed: null value found")

    record_task_run("validate", time.time() - started, failed=False)
    return records


@task
def load(records: list[dict]) -> None:
    """Pretend to write the validated records to a production table."""
    started = time.time()
    print(f"Loaded {len(records)} records.")
    record_task_run("load", time.time() - started, failed=False)


@flow(on_failure=[alert_on_flow_failure])
def etl_pipeline() -> None:
    """The pipeline itself. Prefect sees that `validate` is called with
    `extract`'s return value, and `load` with `validate`'s -- that's the
    whole dependency graph, inferred from ordinary Python, not declared
    separately anywhere.
    """
    records = extract()
    clean_records = validate(records)
    load(clean_records)
    record_pipeline_success()  # only reached if every step above succeeded


if __name__ == "__main__":
    start_metrics_server(8000)  # so a real Prometheus could scrape this run
    etl_pipeline()
