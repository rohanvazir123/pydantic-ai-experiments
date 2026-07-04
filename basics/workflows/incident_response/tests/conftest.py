"""Shared WorkflowEnvironment for incident response tests.

A single ephemeral Temporal server is started once per module and reused across
all tests, which avoids port-conflict races when tests spin up servers back-to-back.
"""
from __future__ import annotations

import pytest_asyncio
from temporalio.testing import WorkflowEnvironment


@pytest_asyncio.fixture(scope="module")
async def temporal_env() -> WorkflowEnvironment:  # type: ignore[misc]
    async with await WorkflowEnvironment.start_time_skipping() as env:
        yield env
