"""Shared WorkflowEnvironment for deployment saga tests."""
from __future__ import annotations

import pytest_asyncio
from temporalio.testing import WorkflowEnvironment


@pytest_asyncio.fixture(scope="module")
async def temporal_env() -> WorkflowEnvironment:  # type: ignore[misc]
    async with await WorkflowEnvironment.start_time_skipping() as env:
        yield env
