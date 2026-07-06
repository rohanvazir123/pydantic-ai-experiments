"""Shared fixtures: in-memory repo + a Temporal time-skipping test environment."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment

from app.store.memory import InMemoryOrderRepository


@pytest.fixture
def repo() -> InMemoryOrderRepository:
    return InMemoryOrderRepository()


@pytest_asyncio.fixture
async def temporal_env() -> AsyncIterator[WorkflowEnvironment]:
    env = await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    )
    try:
        yield env
    finally:
        await env.shutdown()
