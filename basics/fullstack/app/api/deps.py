"""The WorkflowStarter abstraction the API layer depends on.

Production wires in TemporalWorkflowStarter; tests pass a fake. This keeps the API
unit tests free of any Temporal server.
"""

from __future__ import annotations

from typing import Protocol

from app.domain import Decision


class WorkflowStarter(Protocol):
    async def start_order(self, order_id: str) -> None: ...

    async def approve(self, order_id: str, decision: Decision) -> None: ...
