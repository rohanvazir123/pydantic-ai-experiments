"""FastAPI app factory.

`create_app(repo, starter)` wires the store + workflow starter as explicit
dependencies (used by tests). For production, `create_app(lifespan=...)` builds them
on startup and stores them on `app.state` (see `app/api/asgi.py`). Routes read
`app.state.repo` / `app.state.starter` at request time so both paths work.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from app import domain
from app.api.deps import WorkflowStarter
from app.domain import OrderStatus
from app.models import ApprovalInput, Order, OrderInput, OrderResult
from app.store.base import OrderRepository

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "web" / "templates"

# SSE stream tuning: poll the store this often, for at most this many ticks
# (client's EventSource auto-reconnects afterward).
_SSE_INTERVAL_S = 1.0
_SSE_MAX_TICKS = 300
_TERMINAL = {OrderStatus.CONFIRMED, OrderStatus.REJECTED}

Lifespan = Callable[[FastAPI], AbstractAsyncContextManager[None]]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _build_order(inp: OrderInput) -> Order:
    errors = domain.validate_order(inp.item, inp.quantity, inp.unit_price_cents)
    if errors:
        raise HTTPException(status_code=422, detail={"errors": errors})
    total = domain.line_total_cents(inp.quantity, inp.unit_price_cents)
    return Order(
        id=uuid.uuid4().hex[:12],
        item=inp.item,
        quantity=inp.quantity,
        unit_price_cents=inp.unit_price_cents,
        total_cents=total,
        status=OrderStatus.PENDING,
        created_at=_now_iso(),
    )


def _sse(event: str, data: dict[str, str]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def create_app(
    repo: OrderRepository | None = None,
    starter: WorkflowStarter | None = None,
    *,
    lifespan: Lifespan | None = None,
) -> FastAPI:
    app = FastAPI(title="OrderFlow", lifespan=lifespan)
    app.state.repo = repo
    app.state.starter = starter
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    async def _create(inp: OrderInput) -> Order:
        r: OrderRepository = app.state.repo
        s: WorkflowStarter = app.state.starter
        order = _build_order(inp)
        await r.create(order)
        await s.start_order(order.id)
        return order

    # -------- JSON API --------

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/orders", status_code=202)
    async def create_order(inp: OrderInput) -> OrderResult:
        order = await _create(inp)
        return OrderResult(order_id=order.id, status=order.status)

    @app.get("/api/orders")
    async def list_orders() -> list[Order]:
        repo_: OrderRepository = app.state.repo
        return await repo_.list()

    @app.get("/api/orders/{order_id}")
    async def get_order(order_id: str) -> Order:
        repo_: OrderRepository = app.state.repo
        order = await repo_.get(order_id)
        if order is None:
            raise HTTPException(status_code=404, detail="order not found")
        return order

    @app.post("/api/orders/{order_id}/approval", status_code=202)
    async def approve_order(order_id: str, body: ApprovalInput) -> dict[str, str]:
        repo_: OrderRepository = app.state.repo
        starter_: WorkflowStarter = app.state.starter
        if await repo_.get(order_id) is None:
            raise HTTPException(status_code=404, detail="order not found")
        await starter_.approve(order_id, body.decision)
        return {"order_id": order_id, "decision": body.decision.value}

    @app.get("/api/orders/{order_id}/events")
    async def order_events(order_id: str) -> StreamingResponse:
        """Server-Sent Events: push status changes to the browser (no polling)."""
        repo_: OrderRepository = app.state.repo
        if await repo_.get(order_id) is None:
            raise HTTPException(status_code=404, detail="order not found")

        async def gen() -> AsyncIterator[bytes]:
            last: OrderStatus | None = None
            for _ in range(_SSE_MAX_TICKS):
                order = await repo_.get(order_id)
                if order is None:
                    yield _sse("error", {"detail": "order not found"})
                    return
                if order.status != last:
                    last = order.status
                    yield _sse("status", {"order_id": order_id, "status": order.status.value})
                    if order.status in _TERMINAL:
                        return
                await asyncio.sleep(_SSE_INTERVAL_S)

        return StreamingResponse(gen(), media_type="text/event-stream")

    # -------- Web (Jinja2) --------

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        repo_: OrderRepository = app.state.repo
        orders = await repo_.list()
        return templates.TemplateResponse(request, "index.html", {"orders": orders})

    @app.post("/orders")
    async def web_create(
        item: str = Form(...),
        quantity: int = Form(...),
        unit_price_cents: int = Form(...),
    ) -> RedirectResponse:
        order = await _create(
            OrderInput(item=item, quantity=quantity, unit_price_cents=unit_price_cents)
        )
        return RedirectResponse(url=f"/orders/{order.id}", status_code=303)

    @app.get("/orders/{order_id}", response_class=HTMLResponse)
    async def web_order(request: Request, order_id: str) -> HTMLResponse:
        repo_: OrderRepository = app.state.repo
        order = await repo_.get(order_id)
        if order is None:
            raise HTTPException(status_code=404, detail="order not found")
        return templates.TemplateResponse(request, "order.html", {"order": order})

    return app
