"""API tests via FastAPI TestClient with a fake WorkflowStarter (no Temporal server)."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.domain import Decision, OrderStatus
from app.models import Order
from app.store.memory import InMemoryOrderRepository


class FakeStarter:
    """Records start/approve calls so API behavior can be asserted in isolation."""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.approvals: list[tuple[str, Decision]] = []

    async def start_order(self, order_id: str) -> None:
        self.started.append(order_id)

    async def approve(self, order_id: str, decision: Decision) -> None:
        self.approvals.append((order_id, decision))


@pytest.fixture
def client() -> tuple[TestClient, InMemoryOrderRepository, FakeStarter]:
    repo = InMemoryOrderRepository()
    starter = FakeStarter()
    return TestClient(create_app(repo, starter)), repo, starter


def test_health(client: tuple[TestClient, InMemoryOrderRepository, FakeStarter]) -> None:
    tc, _, _ = client
    assert tc.get("/health").json() == {"status": "ok"}


def test_create_order_starts_workflow(
    client: tuple[TestClient, InMemoryOrderRepository, FakeStarter],
) -> None:
    tc, repo, starter = client
    resp = tc.post("/api/orders", json={"item": "Widget", "quantity": 2, "unit_price_cents": 500})
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == OrderStatus.PENDING.value
    assert starter.started == [body["order_id"]]


def test_create_order_validation_error(
    client: tuple[TestClient, InMemoryOrderRepository, FakeStarter],
) -> None:
    tc, _, _ = client
    resp = tc.post("/api/orders", json={"item": "Widget", "quantity": 0, "unit_price_cents": 500})
    assert resp.status_code == 422  # pydantic rejects quantity <= 0


def test_get_order(client: tuple[TestClient, InMemoryOrderRepository, FakeStarter]) -> None:
    tc, _, _ = client
    order_id = tc.post(
        "/api/orders", json={"item": "Widget", "quantity": 1, "unit_price_cents": 100}
    ).json()["order_id"]
    resp = tc.get(f"/api/orders/{order_id}")
    assert resp.status_code == 200
    assert resp.json()["item"] == "Widget"


def test_get_order_404(client: tuple[TestClient, InMemoryOrderRepository, FakeStarter]) -> None:
    tc, _, _ = client
    assert tc.get("/api/orders/missing").status_code == 404


def test_list_orders(client: tuple[TestClient, InMemoryOrderRepository, FakeStarter]) -> None:
    tc, _, _ = client
    tc.post("/api/orders", json={"item": "A", "quantity": 1, "unit_price_cents": 100})
    tc.post("/api/orders", json={"item": "B", "quantity": 1, "unit_price_cents": 100})
    assert len(tc.get("/api/orders").json()) == 2


def test_approval_signals_starter(
    client: tuple[TestClient, InMemoryOrderRepository, FakeStarter],
) -> None:
    tc, _, starter = client
    order_id = tc.post(
        "/api/orders", json={"item": "Big", "quantity": 1, "unit_price_cents": 200000}
    ).json()["order_id"]
    resp = tc.post(f"/api/orders/{order_id}/approval", json={"decision": "approved"})
    assert resp.status_code == 202
    assert starter.approvals == [(order_id, Decision.APPROVED)]


def test_approval_404(client: tuple[TestClient, InMemoryOrderRepository, FakeStarter]) -> None:
    tc, _, _ = client
    resp = tc.post("/api/orders/missing/approval", json={"decision": "approved"})
    assert resp.status_code == 404


def test_web_index(client: tuple[TestClient, InMemoryOrderRepository, FakeStarter]) -> None:
    tc, _, _ = client
    resp = tc.get("/")
    assert resp.status_code == 200
    assert "OrderFlow" in resp.text


def test_web_form_create_redirects(
    client: tuple[TestClient, InMemoryOrderRepository, FakeStarter],
) -> None:
    tc, _, starter = client
    resp = tc.post(
        "/orders",
        data={"item": "Widget", "quantity": "1", "unit_price_cents": "100"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/orders/")
    assert len(starter.started) == 1


def test_web_order_page(client: tuple[TestClient, InMemoryOrderRepository, FakeStarter]) -> None:
    tc, _, _ = client
    order_id = tc.post(
        "/api/orders", json={"item": "Gadget", "quantity": 1, "unit_price_cents": 100}
    ).json()["order_id"]
    resp = tc.get(f"/orders/{order_id}")
    assert resp.status_code == 200
    assert "Gadget" in resp.text


# -- SSE (async, single event loop via httpx ASGI transport) --


async def test_sse_streams_terminal_status() -> None:
    repo = InMemoryOrderRepository()
    app = create_app(repo, FakeStarter())
    await repo.create(
        Order(id="s1", item="W", quantity=1, unit_price_cents=100, total_cents=100,
              status=OrderStatus.CONFIRMED, created_at="t")
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        async with ac.stream("GET", "/api/orders/s1/events") as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            body = "".join([chunk async for chunk in resp.aiter_text()])
    assert "event: status" in body
    assert "confirmed" in body


async def test_sse_404_for_missing_order() -> None:
    app = create_app(InMemoryOrderRepository(), FakeStarter())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/orders/missing/events")
    assert resp.status_code == 404
