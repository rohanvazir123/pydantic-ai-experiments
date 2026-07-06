"""SQLModel OrderRepository — the production store (async SQLAlchemy on asyncpg).

The same implementation runs against Postgres (prod, `postgresql+asyncpg://…`) and
SQLite (`sqlite+aiosqlite://`, used by the repository's own tests). It implements
the `OrderRepository` Protocol, so the API and Temporal activities are unchanged.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.domain import OrderStatus
from app.models import Order


def create_engine(url: str) -> AsyncEngine:
    """Async engine. In-memory SQLite needs a single shared connection (StaticPool)."""
    if url.startswith("sqlite"):
        return create_async_engine(
            url, connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
    return create_async_engine(url)


async def init_db(engine: AsyncEngine) -> None:
    """Create tables from SQLModel metadata (prod uses Alembic; this is for bootstrap/tests)."""
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


class SqlModelOrderRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._session = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )

    async def create(self, order: Order) -> None:
        async with self._session() as session:
            session.add(order)
            await session.commit()

    async def get(self, order_id: str) -> Order | None:
        async with self._session() as session:
            return await session.get(Order, order_id)

    async def set_status(self, order_id: str, status: OrderStatus) -> None:
        async with self._session() as session:
            order = await session.get(Order, order_id)
            if order is None:
                raise KeyError(order_id)
            order.status = status
            await session.commit()

    async def list(self) -> list[Order]:
        async with self._session() as session:
            result = await session.exec(select(Order))
            return list(result.all())
