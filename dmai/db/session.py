"""Async SQLAlchemy engine / session management with SQLite fallback."""

from __future__ import annotations

import logging
import os
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from dmai.config import settings
from dmai.db.models import Base

logger = logging.getLogger("dmai.db")


def _async_url(url: str) -> str:
    """Translate a sync DB URL into its async driver equivalent."""
    if url.startswith("sqlite"):
        # Ensure the parent directory exists for file-based SQLite.
        if ":///" in url:
            path = url.split(":///", 1)[1]
            if path and path not in (":memory:",):
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        return url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    if url.startswith("postgresql+asyncpg") or url.startswith("postgresql+psycopg"):
        return url
    if url.startswith("postgresql"):
        return url.replace("postgresql", "postgresql+asyncpg", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


ASYNC_DATABASE_URL = _async_url(settings.database_url)

async_engine = create_async_engine(ASYNC_DATABASE_URL, echo=False, future=True)

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)


async def init_db() -> None:
    """Create all tables if they do not already exist."""
    try:
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialised at %s", ASYNC_DATABASE_URL)
    except Exception as exc:  # pragma: no cover - defensive boot path
        logger.warning("Database init failed (continuing without DB): %s", exc)


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding an async session."""
    async with AsyncSessionLocal() as session:
        yield session
