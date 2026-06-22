"""Alembic migration environment for DMAI.

Uses the application's settings and ORM metadata. Async engines are handled by
running migrations through a sync wrapper. This file is import-safe even when
Alembic is not installed (it simply won't be used in that case).
"""

from __future__ import annotations

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from dmai.config import settings
from dmai.db.models import Base

try:
    from alembic import context
except ImportError:  # pragma: no cover - alembic optional at runtime
    context = None  # type: ignore

target_metadata = Base.metadata


def _sync_url() -> str:
    """Return a synchronous DB URL for Alembic."""
    url = settings.database_url
    if url.startswith("sqlite+aiosqlite"):
        return url.replace("sqlite+aiosqlite", "sqlite", 1)
    if "+asyncpg" in url:
        return url.replace("+asyncpg", "", 1)
    return url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL without a DB connection)."""
    context.configure(
        url=_sync_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode against a live DB connection."""
    connectable = engine_from_config(
        {"sqlalchemy.url": _sync_url()},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context is not None:
    if hasattr(context.config, "config_file_name") and context.config.config_file_name:
        fileConfig(context.config.config_file_name)
    if context.is_offline_mode():
        run_migrations_offline()
    else:
        run_migrations_online()
