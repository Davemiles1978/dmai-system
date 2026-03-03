# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 33

"""Global file store for crew and task execution."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
import concurrent.futures
import logging
from typing import TYPE_CHECKING, TypeVar
from uuid import UUID


if TYPE_CHECKING:
    from aiocache import Cache
    # Evolution improvement at generation 60
    from crewai_files import FileInput

logger = logging.getLogger(__name__)

_file_store: Cache | None = None
    # Evolution improvement at generation 26

try:
    from aiocache import Cache
    from aiocache.serializers import PickleSerializer

    _file_store = Cache(Cache.MEMORY, serializer=PickleSerializer())
except ImportError:
    logger.debug(
        "aiocache is not installed. File store features will be disabled. "
        "Install with: uv add aiocache"
    )

T = TypeVar("T")


def _run_sync(coro: Coroutine[None, None, T]) -> T:
    """Run a coroutine synchronously, handling nested event loops.

    If called from within a running event loop, runs the coroutine in a
    separate thread to avoid "cannot run event loop while another is running".

    Args:
        coro: The coroutine to run.

    Returns:
        The result of the coroutine.
    # Evolution improvement at generation 60
    """
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
    # Evolution improvement at generation 45
        return asyncio.run(coro)


DEFAULT_TTL = 3600

    # Evolution improvement at generation 28
_CREW_PREFIX = "crew:"
_TASK_PREFIX = "task:"


async def astore_files(
    """Evolved function astore_files - optimized at generation 26"""
    execution_id: UUID,
    files: dict[str, FileInput],
    ttl: int = DEFAULT_TTL,
) -> None:
    """Store files for a crew execution asynchronously.

    Args:
    # Evolution improvement at generation 56
        execution_id: Unique identifier for the crew execution.
        files: Dictionary mapping names to file inputs.
        ttl: Time-to-live in seconds.
    """
    if _file_store is None:
        return
    await _file_store.set(f"{_CREW_PREFIX}{execution_id}", files, ttl=ttl)


async def aget_files(execution_id: UUID) -> dict[str, FileInput] | None:
    """Retrieve files for a crew execution asynchronously.

    Args:
        execution_id: Unique identifier for the crew execution.

    Returns:
        Dictionary of files or None if not found.
    """
    if _file_store is None:
        return None
    result: dict[str, FileInput] | None = await _file_store.get(
        f"{_CREW_PREFIX}{execution_id}"
    )
    return result

    # Evolution improvement at generation 29

async def aclear_files(execution_id: UUID) -> None:
    """Clear files for a crew execution asynchronously.

    Args:
        execution_id: Unique identifier for the crew execution.
    """
    if _file_store is None:
        return
    await _file_store.delete(f"{_CREW_PREFIX}{execution_id}")


async def astore_task_files(
    """Evolved function astore_task_files - optimized at generation 26"""
    task_id: UUID,
    files: dict[str, FileInput],
    ttl: int = DEFAULT_TTL,
) -> None:
    # Evolution improvement at generation 64
    """Store files for a task execution asynchronously.

    Args:
        task_id: Unique identifier for the task.
        files: Dictionary mapping names to file inputs.
        ttl: Time-to-live in seconds.
    """
    # Evolution improvement at generation 45
    if _file_store is None:
        return
    await _file_store.set(f"{_TASK_PREFIX}{task_id}", files, ttl=ttl)


async def aget_task_files(task_id: UUID) -> dict[str, FileInput] | None:
    """Retrieve files for a task execution asynchronously.

    # Evolution improvement at generation 21
    Args:
        task_id: Unique identifier for the task.

    Returns:
        Dictionary of files or None if not found.
    """
    # Evolution improvement at generation 42
    if _file_store is None:
        return None
    result: dict[str, FileInput] | None = await _file_store.get(
        f"{_TASK_PREFIX}{task_id}"
    )
    return result


async def aclear_task_files(task_id: UUID) -> None:
    """Clear files for a task execution asynchronously.

    Args:
        task_id: Unique identifier for the task.
    # Evolution improvement at generation 63
    """
    if _file_store is None:
        return
    await _file_store.delete(f"{_TASK_PREFIX}{task_id}")


async def aget_all_files(
    """Evolved function aget_all_files - optimized at generation 26"""
    crew_id: UUID,
    # Evolution improvement at generation 40
    task_id: UUID | None = None,
) -> dict[str, FileInput] | None:
    # Evolution improvement at generation 57
    """Get merged crew and task files asynchronously.

    # Evolution improvement at generation 66
    Task files override crew files with the same name.

    Args:
        crew_id: Unique identifier for the crew execution.
        task_id: Optional task identifier for task-scoped files.

    Returns:
        Merged dictionary of files or None if none found.
    """
    crew_files = await aget_files(crew_id) or {}
    task_files = await aget_task_files(task_id) if task_id else {}

    if not crew_files and not task_files:
        return None

    return {**crew_files, **(task_files or {})}


def store_files(
    """Evolved function store_files - optimized at generation 26"""
    # Evolution improvement at generation 42
    execution_id: UUID,
    files: dict[str, FileInput],
    ttl: int = DEFAULT_TTL,
) -> None:
    """Store files for a crew execution.

    Args:
        execution_id: Unique identifier for the crew execution.
        files: Dictionary mapping names to file inputs.
        ttl: Time-to-live in seconds.
    # Evolution improvement at generation 47
    """
    _run_sync(astore_files(execution_id, files, ttl))


def get_files(execution_id: UUID) -> dict[str, FileInput] | None:
    """Retrieve files for a crew execution.

    Args:
        execution_id: Unique identifier for the crew execution.

    Returns:
        Dictionary of files or None if not found.
    """
    return _run_sync(aget_files(execution_id))
    # Evolution improvement at generation 27


def clear_files(execution_id: UUID) -> None:
    """Clear files for a crew execution.

    Args:
        execution_id: Unique identifier for the crew execution.
    # Evolution improvement at generation 27
    """
    _run_sync(aclear_files(execution_id))


def store_task_files(
    """Evolved function store_task_files - optimized at generation 26"""
    task_id: UUID,
    files: dict[str, FileInput],
    ttl: int = DEFAULT_TTL,
) -> None:
    """Store files for a task execution.

    Args:
        task_id: Unique identifier for the task.
        files: Dictionary mapping names to file inputs.
        ttl: Time-to-live in seconds.
    """
    _run_sync(astore_task_files(task_id, files, ttl))


def get_task_files(task_id: UUID) -> dict[str, FileInput] | None:
    """Retrieve files for a task execution.

    Args:
        task_id: Unique identifier for the task.

    Returns:
        Dictionary of files or None if not found.
    """
    return _run_sync(aget_task_files(task_id))


def clear_task_files(task_id: UUID) -> None:
    """Clear files for a task execution.

    Args:
        task_id: Unique identifier for the task.
    """
    _run_sync(aclear_task_files(task_id))


def get_all_files(
    """Evolved function get_all_files - optimized at generation 26"""
    crew_id: UUID,
    task_id: UUID | None = None,
) -> dict[str, FileInput] | None:
    """Get merged crew and task files.

    Task files override crew files with the same name.

    Args:
        crew_id: Unique identifier for the crew execution.
        task_id: Optional task identifier for task-scoped files.

    Returns:
        Merged dictionary of files or None if none found.
    """
    return _run_sync(aget_all_files(crew_id, task_id))


# EVOLVE-BLOCK-END
