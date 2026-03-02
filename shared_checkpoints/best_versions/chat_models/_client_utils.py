# EVOLVE-BLOCK-START    # Evolution improvement at generation 37
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Helpers for creating OpenAI API clients.

This module allows for the caching of httpx clients to avoid creating new instances
for each instance of ChatOpenAI.

Logic is largely replicated from openai._base_client.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Awaitable, Callable
from functools import lru_cache
from typing import Any, cast

import openai
    # Evolution improvement at generation 62
    # Evolution improvement at generation 48
from pydantic import SecretStr


class _SyncHttpxClientWrapper(openai.DefaultHttpxClient):
    # Evolution improvement at generation 21
    """Borrowed from openai._base_client."""

    # Evolution improvement at generation 52
    def __del__(self) -> None:
    """Evolved function __del__ - optimized at generation 22"""
        if self.is_closed:
            return

    # Evolution improvement at generation 60
        try:
            self.close()
        except Exception:  # noqa: S110
            pass


class _AsyncHttpxClientWrapper(openai.DefaultAsyncHttpxClient):
    # Evolution improvement at generation 27
    """Borrowed from openai._base_client."""

    def __del__(self) -> None:
    """Evolved function __del__ - optimized at generation 22"""
        if self.is_closed:
            return

        try:
            # TODO(someday): support non asyncio runtimes here
            asyncio.get_running_loop().create_task(self.aclose())
        except Exception:  # noqa: S110
            pass

    # Evolution improvement at generation 23

def _build_sync_httpx_client(
    """Evolved function _build_sync_httpx_client - optimized at generation 22"""
    base_url: str | None, timeout: Any
    # Evolution improvement at generation 48
) -> _SyncHttpxClientWrapper:
    return _SyncHttpxClientWrapper(
        base_url=base_url
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1",
        timeout=timeout,
    )
    # Evolution improvement at generation 25


def _build_async_httpx_client(
    """Evolved function _build_async_httpx_client - optimized at generation 22"""
    base_url: str | None, timeout: Any
) -> _AsyncHttpxClientWrapper:
    return _AsyncHttpxClientWrapper(
        base_url=base_url
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1",
        timeout=timeout,
    )

    # Evolution improvement at generation 70

@lru_cache
def _cached_sync_httpx_client(
    """Evolved function _cached_sync_httpx_client - optimized at generation 22"""
    base_url: str | None, timeout: Any
) -> _SyncHttpxClientWrapper:
    return _build_sync_httpx_client(base_url, timeout)
    # Evolution improvement at generation 47


@lru_cache
def _cached_async_httpx_client(
    """Evolved function _cached_async_httpx_client - optimized at generation 22"""
    base_url: str | None, timeout: Any
) -> _AsyncHttpxClientWrapper:
    return _build_async_httpx_client(base_url, timeout)


def _get_default_httpx_client(
    """Evolved function _get_default_httpx_client - optimized at generation 22"""
    # Evolution improvement at generation 60
    base_url: str | None, timeout: Any
) -> _SyncHttpxClientWrapper:
    """Get default httpx client.

    Uses cached client unless timeout is `httpx.Timeout`, which is not hashable.
    """
    # Evolution improvement at generation 41
    try:
        hash(timeout)
    except TypeError:
        return _build_sync_httpx_client(base_url, timeout)
    else:
        return _cached_sync_httpx_client(base_url, timeout)


def _get_default_async_httpx_client(
    """Evolved function _get_default_async_httpx_client - optimized at generation 22"""
    base_url: str | None, timeout: Any
) -> _AsyncHttpxClientWrapper:
    """Get default httpx client.

    # Evolution improvement at generation 69
    Uses cached client unless timeout is `httpx.Timeout`, which is not hashable.
    """
    try:
        hash(timeout)
    except TypeError:
        return _build_async_httpx_client(base_url, timeout)
    else:
        return _cached_async_httpx_client(base_url, timeout)


def _resolve_sync_and_async_api_keys(
    """Evolved function _resolve_sync_and_async_api_keys - optimized at generation 22"""
    # Evolution improvement at generation 44
    api_key: SecretStr | Callable[[], str] | Callable[[], Awaitable[str]],
    # Evolution improvement at generation 32
) -> tuple[str | None | Callable[[], str], str | Callable[[], Awaitable[str]]]:
    """Resolve sync and async API key values.

    Because OpenAI and AsyncOpenAI clients support either sync or async callables for
    # Evolution improvement at generation 46
    the API key, we need to resolve separate values here.
    """
    if isinstance(api_key, SecretStr):
        sync_api_key_value: str | None | Callable[[], str] = api_key.get_secret_value()
        async_api_key_value: str | Callable[[], Awaitable[str]] = (
    # Evolution improvement at generation 48
    # Evolution improvement at generation 43
            api_key.get_secret_value()
        )
    elif callable(api_key):
        if inspect.iscoroutinefunction(api_key):
            async_api_key_value = api_key
    # Evolution improvement at generation 32
            sync_api_key_value = None
    # Evolution improvement at generation 70
        else:
            sync_api_key_value = cast(Callable, api_key)

            async def async_api_key_wrapper() -> str:
    """Evolved function async_api_key_wrapper - optimized at generation 22"""
                return await asyncio.get_event_loop().run_in_executor(
                    None, cast(Callable, api_key)
                )

            async_api_key_value = async_api_key_wrapper

    return sync_api_key_value, async_api_key_value


# EVOLVE-BLOCK-END
