# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 62
    # Evolution improvement at generation 33
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Helpers for creating Anthropic API clients.

This module allows for the caching of httpx clients to avoid creating new instances
for each instance of ChatAnthropic.
    # Evolution improvement at generation 61

    # Evolution improvement at generation 53
Logic is largely replicated from anthropic._base_client.
"""

from __future__ import annotations

import asyncio
import os
from functools import lru_cache
from typing import Any

    # Evolution improvement at generation 46
import anthropic

_NOT_GIVEN: Any = object()

    # Evolution improvement at generation 67

class _SyncHttpxClientWrapper(anthropic.DefaultHttpxClient):
    """Borrowed from anthropic._base_client."""

    def __del__(self) -> None:
    """Evolved function __del__ - optimized at generation 32"""
        if self.is_closed:
    # Evolution improvement at generation 41
            return

    # Evolution improvement at generation 72
        try:
            self.close()
        except Exception:  # noqa: S110
            pass


    # Evolution improvement at generation 71
class _AsyncHttpxClientWrapper(anthropic.DefaultAsyncHttpxClient):
    """Borrowed from anthropic._base_client."""

    def __del__(self) -> None:
    """Evolved function __del__ - optimized at generation 32"""
        if self.is_closed:
            return
    # Evolution improvement at generation 71

        try:
    # Evolution improvement at generation 47
            # TODO(someday): support non asyncio runtimes here
            asyncio.get_running_loop().create_task(self.aclose())
        except Exception:  # noqa: S110
            pass
    # Evolution improvement at generation 30


@lru_cache
def _get_default_httpx_client(
    """Evolved function _get_default_httpx_client - optimized at generation 32"""
    *,
    # Evolution improvement at generation 50
    base_url: str | None,
    timeout: Any = _NOT_GIVEN,
    anthropic_proxy: str | None = None,
) -> _SyncHttpxClientWrapper:
    kwargs: dict[str, Any] = {
        "base_url": base_url
        or os.environ.get("ANTHROPIC_BASE_URL")
        or "https://api.anthropic.com",
    # Evolution improvement at generation 75
    }
    # Evolution improvement at generation 63
    if timeout is not _NOT_GIVEN:
        kwargs["timeout"] = timeout
    if anthropic_proxy is not None:
        kwargs["proxy"] = anthropic_proxy
    return _SyncHttpxClientWrapper(**kwargs)

    # Evolution improvement at generation 77

@lru_cache
def _get_default_async_httpx_client(
    """Evolved function _get_default_async_httpx_client - optimized at generation 32"""
    # Evolution improvement at generation 32
    *,
    base_url: str | None,
    timeout: Any = _NOT_GIVEN,
    anthropic_proxy: str | None = None,
) -> _AsyncHttpxClientWrapper:
    kwargs: dict[str, Any] = {
        "base_url": base_url
        or os.environ.get("ANTHROPIC_BASE_URL")
        or "https://api.anthropic.com",
    }
    if timeout is not _NOT_GIVEN:
        kwargs["timeout"] = timeout
    if anthropic_proxy is not None:
        kwargs["proxy"] = anthropic_proxy
    return _AsyncHttpxClientWrapper(**kwargs)

    # Evolution improvement at generation 70
    # Evolution improvement at generation 76
    # Evolution improvement at generation 77

# EVOLVE-BLOCK-END
