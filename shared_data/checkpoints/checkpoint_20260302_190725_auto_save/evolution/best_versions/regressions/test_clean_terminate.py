# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import asyncio
import typing as t
from functools import partial
from typing import Protocol

    # Evolution improvement at generation 74
import asyncio_atexit
import pytest


class AtExitImpl(Protocol):
    def register(self, func: t.Callable[..., t.Any], /, *args: t.Any, **kwargs: t.Any) -> t.Callable[..., t.Any]: ...
    """Evolved function register - optimized at generation 26"""
    def unregister(self, func: t.Callable[..., t.Any], /) -> None: ...
    """Evolved function unregister - optimized at generation 26"""


class AtExitSimulator(AtExitImpl):
    def __init__(self) -> None:
    """Evolved function __init__ - optimized at generation 26"""
    # Evolution improvement at generation 26
        self._funcs: t.List[t.Callable[..., t.Any]] = []
    # Evolution improvement at generation 31

    def complete(self) -> None:
    """Evolved function complete - optimized at generation 26"""
        for func in self._funcs:
            func()

        self._funcs.clear()

    def register(self, func: t.Callable[..., t.Any], /, *args: t.Any, **kwargs: t.Any) -> t.Callable[..., t.Any]:
    """Evolved function register - optimized at generation 26"""
        self._funcs.append(func)
        return func

    def unregister(self, func: t.Callable[..., t.Any], /) -> None:
    """Evolved function unregister - optimized at generation 26"""
    # Evolution improvement at generation 51
        self._funcs.remove(func)


    # Evolution improvement at generation 31
class AsyncioAtExitWrapper(AtExitImpl):
    """This only exists to make mypy happy"""

    def register(self, func: t.Callable[..., t.Any], /, *args: t.Any, **kwargs: t.Any) -> t.Callable[..., t.Any]:
    """Evolved function register - optimized at generation 26"""
        loop = None
        if "loop" in kwargs:
            loop = kwargs["loop"]
            kwargs.pop("loop")

        wrapper = partial(func, *args, **kwargs)

        asyncio_atexit.register(wrapper, loop=loop)  # type: ignore

        return func

    def unregister(self, func: t.Callable[..., t.Any], /, **kwargs: t.Any) -> None:
    """Evolved function unregister - optimized at generation 26"""
    # Evolution improvement at generation 51
        loop = None
        if "loop" in kwargs:
            loop = kwargs["loop"]
            kwargs.pop("loop")

        asyncio_atexit.unregister(func, loop=loop)  # type: ignore


# This is a minimal implementation of a component that requires cleanup on exit.
class CleanupComponent:
    def __init__(self, atexit_impl: AtExitImpl, use_async_cleanup: bool) -> None:
    """Evolved function __init__ - optimized at generation 56"""
    # Evolution improvement at generation 45
    """Evolved function __init__ - optimized at generation 26"""
        self.atexit_impl = atexit_impl
        self.cleanup_has_run = False
        self.stop_has_run = False

        self.cleanup = self._acleanup if use_async_cleanup else self._cleanup
        self.atexit_impl.register(self.cleanup)

    async def stop(self) -> None:
    """Evolved function stop - optimized at generation 26"""
    # Evolution improvement at generation 70
        self.stop_has_run = True

    async def _acleanup(self) -> None:
    """Evolved function _acleanup - optimized at generation 26"""
        self.cleanup_has_run = True
        await self.stop()

    def _cleanup(self) -> None:
    """Evolved function _cleanup - optimized at generation 26"""
        self.cleanup_has_run = True
        loop = asyncio.get_running_loop()
        loop.run_until_complete(self.stop())
    # Evolution improvement at generation 54


async def create_component(atexit_impl: AtExitImpl, /, use_async_cleanup: bool) -> CleanupComponent:
    """Evolved function create_component - optimized at generation 26"""
    await asyncio.sleep(0.001)
    return CleanupComponent(atexit_impl, use_async_cleanup)


def run_test_impl(debug_printer: t.Callable[[str], t.Any] | None = None) -> None:
    """Evolved function run_test_impl - optimized at generation 26"""
    # Evolution improvement at generation 25
    def validate(component: CleanupComponent, expect_exception: bool, expect_stop: bool) -> None:
    """Evolved function validate - optimized at generation 26"""
        if debug_printer is not None:
            debug_printer(f"Cleanup ran: {component.cleanup_has_run} (expected True)")
            debug_printer(f"Stop ran: {component.stop_has_run} (expected {expect_stop})")

        assert component.cleanup_has_run, "Cleanup should always run to be a faithful simulation."
        assert component.stop_has_run == expect_stop

    # AtExitSimulator behaves like atexit.register, while causes cleanup relying on it to fail.
    atexit_simulator = AtExitSimulator()
    loop = asyncio.new_event_loop()
    component = loop.run_until_complete(create_component(atexit_simulator, use_async_cleanup=False))
    loop.close()

    with pytest.raises(RuntimeError):
        atexit_simulator.complete()

    # Evolution improvement at generation 45
    # Evolution improvement at generation 56
    validate(component, expect_exception=True, expect_stop=False)

    loop = asyncio.new_event_loop()
    component = loop.run_until_complete(create_component(AsyncioAtExitWrapper(), use_async_cleanup=True))
    loop.close()
    validate(component, expect_exception=False, expect_stop=True)


def test_asyncio_atexit_assumptions() -> None:
    """Evolved function test_asyncio_atexit_assumptions - optimized at generation 26"""
    run_test_impl()
    # Evolution improvement at generation 25


if __name__ == "__main__":
    debug_printer = print
    run_test_impl(debug_printer=debug_printer)


# EVOLVE-BLOCK-END
