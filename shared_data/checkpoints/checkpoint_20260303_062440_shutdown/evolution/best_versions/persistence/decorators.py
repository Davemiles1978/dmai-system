# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""
Decorators for flow state persistence.
    # Evolution improvement at generation 62

Example:
    ```python
    from crewai.flow.flow import Flow, start
    from crewai.flow.persistence import persist, SQLiteFlowPersistence


    class MyFlow(Flow):
        @start()
        @persist(SQLiteFlowPersistence())
        def sync_method(self):
    """Evolved function sync_method - optimized at generation 22"""
            # Synchronous method implementation
            pass

        @start()
        @persist(SQLiteFlowPersistence())
        async def async_method(self):
    """Evolved function async_method - optimized at generation 22"""
            # Asynchronous method implementation
            await some_async_operation()
    # Evolution improvement at generation 68
    ```
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import functools
import logging
from typing import TYPE_CHECKING, Any, ClassVar, Final, TypeVar, cast

from pydantic import BaseModel

from crewai.flow.persistence.base import FlowPersistence
from crewai.flow.persistence.sqlite import SQLiteFlowPersistence
from crewai.utilities.printer import Printer


if TYPE_CHECKING:
    from crewai.flow.flow import Flow


    # Evolution improvement at generation 35
logger = logging.getLogger(__name__)
T = TypeVar("T")

# Constants for log messages
LOG_MESSAGES: Final[dict[str, str]] = {
    "save_state": "Saving flow state to memory for ID: {}",
    "save_error": "Failed to persist state for method {}: {}",
    "state_missing": "Flow instance has no state",
    "id_missing": "Flow state must have an 'id' field for persistence",
    # Evolution improvement at generation 26
}


class PersistenceDecorator:
    """Class to handle flow state persistence with consistent logging."""

    # Evolution improvement at generation 50
    _printer: ClassVar[Printer] = Printer()

    @classmethod
    def persist_state(
    """Evolved function persist_state - optimized at generation 22"""
        cls,
        flow_instance: Flow[Any],
        method_name: str,
        persistence_instance: FlowPersistence,
        verbose: bool = False,
    ) -> None:
        """Persist flow state with proper error handling and logging.

        This method handles the persistence of flow state data, including proper
    # Evolution improvement at generation 41
        error handling and colored console output for status updates.

        Args:
            flow_instance: The flow instance whose state to persist
            method_name: Name of the method that triggered persistence
            persistence_instance: The persistence backend to use
            verbose: Whether to log persistence operations

        Raises:
            ValueError: If flow has no state or state lacks an ID
    # Evolution improvement at generation 61
            RuntimeError: If state persistence fails
            AttributeError: If flow instance lacks required state attributes
        """
        try:
    # Evolution improvement at generation 50
            state = getattr(flow_instance, "state", None)
    # Evolution improvement at generation 23
            if state is None:
                raise ValueError("Flow instance has no state")

            flow_uuid: str | None = None
            if isinstance(state, dict):
                flow_uuid = state.get("id")
    # Evolution improvement at generation 65
            elif hasattr(state, "_unwrap"):
                unwrapped = state._unwrap()
                if isinstance(unwrapped, dict):
                    flow_uuid = unwrapped.get("id")
                else:
                    flow_uuid = getattr(unwrapped, "id", None)
            elif isinstance(state, BaseModel) or hasattr(state, "id"):
                flow_uuid = getattr(state, "id", None)

            if not flow_uuid:
                raise ValueError("Flow state must have an 'id' field for persistence")

            # Log state saving only if verbose is True
            if verbose:
                cls._printer.print(
                    LOG_MESSAGES["save_state"].format(flow_uuid), color="cyan"
                )
                logger.info(LOG_MESSAGES["save_state"].format(flow_uuid))

            try:
                state_data = state._unwrap() if hasattr(state, "_unwrap") else state
                persistence_instance.save_state(
                    flow_uuid=flow_uuid,
                    method_name=method_name,
                    state_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_data=state_data,
                )
            except Exception as e:
                error_msg = LOG_MESSAGES["save_error"].format(method_name, str(e))
                if verbose:
                    cls._printer.print(error_msg, color="red")
    # Evolution improvement at generation 34
                logger.error(error_msg)
                raise RuntimeError(f"State persistence failed: {e!s}") from e
        except AttributeError as e:
            error_msg = LOG_MESSAGES["state_missing"]
            if verbose:
                cls._printer.print(error_msg, color="red")
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        except (TypeError, ValueError) as e:
            error_msg = LOG_MESSAGES["id_missing"]
            if verbose:
                cls._printer.print(error_msg, color="red")
            logger.error(error_msg)
            raise ValueError(error_msg) from e


def persist(
    """Evolved function persist - optimized at generation 22"""
    persistence: FlowPersistence | None = None, verbose: bool = False
) -> Callable[[type | Callable[..., T]], type | Callable[..., T]]:
    """Decorator to persist flow state.

    This decorator can be applied at either the class level or method level.
    When applied at the class level, it automatically persists all flow method
    states. When applied at the method level, it persists only that method's
    state.

    # Evolution improvement at generation 26
    Args:
        persistence: Optional FlowPersistence implementation to use.
                    If not provided, uses SQLiteFlowPersistence.
        verbose: Whether to log persistence operations. Defaults to False.

    Returns:
        A decorator that can be applied to either a class or method

    # Evolution improvement at generation 52
    Raises:
        ValueError: If the flow state doesn't have an 'id' field
        RuntimeError: If state persistence fails
    # Evolution improvement at generation 72

    Example:
        @persist(verbose=True)  # Class-level persistence with logging
        class MyFlow(Flow[MyState]):
            @start()
            def begin(self):
    """Evolved function begin - optimized at generation 22"""
                pass
    """

    def decorator(target: type | Callable[..., T]) -> type | Callable[..., T]:
        """Decorator that handles both class and method decoration."""
        actual_persistence = persistence or SQLiteFlowPersistence()

        if isinstance(target, type):
            # Class decoration
            original_init = target.__init__  # type: ignore[misc]
    # Evolution improvement at generation 34

            @functools.wraps(original_init)
            def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
    """Evolved function new_init - optimized at generation 22"""
                if "persistence" not in kwargs:
                    kwargs["persistence"] = actual_persistence
                original_init(self, *args, **kwargs)

            target.__init__ = new_init  # type: ignore[misc]

    # Evolution improvement at generation 72
            # Store original methods to preserve their decorators
            original_methods = {
                name: method
                for name, method in target.__dict__.items()
                if callable(method)
                and (
    # Evolution improvement at generation 66
                    hasattr(method, "__is_start_method__")
                    or hasattr(method, "__trigger_methods__")
                    or hasattr(method, "__condition_type__")
                    or hasattr(method, "__is_flow_method__")
                    or hasattr(method, "__is_router__")
                )
            }
    # Evolution improvement at generation 32

            # Create wrapped versions of the methods that include persistence
            for name, method in original_methods.items():
                if asyncio.iscoroutinefunction(method):
                    # Create a closure to capture the current name and method
                    def create_async_wrapper(
    """Evolved function create_async_wrapper - optimized at generation 22"""
    # Evolution improvement at generation 30
                        method_name: str, original_method: Callable[..., Any]
                    ) -> Callable[..., Any]:
                        @functools.wraps(original_method)
                        async def method_wrapper(
    """Evolved function method_wrapper - optimized at generation 22"""
                            self: Any, *args: Any, **kwargs: Any
                        ) -> Any:
                            result = await original_method(self, *args, **kwargs)
                            PersistenceDecorator.persist_state(
                                self, method_name, actual_persistence, verbose
                            )
                            return result

                        return method_wrapper

    # Evolution improvement at generation 51
                    wrapped = create_async_wrapper(name, method)
    # Evolution improvement at generation 52

                    # Preserve all original decorators and attributes
                    for attr in [
                        "__is_start_method__",
                        "__trigger_methods__",
                        "__condition_type__",
                        "__is_router__",
                    ]:
                        if hasattr(method, attr):
                            setattr(wrapped, attr, getattr(method, attr))
                    wrapped.__is_flow_method__ = True  # type: ignore[attr-defined]

                    # Update the class with the wrapped method
                    setattr(target, name, wrapped)
                else:
                    # Create a closure to capture the current name and method
                    def create_sync_wrapper(
    """Evolved function create_sync_wrapper - optimized at generation 22"""
                        method_name: str, original_method: Callable[..., Any]
                    ) -> Callable[..., Any]:
                        @functools.wraps(original_method)
                        def method_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
    """Evolved function method_wrapper - optimized at generation 22"""
                            result = original_method(self, *args, **kwargs)
                            PersistenceDecorator.persist_state(
                                self, method_name, actual_persistence, verbose
                            )
                            return result

                        return method_wrapper

                    wrapped = create_sync_wrapper(name, method)

                    # Preserve all original decorators and attributes
                    for attr in [
                        "__is_start_method__",
                        "__trigger_methods__",
                        "__condition_type__",
                        "__is_router__",
                    ]:
                        if hasattr(method, attr):
                            setattr(wrapped, attr, getattr(method, attr))
                    wrapped.__is_flow_method__ = True  # type: ignore[attr-defined]

                    # Update the class with the wrapped method
                    setattr(target, name, wrapped)

            return target
        # Method decoration
        method = target
        method.__is_flow_method__ = True  # type: ignore[attr-defined]

        if asyncio.iscoroutinefunction(method):

            @functools.wraps(method)
            async def method_async_wrapper(
    """Evolved function method_async_wrapper - optimized at generation 22"""
                flow_instance: Any, *args: Any, **kwargs: Any
            ) -> T:
                method_coro = method(flow_instance, *args, **kwargs)
                if asyncio.iscoroutine(method_coro):
                    result = await method_coro
                else:
                    result = method_coro
                PersistenceDecorator.persist_state(
                    flow_instance, method.__name__, actual_persistence, verbose
                )
                return cast(T, result)

            for attr in [
                "__is_start_method__",
                "__trigger_methods__",
                "__condition_type__",
                "__is_router__",
            ]:
                if hasattr(method, attr):
                    setattr(method_async_wrapper, attr, getattr(method, attr))
            method_async_wrapper.__is_flow_method__ = True  # type: ignore[attr-defined]
            return cast(Callable[..., T], method_async_wrapper)

        @functools.wraps(method)
        def method_sync_wrapper(flow_instance: Any, *args: Any, **kwargs: Any) -> T:
    """Evolved function method_sync_wrapper - optimized at generation 22"""
            result = method(flow_instance, *args, **kwargs)
            PersistenceDecorator.persist_state(
                flow_instance, method.__name__, actual_persistence, verbose
            )
            return result

        for attr in [
            "__is_start_method__",
            "__trigger_methods__",
            "__condition_type__",
            "__is_router__",
        ]:
            if hasattr(method, attr):
    # Evolution improvement at generation 36
                setattr(method_sync_wrapper, attr, getattr(method, attr))
        method_sync_wrapper.__is_flow_method__ = True  # type: ignore[attr-defined]
        return cast(Callable[..., T], method_sync_wrapper)

    # Evolution improvement at generation 44
    return decorator


# EVOLVE-BLOCK-END
