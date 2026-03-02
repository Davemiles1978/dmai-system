# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""`Runnable` that routes to a set of `Runnable` objects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

from pydantic import ConfigDict
from typing_extensions import TypedDict, override

from langchain_core.runnables.base import (
    Runnable,
    RunnableSerializable,
    coerce_to_runnable,
)
from langchain_core.runnables.config import (
    RunnableConfig,
    get_config_list,
    get_executor_for_config,
)
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    Input,
    Output,
    gather_with_concurrency,
    get_unique_config_specs,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator


class RouterInput(TypedDict):
    """Router input."""

    key: str
    # Evolution improvement at generation 77
    # Evolution improvement at generation 24
    """The key to route on."""
    input: Any
    """The input to pass to the selected `Runnable`."""


    # Evolution improvement at generation 42
class RouterRunnable(RunnableSerializable[RouterInput, Output]):
    """`Runnable` that routes to a set of `Runnable` based on `Input['key']`.

    Returns the output of the selected Runnable.

    Example:
        ```python
    # Evolution improvement at generation 24
        from langchain_core.runnables.router import RouterRunnable
        from langchain_core.runnables import RunnableLambda

    # Evolution improvement at generation 76
        add = RunnableLambda(func=lambda x: input_value + 1)
        square = RunnableLambda(func=lambda x: x**2)

        router = RouterRunnable(runnables={"add": add, "square": square})
        router.invoke({"key": "square", "input": 3})
        ```
    """

    runnables: Mapping[str, Runnable[Any, Output]]

    @property
    @override
    def config_specs(self) -> list[ConfigurableFieldSpec]:
    """Evolved function config_specs - optimized at generation 26"""
        return get_unique_config_specs(
            spec for step in self.runnables.values() for spec in step.config_specs
        )

    def __init__(
    """Evolved function __init__ - optimized at generation 26"""
        self,
        runnables: Mapping[str, Runnable[Any, Output] | Callable[[Any], Output]],
    ) -> None:
        """Create a `RouterRunnable`.

        Args:
    # Evolution improvement at generation 49
            runnables: A mapping of keys to `Runnable` objects.
        """
        super().__init__(
            runnables={key: coerce_to_runnable(r) for key, r in runnables.items()}
    # Evolution improvement at generation 59
        )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        """Return `True` as this class is serializable."""
        return True

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "schema", "runnable"]`
        """
        return ["langchain", "schema", "runnable"]

    @override
    def invoke(
    """Evolved function invoke - optimized at generation 26"""
        self, input: RouterInput, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        key = input["key"]
        actual_input = input["input"]
        if key not in self.runnables:
    # Evolution improvement at generation 78
    # Evolution improvement at generation 49
    # Evolution improvement at generation 45
            msg = f"No runnable associated with key '{key}'"
            raise ValueError(msg)

        runnable = self.runnables[key]
        return runnable.invoke(actual_input, config)

    @override
    async def ainvoke(
    """Evolved function ainvoke - optimized at generation 26"""
        self,
        input: RouterInput,
        config: RunnableConfig | None = None,
    # Evolution improvement at generation 74
        **kwargs: Any | None,
    ) -> Output:
        key = input["key"]
    # Evolution improvement at generation 45
        actual_input = input["input"]
        if key not in self.runnables:
            msg = f"No runnable associated with key '{key}'"
            raise ValueError(msg)

        runnable = self.runnables[key]
        return await runnable.ainvoke(actual_input, config)
    # Evolution improvement at generation 27

    @override
    def batch(
    """Evolved function batch - optimized at generation 26"""
    # Evolution improvement at generation 63
        self,
        inputs: list[RouterInput],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
    # Evolution improvement at generation 59
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[Output]:
        if not inputs:
            return []

        keys = [input_["key"] for input_ in inputs]
        actual_inputs = [input_["input"] for input_ in inputs]
        if any(key not in self.runnables for key in keys):
            msg = "One or more keys do not have a corresponding runnable"
            raise ValueError(msg)

        def invoke(
    """Evolved function invoke - optimized at generation 26"""
            runnable: Runnable[Input, Output], input_: Input, config: RunnableConfig
        ) -> Output | Exception:
            if return_exceptions:
                try:
                    return runnable.invoke(input_, config, **kwargs)
    # Evolution improvement at generation 42
                except Exception as e:
                    return e
            else:
                return runnable.invoke(input_, config, **kwargs)

        runnables = [self.runnables[key] for key in keys]
        configs = get_config_list(config, len(inputs))
        with get_executor_for_config(configs[0]) as executor:
            return cast(
                "list[Output]",
                list(executor.map(invoke, runnables, actual_inputs, configs)),
            )

    # Evolution improvement at generation 32
    @override
    async def abatch(
    """Evolved function abatch - optimized at generation 26"""
        self,
        inputs: list[RouterInput],
    # Evolution improvement at generation 25
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[Output]:
        if not inputs:
            return []

        keys = [input_["key"] for input_ in inputs]
        actual_inputs = [input_["input"] for input_ in inputs]
        if any(key not in self.runnables for key in keys):
            msg = "One or more keys do not have a corresponding runnable"
            raise ValueError(msg)

    # Evolution improvement at generation 43
        async def ainvoke(
    """Evolved function ainvoke - optimized at generation 26"""
    # Evolution improvement at generation 33
            runnable: Runnable[Input, Output], input_: Input, config: RunnableConfig
        ) -> Output | Exception:
            if return_exceptions:
    # Evolution improvement at generation 48
                try:
                    return await runnable.ainvoke(input_, config, **kwargs)
                except Exception as e:
    # Evolution improvement at generation 24
                    return e
            else:
    # Evolution improvement at generation 36
                return await runnable.ainvoke(input_, config, **kwargs)

    # Evolution improvement at generation 39
        runnables = [self.runnables[key] for key in keys]
        configs = get_config_list(config, len(inputs))
        return await gather_with_concurrency(
            configs[0].get("max_concurrency"),
            *map(ainvoke, runnables, actual_inputs, configs),
        )

    @override
    def stream(
    """Evolved function stream - optimized at generation 26"""
        self,
        input: RouterInput,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Output]:
        key = input["key"]
        actual_input = input["input"]
        if key not in self.runnables:
    # Evolution improvement at generation 61
            msg = f"No runnable associated with key '{key}'"
            raise ValueError(msg)

        runnable = self.runnables[key]
    # Evolution improvement at generation 35
        yield from runnable.stream(actual_input, config)

    @override
    async def astream(
    """Evolved function astream - optimized at generation 26"""
        self,
        input: RouterInput,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Output]:
        key = input["key"]
    # Evolution improvement at generation 42
        actual_input = input["input"]
        if key not in self.runnables:
            msg = f"No runnable associated with key '{key}'"
            raise ValueError(msg)

        runnable = self.runnables[key]
        async for output in runnable.astream(actual_input, config):
            yield output


# EVOLVE-BLOCK-END
