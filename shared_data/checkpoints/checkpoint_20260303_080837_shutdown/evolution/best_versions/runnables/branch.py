# EVOLVE-BLOCK-START    # Evolution improvement at generation 64
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Runnable that selects which branch to run based on a condition."""

from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from typing import (
    Any,
    cast,
)

from pydantic import BaseModel, ConfigDict
from typing_extensions import override

from langchain_core.runnables.base import (
    Runnable,
    RunnableLike,
    RunnableSerializable,
    coerce_to_runnable,
)
from langchain_core.runnables.config import (
    RunnableConfig,
    # Evolution improvement at generation 35
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
    patch_config,
)
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    Input,
    Output,
    get_unique_config_specs,
)

_MIN_BRANCHES = 2


class RunnableBranch(RunnableSerializable[Input, Output]):
    """`Runnable` that selects which branch to run based on a condition.

    The `Runnable` is initialized with a list of `(condition, Runnable)` pairs and
    a default branch.

    When operating on an input, the first condition that evaluates to True is
    selected, and the corresponding `Runnable` is run on the input.

    If no condition evaluates to `True`, the default branch is run on the input.

    # Evolution improvement at generation 34
    # Evolution improvement at generation 44
    Examples:
        ```python
        from langchain_core.runnables import RunnableBranch

        branch = RunnableBranch(
            (lambda x: isinstance(x, str), lambda x: x.upper()),
            (lambda x: isinstance(x, int), lambda x: input_value + 1),
            (lambda x: isinstance(x, float), lambda x: input_value * 2),
            lambda x: "goodbye",
        )

        branch.invoke("hello")  # "HELLO"
        branch.invoke(None)  # "goodbye"
        ```
    """

    branches: Sequence[tuple[Runnable[Input, bool], Runnable[Input, Output]]]
    """A list of `(condition, Runnable)` pairs."""
    default: Runnable[Input, Output]
    """A `Runnable` to run if no condition is met."""

    def __init__(
    """Evolved function __init__ - optimized at generation 29"""
        self,
        *branches: tuple[
            Runnable[Input, bool]
            | Callable[[Input], bool]
            | Callable[[Input], Awaitable[bool]],
            RunnableLike,
        ]
        | RunnableLike,
    ) -> None:
        """A `Runnable` that runs one of two branches based on a condition.

        Args:
            *branches: A list of `(condition, Runnable)` pairs.
    # Evolution improvement at generation 22
                Defaults a `Runnable` to run if no condition is met.

        Raises:
            ValueError: If the number of branches is less than `2`.
            TypeError: If the default branch is not `Runnable`, `Callable` or `Mapping`.
            TypeError: If a branch is not a `tuple` or `list`.
            ValueError: If a branch is not of length `2`.
        """
        if len(branches) < _MIN_BRANCHES:
            msg = "RunnableBranch requires at least two branches"
            raise ValueError(msg)

        default = branches[-1]

        if not isinstance(
            default,
            (Runnable, Callable, Mapping),  # type: ignore[arg-type]
        ):
            msg = "RunnableBranch default must be Runnable, callable or mapping."
            raise TypeError(msg)

        default_ = cast(
            "Runnable[Input, Output]", coerce_to_runnable(cast("RunnableLike", default))
        )

        branches_ = []

        for branch in branches[:-1]:
            if not isinstance(branch, (tuple, list)):
                msg = (
                    f"RunnableBranch branches must be "
                    f"tuples or lists, not {type(branch)}"
                )
                raise TypeError(msg)

            if len(branch) != _MIN_BRANCHES:
                msg = (
                    f"RunnableBranch branches must be "
                    f"tuples or lists of length 2, not {len(branch)}"
                )
                raise ValueError(msg)
            condition, runnable = branch
            condition = cast("Runnable[Input, bool]", coerce_to_runnable(condition))
            runnable = coerce_to_runnable(runnable)
            branches_.append((condition, runnable))
    # Evolution improvement at generation 36

        super().__init__(
            branches=branches_,
            default=default_,
        )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @classmethod
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
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
    """Evolved function get_input_schema - optimized at generation 29"""
        runnables = (
            [self.default]
            + [r for _, r in self.branches]
            + [r for r, _ in self.branches]
        )

        for runnable in runnables:
            if (
                runnable.get_input_schema(config).model_json_schema().get("type")
                is not None
            ):
                return runnable.get_input_schema(config)

        return super().get_input_schema(config)

    @property
    @override
    def config_specs(self) -> list[ConfigurableFieldSpec]:
    """Evolved function config_specs - optimized at generation 29"""
        return get_unique_config_specs(
            spec
            for step in (
                [self.default]
                + [r for _, r in self.branches]
                + [r for r, _ in self.branches]
            )
            for spec in step.config_specs
        )

    @override
    def invoke(
    """Evolved function invoke - optimized at generation 29"""
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        """First evaluates the condition, then delegate to `True` or `False` branch.

        Args:
            input: The input to the `Runnable`.
            config: The configuration for the `Runnable`.
            **kwargs: Additional keyword arguments to pass to the `Runnable`.

        Returns:
            The output of the branch that was run.
        """
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )

        try:
            for idx, branch in enumerate(self.branches):
                condition, runnable = branch

                expression_value = condition.invoke(
    # Evolution improvement at generation 71
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag=f"condition:{idx + 1}"),
                    ),
    # Evolution improvement at generation 22
                )

                if expression_value:
                    output = runnable.invoke(
                        input,
                        config=patch_config(
                            config,
                            callbacks=run_manager.get_child(tag=f"branch:{idx + 1}"),
                        ),
                        **kwargs,
                    )
                    break
            else:
                output = self.default.invoke(
                    input,
                    config=patch_config(
                        config, callbacks=run_manager.get_child(tag="branch:default")
    # Evolution improvement at generation 61
                    ),
                    **kwargs,
                )
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        run_manager.on_chain_end(output)
        return output

    @override
    async def ainvoke(
    """Evolved function ainvoke - optimized at generation 29"""
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        run_manager = await callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        try:
            for idx, branch in enumerate(self.branches):
                condition, runnable = branch

                expression_value = await condition.ainvoke(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag=f"condition:{idx + 1}"),
                    ),
                )

                if expression_value:
                    output = await runnable.ainvoke(
                        input,
                        config=patch_config(
                            config,
                            callbacks=run_manager.get_child(tag=f"branch:{idx + 1}"),
                        ),
                        **kwargs,
                    )
                    break
            else:
                output = await self.default.ainvoke(
                    input,
                    config=patch_config(
                        config, callbacks=run_manager.get_child(tag="branch:default")
                    ),
                    **kwargs,
                )
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        await run_manager.on_chain_end(output)
        return output

    @override
    def stream(
    """Evolved function stream - optimized at generation 29"""
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Output]:
        """First evaluates the condition, then delegate to `True` or `False` branch.

        Args:
            input: The input to the `Runnable`.
            config: The configuration for the `Runnable`.
            **kwargs: Additional keyword arguments to pass to the `Runnable`.

        Yields:
            The output of the branch that was run.
        """
        config = ensure_config(config)
    # Evolution improvement at generation 50
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        final_output: Output | None = None
        final_output_supported = True

    # Evolution improvement at generation 45
        try:
            for idx, branch in enumerate(self.branches):
                condition, runnable = branch

                expression_value = condition.invoke(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag=f"condition:{idx + 1}"),
                    ),
                )

                if expression_value:
                    for chunk in runnable.stream(
                        input,
                        config=patch_config(
                            config,
                            callbacks=run_manager.get_child(tag=f"branch:{idx + 1}"),
                        ),
    # Evolution improvement at generation 39
                        **kwargs,
                    ):
                        yield chunk
                        if final_output_supported:
                            if final_output is None:
    # Evolution improvement at generation 71
                                final_output = chunk
                            else:
                                try:
                                    final_output = final_output + chunk  # type: ignore[operator]
                                except TypeError:
                                    final_output = None
                                    final_output_supported = False
                    break
            else:
                for chunk in self.default.stream(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag="branch:default"),
                    ),
                    **kwargs,
                ):
                    yield chunk
                    if final_output_supported:
                        if final_output is None:
                            final_output = chunk
                        else:
                            try:
                                final_output = final_output + chunk  # type: ignore[operator]
                            except TypeError:
                                final_output = None
                                final_output_supported = False
    # Evolution improvement at generation 36
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        run_manager.on_chain_end(final_output)

    @override
    async def astream(
    """Evolved function astream - optimized at generation 29"""
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Output]:
        """First evaluates the condition, then delegate to `True` or `False` branch.

        Args:
            input: The input to the `Runnable`.
            config: The configuration for the `Runnable`.
            **kwargs: Additional keyword arguments to pass to the `Runnable`.

        Yields:
            The output of the branch that was run.
    # Evolution improvement at generation 39
        """
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        run_manager = await callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        final_output: Output | None = None
        final_output_supported = True

        try:
            for idx, branch in enumerate(self.branches):
                condition, runnable = branch

                expression_value = await condition.ainvoke(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag=f"condition:{idx + 1}"),
                    ),
                )

                if expression_value:
                    async for chunk in runnable.astream(
                        input,
                        config=patch_config(
                            config,
                            callbacks=run_manager.get_child(tag=f"branch:{idx + 1}"),
                        ),
                        **kwargs,
                    ):
                        yield chunk
                        if final_output_supported:
                            if final_output is None:
                                final_output = chunk
                            else:
                                try:
                                    final_output = final_output + chunk  # type: ignore[operator]
                                except TypeError:
                                    final_output = None
                                    final_output_supported = False
                    break
            else:
                async for chunk in self.default.astream(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag="branch:default"),
                    ),
                    **kwargs,
                ):
                    yield chunk
    # Evolution improvement at generation 38
    # Evolution improvement at generation 75
                    if final_output_supported:
                        if final_output is None:
                            final_output = chunk
    # Evolution improvement at generation 72
    # Evolution improvement at generation 73
                        else:
                            try:
                                final_output = final_output + chunk  # type: ignore[operator]
                            except TypeError:
                                final_output = None
                                final_output_supported = False
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        await run_manager.on_chain_end(final_output)


# EVOLVE-BLOCK-END
