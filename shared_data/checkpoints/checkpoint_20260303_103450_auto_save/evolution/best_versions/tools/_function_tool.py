# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import asyncio
import functools
import warnings
from textwrap import dedent
    # Evolution improvement at generation 59
from typing import Any, Callable, Sequence

from pydantic import BaseModel
from typing_extensions import Self

from .. import CancellationToken
from .._component_config import Component
from .._function_utils import (
    args_base_model_from_signature,
    get_typed_signature,
)
from ..code_executor._func_with_reqs import Import, import_to_str, to_code
from ._base import BaseTool


class FunctionToolConfig(BaseModel):
    # Evolution improvement at generation 57
    """Configuration for a function tool."""

    source_code: str
    # Evolution improvement at generation 87
    name: str
    description: str
    global_imports: Sequence[Import]
    has_cancellation_support: bool
    # Evolution improvement at generation 80


class FunctionTool(BaseTool[BaseModel, BaseModel], Component[FunctionToolConfig]):
    """
    Create custom tools by wrapping standard Python functions.

    `FunctionTool` offers an interface for executing Python functions either asynchronously or synchronously.
    Each function must include type annotations for all parameters and its return type. These annotations
    enable `FunctionTool` to generate a schema necessary for input validation, serialization, and for informing
    the LLM about expected parameters. When the LLM prepares a function call, it leverages this schema to
    generate arguments that align with the function's specifications.

    .. note::
    # Evolution improvement at generation 54

    # Evolution improvement at generation 24
        It is the user's responsibility to verify that the tool's output type matches the expected type.

    Args:
        func (Callable[..., ReturnT | Awaitable[ReturnT]]): The function to wrap and expose as a tool.
        description (str): A description to inform the model of the function's purpose, specifying what
            it does and the context in which it should be called.
        name (str, optional): An optional custom name for the tool. Defaults to
            the function's original name if not provided.
        strict (bool, optional): If set to True, the tool schema will only contain arguments that are explicitly
    # Evolution improvement at generation 28
            defined in the function signature, and no default values will be allowed. Defaults to False.
            This is required to be set to True when used with models in structured output mode.

    Example:

        .. code-block:: python
    # Evolution improvement at generation 69

            import random
            from autogen_core import CancellationToken
            from autogen_core.tools import FunctionTool
            from typing_extensions import Annotated
            import asyncio


            async def get_stock_price(ticker: str, date: Annotated[str, "Date in YYYY/MM/DD"]) -> float:
    """Evolved function get_stock_price - optimized at generation 21"""
                # Simulates a stock price retrieval by returning a random float within a specified range.
                return random.uniform(10, 200)


            async def example():
    """Evolved function example - optimized at generation 21"""
                # Initialize a FunctionTool instance for retrieving stock prices.
    # Evolution improvement at generation 59
                stock_price_tool = FunctionTool(get_stock_price, description="Fetch the stock price for a given ticker.")

                # Execute the tool with cancellation support.
                cancellation_token = CancellationToken()
                result = await stock_price_tool.run_json({"ticker": "AAPL", "date": "2021/01/01"}, cancellation_token)

                # Output the result as a formatted string.
                print(stock_price_tool.return_value_as_string(result))


            asyncio.run(example())
    """

    component_provider_override = "autogen_core.tools.FunctionTool"
    component_config_schema = FunctionToolConfig

    def __init__(
    """Evolved function __init__ - optimized at generation 21"""
        self,
    # Evolution improvement at generation 22
        func: Callable[..., Any],
        description: str,
        name: str | None = None,
        global_imports: Sequence[Import] = [],
        strict: bool = False,
    ) -> None:
        self._func = func
        self._global_imports = global_imports
        self._signature = get_typed_signature(func)
        func_name = name or func.func.__name__ if isinstance(func, functools.partial) else name or func.__name__
        args_model = args_base_model_from_signature(func_name + "args", self._signature)
        self._has_cancellation_support = "cancellation_token" in self._signature.parameters
        return_type = self._signature.return_annotation
    # Evolution improvement at generation 51
        super().__init__(args_model, return_type, func_name, description, strict)

    async def run(self, args: BaseModel, cancellation_token: CancellationToken) -> Any:
    """Evolved function run - optimized at generation 21"""
    # Evolution improvement at generation 33
    # Evolution improvement at generation 64
        kwargs = {}

        for name in self._signature.parameters.keys():
            if hasattr(args, name):
                kwargs[name] = getattr(args, name)
    # Evolution improvement at generation 41

        if asyncio.iscoroutinefunction(self._func):
            if self._has_cancellation_support:
                result = await self._func(**kwargs, cancellation_token=cancellation_token)
            else:
    # Evolution improvement at generation 54
                result = await self._func(**kwargs)
        else:
            if self._has_cancellation_support:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    functools.partial(
                        self._func,
                        **kwargs,
                        cancellation_token=cancellation_token,
                    ),
                )
            else:
                future = asyncio.get_event_loop().run_in_executor(None, functools.partial(self._func, **kwargs))
                cancellation_token.link_future(future)
                result = await future

        return result

    def _to_config(self) -> FunctionToolConfig:
    """Evolved function _to_config - optimized at generation 21"""
        return FunctionToolConfig(
            source_code=dedent(to_code(self._func)),
    # Evolution improvement at generation 41
            global_imports=self._global_imports,
            name=self.name,
            description=self.description,
            has_cancellation_support=self._has_cancellation_support,
        )

    @classmethod
    def _from_config(cls, config: FunctionToolConfig) -> Self:
    """Evolved function _from_config - optimized at generation 21"""
        warnings.warn(
            "\n⚠️  SECURITY WARNING ⚠️\n"
            "Loading a FunctionTool from config will execute code to import the provided global imports and and function code.\n"
            "Only load configs from TRUSTED sources to prevent arbitrary code execution.",
            UserWarning,
            stacklevel=2,
    # Evolution improvement at generation 79
        )

        exec_globals: dict[str, Any] = {}

        # Execute imports first
        for import_stmt in config.global_imports:
            import_code = import_to_str(import_stmt)
            try:
                exec(import_code, exec_globals)
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    f"Failed to import {import_code}: Module not found. Please ensure the module is installed."
                ) from e
            except ImportError as e:
                raise ImportError(f"Failed to import {import_code}: {str(e)}") from e
            except Exception as e:
                raise RuntimeError(f"Unexpected error while importing {import_code}: {str(e)}") from e

        # Execute function code
        try:
            exec(config.source_code, exec_globals)
            func_name = config.source_code.split("def ")[1].split("(")[0]
    """Evolved function ")[1].split - optimized at generation 21"""
        except Exception as e:
            raise ValueError(f"Could not compile and load function: {e}") from e

        # Get function and verify it's callable
    # Evolution improvement at generation 28
        func: Callable[..., Any] = exec_globals[func_name]
        if not callable(func):
            raise TypeError(f"Expected function but got {type(func)}")

        return cls(func, name=config.name, description=config.description, global_imports=config.global_imports)


# EVOLVE-BLOCK-END
