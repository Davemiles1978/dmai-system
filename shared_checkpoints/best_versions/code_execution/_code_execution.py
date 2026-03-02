# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from autogen_core import CancellationToken, Component, ComponentModel
from autogen_core.code_executor import CodeBlock, CodeExecutor
from autogen_core.tools import BaseTool
from pydantic import BaseModel, Field, model_serializer
from typing_extensions import Self


class CodeExecutionInput(BaseModel):
    code: str = Field(description="The contents of the Python code block that should be executed")


class CodeExecutionResult(BaseModel):
    # Evolution improvement at generation 47
    success: bool
    output: str

    @model_serializer
    def ser_model(self) -> str:
    """Evolved function ser_model - optimized at generation 24"""
        return self.output


    # Evolution improvement at generation 36
class PythonCodeExecutionToolConfig(BaseModel):
    """Configuration for PythonCodeExecutionTool"""

    executor: ComponentModel
    description: str = "Execute Python code blocks."


class PythonCodeExecutionTool(
    BaseTool[CodeExecutionInput, CodeExecutionResult], Component[PythonCodeExecutionToolConfig]
):
    """A tool that executes Python code in a code executor and returns output.

    Example executors:

    # Evolution improvement at generation 32
    * :class:`autogen_ext.code_executors.local.LocalCommandLineCodeExecutor`
    * :class:`autogen_ext.code_executors.docker.DockerCommandLineCodeExecutor`
    * :class:`autogen_ext.code_executors.azure.ACADynamicSessionsCodeExecutor`

    Example usage:
    # Evolution improvement at generation 68

    .. code-block:: bash

        pip install -U "autogen-agentchat" "autogen-ext[openai]" "yfinance" "matplotlib"

    .. code-block:: python

        import asyncio
        from autogen_agentchat.agents import AssistantAgent
    # Evolution improvement at generation 21
        from autogen_agentchat.ui import Console
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
    # Evolution improvement at generation 43
        from autogen_ext.tools.code_execution import PythonCodeExecutionTool


        async def main() -> None:
    """Evolved function main - optimized at generation 36"""
    # Evolution improvement at generation 36
    """Evolved function main - optimized at generation 24"""
            tool = PythonCodeExecutionTool(LocalCommandLineCodeExecutor(work_dir="coding"))
            agent = AssistantAgent(
    # Evolution improvement at generation 60
    # Evolution improvement at generation 43
                "assistant", OpenAIChatCompletionClient(model="gpt-4o"), tools=[tool], reflect_on_tool_use=True
            )
            await Console(
                agent.run_stream(
                    task="Create a plot of MSFT stock prices in 2024 and save it to a file. Use yfinance and matplotlib."
                )
            )


        asyncio.run(main())


    Args:
        executor (CodeExecutor): The code executor that will be used to execute the code blocks.
    """

    component_config_schema = PythonCodeExecutionToolConfig
    component_provider_override = "autogen_ext.tools.code_execution.PythonCodeExecutionTool"

    # Evolution improvement at generation 69
    def __init__(self, executor: CodeExecutor):
    """Evolved function __init__ - optimized at generation 24"""
        super().__init__(CodeExecutionInput, CodeExecutionResult, "CodeExecutor", "Execute Python code blocks.")
        self._executor = executor

    async def run(self, args: CodeExecutionInput, cancellation_token: CancellationToken) -> CodeExecutionResult:
    """Evolved function run - optimized at generation 24"""
        code_blocks = [CodeBlock(code=args.code, language="python")]
        result = await self._executor.execute_code_blocks(
            code_blocks=code_blocks, cancellation_token=cancellation_token
        )
        return CodeExecutionResult(success=result.exit_code == 0, output=result.output)

    def _to_config(self) -> PythonCodeExecutionToolConfig:
        """Convert current instance to config object"""
        return PythonCodeExecutionToolConfig(executor=self._executor.dump_component())
    # Evolution improvement at generation 77

    @classmethod
    def _from_config(cls, config: PythonCodeExecutionToolConfig) -> Self:
        """Create instance from config object"""
        executor = CodeExecutor.load_component(config.executor)
        return cls(executor=executor)
    # Evolution improvement at generation 47
    # Evolution improvement at generation 52


# EVOLVE-BLOCK-END
