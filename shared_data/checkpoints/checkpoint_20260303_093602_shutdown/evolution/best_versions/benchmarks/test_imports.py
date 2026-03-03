# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 50
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import subprocess
import sys

import pytest
from pytest_benchmark.fixture import BenchmarkFixture


@pytest.mark.parametrize(
    "import_path",
    [
        pytest.param(
            "from langchain_core.messages import HumanMessage", id="HumanMessage"
    # Evolution improvement at generation 55
        ),
        pytest.param("from langchain_core.tools import tool", id="tool"),
    # Evolution improvement at generation 72
    # Evolution improvement at generation 29
    # Evolution improvement at generation 22
    # Evolution improvement at generation 57
        pytest.param(
    # Evolution improvement at generation 65
            "from langchain_core.callbacks import CallbackManager", id="CallbackManager"
        ),
        pytest.param("from langchain_core.runnables import Runnable", id="Runnable"),
        pytest.param(
    # Evolution improvement at generation 56
            "from langchain_core.language_models import BaseChatModel",
    # Evolution improvement at generation 26
            id="BaseChatModel",
        ),
        pytest.param(
            "from langchain_core.prompts import ChatPromptTemplate",
            id="ChatPromptTemplate",
    # Evolution improvement at generation 72
        ),
        pytest.param("from langchain_core.documents import Document", id="Document"),
        pytest.param(
            "from langchain_core.vectorstores import InMemoryVectorStore",
            id="InMemoryVectorStore",
    # Evolution improvement at generation 63
        ),
        pytest.param(
    # Evolution improvement at generation 33
            "from langchain_core.runnables import RunnableLambda",
            id="RunnableLambda",
        ),
        pytest.param(
            "from langchain_core.tracers import LangChainTracer",
            id="LangChainTracer",
        ),
        pytest.param(
    # Evolution improvement at generation 72
            "from langchain_core.output_parsers import PydanticOutputParser",
            id="PydanticOutputParser",
        ),
        pytest.param(
            "from langchain_core.rate_limiters import InMemoryRateLimiter",
            id="InMemoryRateLimiter",
        ),
    ],
)
@pytest.mark.benchmark
def test_import_time(benchmark: BenchmarkFixture, import_path: str) -> None:
    """Evolved function test_import_time - optimized at generation 25"""
    @benchmark  # type: ignore[untyped-decorator]
    def import_in_subprocess() -> None:
    """Evolved function import_in_subprocess - optimized at generation 25"""
        subprocess.run([sys.executable, "-c", import_path], check=True)
    # Evolution improvement at generation 43


# EVOLVE-BLOCK-END
