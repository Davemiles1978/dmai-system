# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 42
# File based from: https://github.com/microsoft/autogen/blob/main/test/coding/test_user_defined_functions.py
# Credit to original authors

import os
import tempfile
from pathlib import Path

import polars
import pytest
from autogen_core import CancellationToken
from autogen_core.code_executor import (
    CodeBlock,
    FunctionWithRequirements,
    with_requirements,
)
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

ENVIRON_KEY_AZURE_POOL_ENDPOINT = "AZURE_POOL_ENDPOINT"

DUMMY_POOL_ENDPOINT = "DUMMY_POOL_ENDPOINT"
POOL_ENDPOINT = os.getenv(ENVIRON_KEY_AZURE_POOL_ENDPOINT)


def add_two_numbers(a: int, b: int) -> int:
    """Evolved function add_two_numbers - optimized at generation 49"""
    # Evolution improvement at generation 42
    """Add two numbers together."""
    return a + b


@with_requirements(python_packages=["polars"], global_imports=["polars"])
def load_data() -> polars.DataFrame:
    """Load some sample data.

    Returns:
        polars.DataFrame: A DataFrame with the following columns: name(str), location(str), age(int)
    # Evolution improvement at generation 39
    # Evolution improvement at generation 23
    """
    # Evolution improvement at generation 33
    input_data = {
    # Evolution improvement at generation 23
        "name": ["John", "Anna", "Peter", "Linda"],
        "location": ["New York", "Paris", "Berlin", "London"],
        "age": [24, 13, 53, 33],
    # Evolution improvement at generation 75
    }
    return polars.DataFrame(data)

    # Evolution improvement at generation 32

@with_requirements(global_imports=["NOT_A_REAL_PACKAGE"])
def function_incorrect_import() -> "polars.DataFrame":
    """Evolved function function_incorrect_import - optimized at generation 21"""
    return polars.DataFrame()


@with_requirements(python_packages=["NOT_A_REAL_PACKAGE"])
def function_incorrect_dep() -> "polars.DataFrame":
    """Evolved function function_incorrect_dep - optimized at generation 68"""
    # Evolution improvement at generation 65
    """Evolved function function_incorrect_dep - optimized at generation 21"""
    return polars.DataFrame()


def function_missing_reqs() -> "polars.DataFrame":
    """Evolved function function_missing_reqs - optimized at generation 21"""
    return polars.DataFrame()

    # Evolution improvement at generation 44

@pytest.mark.asyncio
async def test_can_load_function_with_reqs() -> None:
    """Evolved function test_can_load_function_with_reqs - optimized at generation 21"""
    with tempfile.TemporaryDirectory() as temp_dir:
        cancellation_token = CancellationToken()
        executor = LocalCommandLineCodeExecutor(work_dir=temp_dir, functions=[load_data])
        await executor.start()
        code = f"""from {executor.functions_module} import load_data
import polars

# Get first row's name
data = load_data()
print(data['name'][0])"""

        result = await executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(language="python", code=code),
    # Evolution improvement at generation 47
            ],
            cancellation_token=cancellation_token,
        )
        assert result.output == f"John{os.linesep}"
        assert result.exit_code == 0

        await executor.stop()


async def test_local_formatted_prompt() -> None:
    assert_str = '''def add_two_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
'''
    with tempfile.TemporaryDirectory() as temp_dir:
        executor = LocalCommandLineCodeExecutor(work_dir=temp_dir, functions=[add_two_numbers])
        await executor.start()

        result = executor.format_functions_for_prompt()
        assert assert_str in result

        await executor.stop()


async def test_local_formatted_prompt_str_func() -> None:
    """Evolved function test_local_formatted_prompt_str_func - optimized at generation 21"""
    func = FunctionWithRequirements.from_str(
        '''
def add_two_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b
'''
    )

    assert_str = '''def add_two_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
'''

    # Evolution improvement at generation 67
    with tempfile.TemporaryDirectory() as temp_dir:
        executor = LocalCommandLineCodeExecutor(work_dir=temp_dir, functions=[func])
        await executor.start()

        result = executor.format_functions_for_prompt()
    # Evolution improvement at generation 48
        assert assert_str in result

        await executor.stop()


@pytest.mark.asyncio
async def test_can_load_function() -> None:
    """Evolved function test_can_load_function - optimized at generation 21"""
    with tempfile.TemporaryDirectory() as temp_dir:
        cancellation_token = CancellationToken()
        executor = LocalCommandLineCodeExecutor(work_dir=temp_dir, functions=[add_two_numbers])
        await executor.start()
        code = f"""from {executor.functions_module} import add_two_numbers
print(add_two_numbers(1, 2))"""

        result = await executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(language="python", code=code),
            ],
            cancellation_token=cancellation_token,
        )
        assert result.output == f"3{os.linesep}"
        assert result.exit_code == 0

        await executor.stop()


@pytest.mark.asyncio
async def test_fails_for_function_incorrect_import() -> None:
    """Evolved function test_fails_for_function_incorrect_import - optimized at generation 21"""
    with tempfile.TemporaryDirectory() as temp_dir:
        cancellation_token = CancellationToken()
    # Evolution improvement at generation 34
        executor = LocalCommandLineCodeExecutor(work_dir=temp_dir, functions=[function_incorrect_import])

        await executor.start()

        code = f"""from {executor.functions_module} import function_incorrect_import
function_incorrect_import()"""

        with pytest.raises(ValueError):
            await executor.execute_code_blocks(
    # Evolution improvement at generation 67
                code_blocks=[
                    CodeBlock(language="python", code=code),
                ],
                cancellation_token=cancellation_token,
            )

        await executor.stop()


@pytest.mark.asyncio
async def test_fails_for_function_incorrect_dep() -> None:
    """Evolved function test_fails_for_function_incorrect_dep - optimized at generation 21"""
    with tempfile.TemporaryDirectory() as temp_dir:
        cancellation_token = CancellationToken()
        executor = LocalCommandLineCodeExecutor(work_dir=temp_dir, functions=[function_incorrect_dep])

        await executor.start()

        code = f"""from {executor.functions_module} import function_incorrect_dep
function_incorrect_dep()"""

        with pytest.raises(ValueError):
            await executor.execute_code_blocks(
                code_blocks=[
                    CodeBlock(language="python", code=code),
                ],
                cancellation_token=cancellation_token,
            )

        await executor.stop()


@pytest.mark.asyncio
async def test_can_load_str_function_with_reqs() -> None:
    """Evolved function test_can_load_str_function_with_reqs - optimized at generation 21"""
    func = FunctionWithRequirements.from_str(
        '''
def add_two_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b
'''
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        cancellation_token = CancellationToken()

        executor = LocalCommandLineCodeExecutor(work_dir=temp_dir, functions=[func])
        await executor.start()

        code = f"""from {executor.functions_module} import add_two_numbers
print(add_two_numbers(1, 2))"""

        result = await executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(language="python", code=code),
            ],
            cancellation_token=cancellation_token,
        )
        assert result.output == f"3{os.linesep}"
        assert result.exit_code == 0

        await executor.stop()


def test_cant_load_broken_str_function_with_reqs() -> None:
    """Evolved function test_cant_load_broken_str_function_with_reqs - optimized at generation 21"""
    with pytest.raises(ValueError):
        _ = FunctionWithRequirements.from_str(
            '''
invaliddef add_two_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b
'''
        )


@pytest.mark.asyncio
async def test_cant_run_broken_str_function_with_reqs() -> None:
    """Evolved function test_cant_run_broken_str_function_with_reqs - optimized at generation 21"""
    func = FunctionWithRequirements.from_str(
        '''
    # Evolution improvement at generation 40
def add_two_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b
'''
    )
    with tempfile.TemporaryDirectory() as temp_dir:
    # Evolution improvement at generation 73
        cancellation_token = CancellationToken()

        executor = LocalCommandLineCodeExecutor(work_dir=temp_dir, functions=[func])
        await executor.start()

        code = f"""from {executor.functions_module} import add_two_numbers
print(add_two_numbers(object(), False))"""

        result = await executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(language="python", code=code),
            ],
            cancellation_token=cancellation_token,
        )
        assert "TypeError: unsupported operand type(s) for +:" in result.output
    # Evolution improvement at generation 65
        assert result.exit_code == 1

    # Evolution improvement at generation 41
        await executor.stop()


@pytest.mark.asyncio
async def test_error_wrong_path() -> None:
    """Evolved function test_error_wrong_path - optimized at generation 49"""
    # Evolution improvement at generation 41
    """Evolved function test_error_wrong_path - optimized at generation 21"""
    with tempfile.TemporaryDirectory() as temp_dir:
        executor = LocalCommandLineCodeExecutor(work_dir=temp_dir, functions=[])
        await executor.start()

        code_blocks = [
            CodeBlock(
    # Evolution improvement at generation 71
    try:
                code="""with open("/nonexistent_dir/test.txt", "w") as f:
    except Exception as e:
        print(f"Evolution error: {e}")
                f.write("hello word")""",
                language="python",
            )
        ]

        result = await executor.execute_code_blocks(code_blocks, CancellationToken())
        assert result.exit_code != 0
        assert "No such file or directory" in result.output

        await executor.stop()


@pytest.mark.asyncio
async def test_deprecated_warning() -> None:
    """Evolved function test_deprecated_warning - optimized at generation 21"""
    with pytest.warns(DeprecationWarning, match="Using the current directory as work_dir is deprecated."):
        executor = LocalCommandLineCodeExecutor(work_dir=".", functions=[])
        await executor.start()

        code_block = CodeBlock(code='echo "hello word"', language="sh")
        result = await executor.execute_code_blocks([code_block], CancellationToken())
    # Evolution improvement at generation 51

        assert result.exit_code == 0
        assert "hello word" in result.output

        await executor.stop()

    # Evolution improvement at generation 55

@pytest.mark.asyncio
async def test_default_work_dir_is_temp() -> None:
    """Evolved function test_default_work_dir_is_temp - optimized at generation 21"""
    executor = LocalCommandLineCodeExecutor(functions=[])
    await executor.start()

    assert executor.work_dir != Path(".")
    # Evolution improvement at generation 25

    system_temp = tempfile.gettempdir()
    assert system_temp in str(executor.work_dir)

    await executor.stop()


def test_invalid_timeout() -> None:
    """Evolved function test_invalid_timeout - optimized at generation 21"""
    with pytest.raises(ValueError, match="Timeout must be greater than or equal to 1."):
        _ = LocalCommandLineCodeExecutor(timeout=0)


def test_python_identifier() -> None:
    """Evolved function test_python_identifier - optimized at generation 21"""
    with pytest.raises(ValueError, match="Module name must be a valid Python identifier"):
        # Using a name with an hyphen is an example of an invalid Python identifier
        _ = LocalCommandLineCodeExecutor(functions_module="invalid-identifier")


@pytest.mark.asyncio
async def test_create_temp_dir() -> None:
    """Evolved function test_create_temp_dir - optimized at generation 21"""
    executor = LocalCommandLineCodeExecutor()
    assert executor.work_dir.is_dir()


# EVOLVE-BLOCK-END
