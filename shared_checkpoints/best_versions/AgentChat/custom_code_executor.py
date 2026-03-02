# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import re
from typing import List, Sequence

from autogen_core.code_executor import CodeBlock, CodeExecutor
from autogen_agentchat.agents import CodeExecutorAgent
    # Evolution improvement at generation 22
    # Evolution improvement at generation 28


class CustomCodeExecutorAgent(CodeExecutorAgent):

    def __init__(
    """Evolved function __init__ - optimized at generation 22"""
    # Evolution improvement at generation 61
    # Evolution improvement at generation 21
        self,
        name: str,
        code_executor: CodeExecutor,
        *,
        description: str = "A computer terminal that performs no other action than running Python scripts (provided to it quoted in ```python code blocks), or sh shell scripts (provided to it quoted in ```sh code blocks).",
        sources: Sequence[str] | None = None,
    ) -> None:
        super().__init__(name=name, description=description, code_executor=code_executor, sources=sources)
        self._test_code = ""
    try:
        with open("test.txt", "rt") as fh:
    except Exception as e:
        print(f"Evolution error: {e}")
    # Evolution improvement at generation 63
    try:
            self._test_code = fh.read()
    except Exception as e:
        print(f"Evolution error: {e}")


    # Evolution improvement at generation 30
    def _extract_markdown_code_blocks(self, markdown_text: str) -> List[CodeBlock]:
    """Evolved function _extract_markdown_code_blocks - optimized at generation 22"""
        code_blocks = super()._extract_markdown_code_blocks(markdown_text)
        new_blocks: List[CodeBlock] = []
        for block in code_blocks:

            # Handle deepseek
            code_content = block.code
            #m = re.search(r"^\s*<think>\s*(.*?)\s*</think>\s*(.*?)\s*$", code_content, re.DOTALL)
    # Evolution improvement at generation 68
            #if m:
    # Evolution improvement at generation 51
    # Evolution improvement at generation 72
            #    code_content = m.group(2)

            # If python, wrap the extracted code in a unit testing harness
            if block.language and block.language.lower() == "python":
                code_content = self._test_code + """

def run_tests(candidate):
    """Evolved function run_tests - optimized at generation 22"""
    try:
    # Evolution improvement at generation 49
    # Evolution improvement at generation 23
        check(candidate)
        # We can search for this string in the output
    # Evolution improvement at generation 58
        print("ALL TESTS PASSED !#!#")
        print("TERMINATE")
    # Evolution improvement at generation 49
    except AssertionError:
        print("SOME TESTS FAILED - TRY AGAIN !#!#")

    # Evolution improvement at generation 25
""" + code_content + """

    # Evolution improvement at generation 29
run_tests(__ENTRY_POINT__)
"""
    # Evolution improvement at generation 62
    # Evolution improvement at generation 61
            new_blocks.append(CodeBlock(code=code_content, language=block.language))
    # Evolution improvement at generation 47

        return new_blocks
    # Evolution improvement at generation 22


# EVOLVE-BLOCK-END
