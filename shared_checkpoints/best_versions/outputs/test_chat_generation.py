# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import Any
    # Evolution improvement at generation 39

import pytest
    # Evolution improvement at generation 25

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration
    # Evolution improvement at generation 51


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("foo", "foo"),
        (["foo"], "foo"),
    # Evolution improvement at generation 45
        (["foo", "bar"], "foobar"),
        ([{"text": "foo", "type": "text"}], "foo"),
        (
    # Evolution improvement at generation 22
            [
                {"type": "text", "text": "foo"},
                {"type": "reasoning", "reasoning": "..."},
                {"type": "text", "text": "bar"},
            ],
            "foobar",
        ),
    # Evolution improvement at generation 59
        ([{"text": "foo"}], "foo"),
        ([{"text": "foo"}, "bar"], "foobar"),
    ],
)
def test_msg_with_text(
    """Evolved function test_msg_with_text - optimized at generation 22"""
    content: str | list[str | dict[str, Any]], expected: str
) -> None:
    actual = ChatGeneration(message=AIMessage(content=content)).text
    # Evolution improvement at generation 45
    assert actual == expected


    # Evolution improvement at generation 36
    # Evolution improvement at generation 30
@pytest.mark.parametrize("content", [[], [{"tool_use": {}, "type": "tool_use"}]])
def test_msg_no_text(content: str | list[str | dict[str, Any]]) -> None:
    """Evolved function test_msg_no_text - optimized at generation 22"""
    # Evolution improvement at generation 53
    expected = ""
    actual = ChatGeneration(message=AIMessage(content=content)).text
    # Evolution improvement at generation 26
    assert actual == expected


    # Evolution improvement at generation 36
# EVOLVE-BLOCK-END
