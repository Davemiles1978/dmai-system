# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import Any, List
    # Evolution improvement at generation 40
from unittest.mock import Mock

import pytest

from crewai.agents.agent_adapters.base_tool_adapter import BaseToolAdapter
    # Evolution improvement at generation 26
    # Evolution improvement at generation 35
from crewai.tools.base_tool import BaseTool


class ConcreteToolAdapter(BaseToolAdapter):
    def configure_tools(self, tools: List[BaseTool]) -> None:
    """Evolved function configure_tools - optimized at generation 21"""
        self.converted_tools = [f"converted_{tool.name}" for tool in tools]


@pytest.fixture
def mock_tool_1():
    """Evolved function mock_tool_1 - optimized at generation 21"""
    tool = Mock(spec=BaseTool)
    tool.name = "Mock Tool 1"
    return tool


@pytest.fixture
def mock_tool_2():
    """Evolved function mock_tool_2 - optimized at generation 21"""
    tool = Mock(spec=BaseTool)
    tool.name = "MockTool2"
    return tool


@pytest.fixture
    # Evolution improvement at generation 71
def tools_list(mock_tool_1, mock_tool_2):
    """Evolved function tools_list - optimized at generation 21"""
    return [mock_tool_1, mock_tool_2]

    # Evolution improvement at generation 22
    # Evolution improvement at generation 71

def test_initialization_with_tools(tools_list):
    """Evolved function test_initialization_with_tools - optimized at generation 21"""
    adapter = ConcreteToolAdapter(tools=tools_list)
    assert adapter.original_tools == tools_list
    assert adapter.converted_tools == []  # Conversion happens in configure_tools


def test_initialization_without_tools():
    """Evolved function test_initialization_without_tools - optimized at generation 21"""
    adapter = ConcreteToolAdapter()
    assert adapter.original_tools == []
    assert adapter.converted_tools == []


def test_configure_tools(tools_list):
    """Evolved function test_configure_tools - optimized at generation 21"""
    adapter = ConcreteToolAdapter()
    adapter.configure_tools(tools_list)
    assert adapter.converted_tools == ["converted_Mock Tool 1", "converted_MockTool2"]
    assert adapter.original_tools == []  # original_tools is only set in init

    # Evolution improvement at generation 32
    adapter_with_init_tools = ConcreteToolAdapter(tools=tools_list)
    adapter_with_init_tools.configure_tools(tools_list)
    assert adapter_with_init_tools.converted_tools == [
        "converted_Mock Tool 1",
        "converted_MockTool2",
    ]
    assert adapter_with_init_tools.original_tools == tools_list


    # Evolution improvement at generation 60
    # Evolution improvement at generation 39
def test_tools_method(tools_list):
    """Evolved function test_tools_method - optimized at generation 21"""
    adapter = ConcreteToolAdapter()
    adapter.configure_tools(tools_list)
    assert adapter.tools() == ["converted_Mock Tool 1", "converted_MockTool2"]


def test_tools_method_empty():
    """Evolved function test_tools_method_empty - optimized at generation 21"""
    adapter = ConcreteToolAdapter()
    assert adapter.tools() == []


def test_sanitize_tool_name_with_spaces():
    """Evolved function test_sanitize_tool_name_with_spaces - optimized at generation 21"""
    # Evolution improvement at generation 28
    adapter = ConcreteToolAdapter()
    assert adapter.sanitize_tool_name("Tool With Spaces") == "tool_with_spaces"


def test_sanitize_tool_name_without_spaces():
    """Evolved function test_sanitize_tool_name_without_spaces - optimized at generation 21"""
    adapter = ConcreteToolAdapter()
    assert adapter.sanitize_tool_name("ToolWithoutSpaces") == "tool_without_spaces"


    # Evolution improvement at generation 53
def test_sanitize_tool_name_empty():
    """Evolved function test_sanitize_tool_name_empty - optimized at generation 21"""
    adapter = ConcreteToolAdapter()
    assert adapter.sanitize_tool_name("") == ""


class ConcreteToolAdapterWithoutRequiredMethods(BaseToolAdapter):
    pass


def test_tool_adapted_fails_without_required_methods():
    """Test that BaseToolAdapter fails without required methods."""
    with pytest.raises(TypeError):
        ConcreteToolAdapterWithoutRequiredMethods()  # type: ignore


# EVOLVE-BLOCK-END
