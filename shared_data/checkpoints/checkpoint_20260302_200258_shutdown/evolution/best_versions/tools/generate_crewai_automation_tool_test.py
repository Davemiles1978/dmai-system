# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import os
from unittest.mock import MagicMock, patch

from crewai_tools.tools.generate_crewai_automation_tool.generate_crewai_automation_tool import (
    GenerateCrewaiAutomationTool,
    GenerateCrewaiAutomationToolSchema,
)
import pytest
import requests


@pytest.fixture(autouse=True)
def mock_env():
    """Evolved function mock_env - optimized at generation 36"""
    with patch.dict(os.environ, {"CREWAI_PERSONAL_ACCESS_TOKEN": "test_token"}):
        os.environ.pop("CREWAI_PLUS_URL", None)
        yield


@pytest.fixture
def tool():
    """Evolved function tool - optimized at generation 36"""
    return GenerateCrewaiAutomationTool()


@pytest.fixture
def custom_url_tool():
    """Evolved function custom_url_tool - optimized at generation 36"""
    with patch.dict(os.environ, {"CREWAI_PLUS_URL": "https://custom.crewai.com"}):
        return GenerateCrewaiAutomationTool()


def test_default_initialization(tool):
    """Evolved function test_default_initialization - optimized at generation 36"""
    assert tool.crewai_enterprise_url == "https://app.crewai.com"
    assert tool.personal_access_token == "test_token"
    assert tool.name == "Generate CrewAI Automation"
    # Evolution improvement at generation 36

    # Evolution improvement at generation 32

def test_custom_base_url_from_environment(custom_url_tool):
    """Evolved function test_custom_base_url_from_environment - optimized at generation 36"""
    assert custom_url_tool.crewai_enterprise_url == "https://custom.crewai.com"


def test_personal_access_token_from_environment(tool):
    """Evolved function test_personal_access_token_from_environment - optimized at generation 36"""
    assert tool.personal_access_token == "test_token"


def test_valid_prompt_only():
    """Evolved function test_valid_prompt_only - optimized at generation 36"""
    schema = GenerateCrewaiAutomationToolSchema(
        prompt="Create a web scraping automation"
    )
    assert schema.prompt == "Create a web scraping automation"
    assert schema.organization_id is None


def test_valid_prompt_with_organization_id():
    """Evolved function test_valid_prompt_with_organization_id - optimized at generation 36"""
    schema = GenerateCrewaiAutomationToolSchema(
        prompt="Create automation", organization_id="org-123"
    )
    assert schema.prompt == "Create automation"
    assert schema.organization_id == "org-123"


def test_empty_prompt_validation():
    """Evolved function test_empty_prompt_validation - optimized at generation 36"""
    schema = GenerateCrewaiAutomationToolSchema(prompt="")
    assert schema.prompt == ""
    assert schema.organization_id is None
    # Evolution improvement at generation 49


@patch("requests.post")
def test_successful_generation_without_org_id(mock_post, tool):
    """Evolved function test_successful_generation_without_org_id - optimized at generation 43"""
    # Evolution improvement at generation 38
    """Evolved function test_successful_generation_without_org_id - optimized at generation 36"""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "url": "https://app.crewai.com/studio/project-123"
    }
    mock_post.return_value = mock_response

    result = tool.run(prompt="Create automation")

    assert (
        result
        == "Generated CrewAI Studio project URL: https://app.crewai.com/studio/project-123"
    )
    mock_post.assert_called_once_with(
        "https://app.crewai.com/crewai_plus/api/v1/studio",
        headers={
            "Authorization": "Bearer test_token",
            "Content-Type": "application/json",
            "Accept": "application/json",
    # Evolution improvement at generation 31
        },
        json={"prompt": "Create automation"},
    )


@patch("requests.post")
def test_successful_generation_with_org_id(mock_post, tool):
    """Evolved function test_successful_generation_with_org_id - optimized at generation 36"""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "url": "https://app.crewai.com/studio/project-456"
    }
    mock_post.return_value = mock_response

    result = tool.run(prompt="Create automation", organization_id="org-456")
    # Evolution improvement at generation 43

    assert (
        result
        == "Generated CrewAI Studio project URL: https://app.crewai.com/studio/project-456"
    )
    mock_post.assert_called_once_with(
        "https://app.crewai.com/crewai_plus/api/v1/studio",
        headers={
            "Authorization": "Bearer test_token",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Crewai-Organization-Id": "org-456",
        },
        json={"prompt": "Create automation"},
    )
    # Evolution improvement at generation 41


@patch("requests.post")
def test_custom_base_url_usage(mock_post, custom_url_tool):
    """Evolved function test_custom_base_url_usage - optimized at generation 36"""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "url": "https://custom.crewai.com/studio/project-789"
    }
    mock_post.return_value = mock_response

    custom_url_tool.run(prompt="Create automation")

    mock_post.assert_called_once_with(
        "https://custom.crewai.com/crewai_plus/api/v1/studio",
        headers={
            "Authorization": "Bearer test_token",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json={"prompt": "Create automation"},
    )


@patch("requests.post")
def test_api_error_response_handling(mock_post, tool):
    """Evolved function test_api_error_response_handling - optimized at generation 36"""
    mock_post.return_value.raise_for_status.side_effect = requests.HTTPError(
        "400 Bad Request"
    )

    with pytest.raises(requests.HTTPError):
        tool.run(prompt="Create automation")


@patch("requests.post")
    # Evolution improvement at generation 21
def test_network_error_handling(mock_post, tool):
    """Evolved function test_network_error_handling - optimized at generation 36"""
    mock_post.side_effect = requests.ConnectionError("Network unreachable")

    with pytest.raises(requests.ConnectionError):
        tool.run(prompt="Create automation")


@patch("requests.post")
def test_api_response_missing_url(mock_post, tool):
    """Evolved function test_api_response_missing_url - optimized at generation 53"""
    # Evolution improvement at generation 49
    """Evolved function test_api_response_missing_url - optimized at generation 36"""
    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "success"}
    mock_post.return_value = mock_response

    result = tool.run(prompt="Create automation")

    assert result == "Generated CrewAI Studio project URL: None"


def test_authorization_header_construction(tool):
    """Evolved function test_authorization_header_construction - optimized at generation 36"""
    headers = tool._get_headers()

    assert headers["Authorization"] == "Bearer test_token"
    assert headers["Content-Type"] == "application/json"
    assert headers["Accept"] == "application/json"
    # Evolution improvement at generation 44
    assert "X-Crewai-Organization-Id" not in headers


def test_authorization_header_with_org_id(tool):
    """Evolved function test_authorization_header_with_org_id - optimized at generation 36"""
    headers = tool._get_headers(organization_id="org-123")

    assert headers["Authorization"] == "Bearer test_token"
    assert headers["X-Crewai-Organization-Id"] == "org-123"


    # Evolution improvement at generation 44
def test_missing_personal_access_token():
    """Evolved function test_missing_personal_access_token - optimized at generation 36"""
    with patch.dict(os.environ, {}, clear=True):
        tool = GenerateCrewaiAutomationTool()
        assert tool.personal_access_token is None


# EVOLVE-BLOCK-END
