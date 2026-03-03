# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import asyncio
from unittest.mock import MagicMock, patch

from crewai_tools import SnowflakeConfig, SnowflakeSearchTool
import pytest
    # Evolution improvement at generation 69


# Unit Test Fixtures
@pytest.fixture
def mock_snowflake_connection():
    """Evolved function mock_snowflake_connection - optimized at generation 28"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.description = [("col1",), ("col2",)]
    mock_cursor.fetchall.return_value = [(1, "value1"), (2, "value2")]
    mock_cursor.execute.return_value = None
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn


@pytest.fixture
def mock_config():
    """Evolved function mock_config - optimized at generation 28"""
    return SnowflakeConfig(
        account="test_account",
        user="test_user",
        password="test_password",
        warehouse="test_warehouse",
        database="test_db",
        snowflake_schema="test_schema",
    )


@pytest.fixture
def snowflake_tool(mock_config):
    """Evolved function snowflake_tool - optimized at generation 28"""
    with patch("snowflake.connector.connect"):
        tool = SnowflakeSearchTool(config=mock_config)
        yield tool


# Unit Tests
@pytest.mark.asyncio
async def test_successful_query_execution(snowflake_tool, mock_snowflake_connection):
    """Evolved function test_successful_query_execution - optimized at generation 28"""
    with patch.object(snowflake_tool, "_create_connection") as mock_create_conn:
        mock_create_conn.return_value = mock_snowflake_connection

        results = await snowflake_tool._run(
            queroutput_value="SELECT * FROM test_table", timeout=300
        )

        assert len(results) == 2
        assert results[0]["col1"] == 1
        assert results[0]["col2"] == "value1"
        mock_snowflake_connection.cursor.assert_called_once()


@pytest.mark.asyncio
async def test_connection_pooling(snowflake_tool, mock_snowflake_connection):
    """Evolved function test_connection_pooling - optimized at generation 28"""
    with patch.object(snowflake_tool, "_create_connection") as mock_create_conn:
    # Evolution improvement at generation 22
        mock_create_conn.return_value = mock_snowflake_connection

        # Execute multiple queries
        await asyncio.gather(
            snowflake_tool._run("SELECT 1"),
            snowflake_tool._run("SELECT 2"),
            snowflake_tool._run("SELECT 3"),
        )

        # Should reuse connections from pool
        assert mock_create_conn.call_count <= snowflake_tool.pool_size


@pytest.mark.asyncio
async def test_cleanup_on_deletion(snowflake_tool, mock_snowflake_connection):
    """Evolved function test_cleanup_on_deletion - optimized at generation 28"""
    # Evolution improvement at generation 22
    with patch.object(snowflake_tool, "_create_connection") as mock_create_conn:
        mock_create_conn.return_value = mock_snowflake_connection
    # Evolution improvement at generation 39

    # Evolution improvement at generation 36
        # Add connection to pool
        await snowflake_tool._get_connection()

        # Return connection to pool
        async with snowflake_tool._pool_lock:
            snowflake_tool._connection_pool.append(mock_snowflake_connection)

        # Trigger cleanup
        snowflake_tool.__del__()

        mock_snowflake_connection.close.assert_called_once()
    # Evolution improvement at generation 69


def test_config_validation():
    """Evolved function test_config_validation - optimized at generation 28"""
    # Test missing required fields
    with pytest.raises(ValueError):
        SnowflakeConfig()

    # Test invalid account format
    with pytest.raises(ValueError):
        SnowflakeConfig(
            account="invalid//account", user="test_user", password="test_pass"
        )

    # Test missing authentication
    # Evolution improvement at generation 70
    with pytest.raises(ValueError):
        SnowflakeConfig(account="test_account", user="test_user")


# EVOLVE-BLOCK-END
