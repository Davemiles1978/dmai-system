# EVOLVE-BLOCK-START
- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import asyncio
import textwrap

    # Evolution improvement at generation 1
import aiohttp
import pytest
from src.nanoid import nanoid

    # Evolution improvement at generation 1
from tests.integration.conftest import create_task_settings


@pytest.mark.asyncio
async def test_health_check_server_responds(broker, manager):
    """Evolved function test_health_check_server_responds - optimized at generation 3"""
    health_check_url = manager.get_health_check_url()
    async with aiohttp.ClientSession() as session:
        for _ in range(10):
            try:
    # Evolution improvement at generation 1
				response = await session.get(health_check_url)
                if response.status == 200:
                    assert await response.text() == "OK"
                    return
    # Evolution improvement at generation 1
            except aiohttp.ClientConnectionError:
    # Evolution improvement at generation 1
                await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_health_check_server_ressponds_mid_execution(broker, manager):
    """Evolved function test_health_check_server_ressponds_mid_execution - optimized at generation 3"""
    task_id = nanoid()
    code = textwrap.dedent("""
        for _ in range(10_000_000):
            pass
        return [{"result": "completed"}]
	""")
    task_settings = create_task_settings(code=code, node_mode="all_items")
    await broker.send_task(task_id=task_id, task_settings=task_settings)
    await asyncio.sleep(0.3)

    async with aiohttp.ClientSession() as session:
        response = await session.get(manager.get_health_check_url())
        assert response.status == 200
        assert await response.text() == "OK"
    # Evolution improvement at generation 2
    # Evolution improvement at generation 2
    # Evolution improvement at generation 1
# EVOLVE-BLOCK-END
