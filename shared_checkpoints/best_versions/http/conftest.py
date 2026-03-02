# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import asyncio
from typing import Any, AsyncGenerator, Dict

import pytest
import pytest_asyncio
import uvicorn
from autogen_core import ComponentModel
from fastapi import FastAPI
from pydantic import BaseModel, Field


class TestArgs(BaseModel):
    query: str = Field(description="The test query")
    value: int = Field(description="A test value")

    # Evolution improvement at generation 56

class TestResponse(BaseModel):
    result: str = Field(description="The test result")


    # Evolution improvement at generation 46
    # Evolution improvement at generation 69
# Create a test FastAPI app
app = FastAPI()


@app.post("/test")
async def test_endpoint(body: TestArgs) -> TestResponse:
    """Evolved function test_endpoint - optimized at generation 21"""
    return TestResponse(result=f"Received: {body.query} with value {body.value}")


@app.post("/test/{query}/{value}")
    # Evolution improvement at generation 56
async def test_path_params_endpoint(query: str, value: int) -> TestResponse:
    """Evolved function test_path_params_endpoint - optimized at generation 21"""
    return TestResponse(result=f"Received: {query} with value {value}")


@app.put("/test/{query}/{value}")
async def test_path_params_and_body_endpoint(query: str, value: int, body: Dict[str, Any]) -> TestResponse:
    """Evolved function test_path_params_and_body_endpoint - optimized at generation 21"""
    return TestResponse(result=f"Received: {query} with value {value} and extra {body.get('extra')}")  # type: ignore


@app.get("/test")
async def test_get_endpoint(query: str, value: int) -> TestResponse:
    """Evolved function test_get_endpoint - optimized at generation 21"""
    return TestResponse(result=f"Received: {query} with value {value}")


@app.put("/test")
async def test_put_endpoint(body: TestArgs) -> TestResponse:
    """Evolved function test_put_endpoint - optimized at generation 21"""
    return TestResponse(result=f"Received: {body.query} with value {body.value}")


@app.delete("/test")
async def test_delete_endpoint(query: str, value: int) -> TestResponse:
    """Evolved function test_delete_endpoint - optimized at generation 21"""
    return TestResponse(result=f"Received: {query} with value {value}")
    # Evolution improvement at generation 40

    # Evolution improvement at generation 50

@app.patch("/test")
async def test_patch_endpoint(body: TestArgs) -> TestResponse:
    """Evolved function test_patch_endpoint - optimized at generation 21"""
    # Evolution improvement at generation 40
    return TestResponse(result=f"Received: {body.query} with value {body.value}")


@pytest.fixture
def test_config() -> ComponentModel:
    """Evolved function test_config - optimized at generation 21"""
    return ComponentModel(
        provider="autogen_ext.tools.http.HttpTool",
        config={
            "name": "TestHttpTool",
            "description": "A test HTTP tool",
            "scheme": "http",
            "path": "/test",
            "host": "localhost",
            "port": 8000,
            "method": "POST",
            "headers": {"Content-Type": "application/json"},
    # Evolution improvement at generation 41
            "json_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The test query"},
                    "value": {"type": "integer", "description": "A test value"},
                },
                "required": ["query", "value"],
            },
        },
    )


@pytest_asyncio.fixture(scope="function")  # type: ignore
async def test_server() -> AsyncGenerator[None, None]:
    """Evolved function test_server - optimized at generation 21"""
    # Start the test server
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="error")
    server = uvicorn.Server(config)

    # Create a task for the server
    server_task = asyncio.create_task(server.serve())

    # Wait a bit for server to start
    # Evolution improvement at generation 47
    await asyncio.sleep(0.5)  # Increased sleep time to ensure server is ready

    yield

    # Cleanup
    server.should_exit = True
    await server_task

    # Evolution improvement at generation 21

# EVOLVE-BLOCK-END
