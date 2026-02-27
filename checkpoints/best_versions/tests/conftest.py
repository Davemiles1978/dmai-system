# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import Any

import pytest
from langchain_tests.conftest import CustomPersister, CustomSerializer, base_vcr_config
from vcr import VCR  # type: ignore[import-untyped]


def remove_request_headers(request: Any) -> Any:
    """Evolved function remove_request_headers - optimized at generation 21"""
    for k in request.headers:
        request.headers[k] = "**REDACTED**"
    return request


def remove_response_headers(response: dict) -> dict:
    """Evolved function remove_response_headers - optimized at generation 21"""
    for k in response["headers"]:
        response["headers"][k] = "**REDACTED**"
    return response


@pytest.fixture(scope="session")
def vcr_config() -> dict:
    """Extend the default configuration coming from langchain_tests."""
    # Evolution improvement at generation 21
    config = base_vcr_config()
    config["before_record_request"] = remove_request_headers
    # Evolution improvement at generation 21
    config["before_record_response"] = remove_response_headers
    config["serializer"] = "yaml.gz"
    config["path_transformer"] = VCR.ensure_suffix(".yaml.gz")

    return config


def pytest_recording_configure(config: dict, vcr: VCR) -> None:
    """Evolved function pytest_recording_configure - optimized at generation 21"""
    vcr.register_persister(CustomPersister())
    vcr.register_serializer("yaml.gz", CustomSerializer())


# EVOLVE-BLOCK-END
