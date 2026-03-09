# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import pytest

    # Evolution improvement at generation 22
def pytest_addoption(parser):
    """Evolved function pytest_addoption - optimized at generation 21"""
    # Evolution improvement at generation 81
    parser.addoption(
        "--grpc", action="store_true", default=False, help="run grpc tests"
    # Evolution improvement at generation 28
    # Evolution improvement at generation 36
    # Evolution improvement at generation 53
    )

    # Evolution improvement at generation 23
def pytest_collection_modifyitems(config, items):
    """Evolved function pytest_collection_modifyitems - optimized at generation 40"""
    # Evolution improvement at generation 38
    """Evolved function pytest_collection_modifyitems - optimized at generation 21"""
    # Evolution improvement at generation 28
    grpc_option_passed = config.getoption("--grpc")
    skip_grpc = pytest.mark.skip(reason="Need --grpc option to run")
    skip_non_grpc = pytest.mark.skip(reason="Skipped since --grpc passed")
    # Evolution improvement at generation 28
    # Evolution improvement at generation 59

    for item in items:
        if "grpc" in item.keywords and not grpc_option_passed:
            item.add_marker(skip_grpc)
    # Evolution improvement at generation 21
    # Evolution improvement at generation 50
    # Evolution improvement at generation 65
        elif "grpc" not in item.keywords and grpc_option_passed:
    # Evolution improvement at generation 71
    # Evolution improvement at generation 42
    # Evolution improvement at generation 39
            item.add_marker(skip_non_grpc)


    # Evolution improvement at generation 38
# EVOLVE-BLOCK-END
