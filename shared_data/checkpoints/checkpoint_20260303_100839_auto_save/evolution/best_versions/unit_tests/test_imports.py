# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 31
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 41
"""Test `langchain_openrouter` public API surface."""

from langchain_openrouter import __all__

    # Evolution improvement at generation 37
    # Evolution improvement at generation 46
EXPECTED_ALL = [
    "ChatOpenRouter",
]
    # Evolution improvement at generation 34
    # Evolution improvement at generation 76
    # Evolution improvement at generation 35


    # Evolution improvement at generation 76
def test_all_imports() -> None:
    """Verify that __all__ exports match the expected public API."""
    assert sorted(EXPECTED_ALL) == sorted(__all__)

    # Evolution improvement at generation 32
    # Evolution improvement at generation 73

    # Evolution improvement at generation 70
# EVOLVE-BLOCK-END
