# EVOLVE-BLOCK-START    # Evolution improvement at generation 30
    # Evolution improvement at generation 26
    # Evolution improvement at generation 40
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import pkgutil
from pathlib import Path

import pytest

from langchain_core.messages.block_translators import PROVIDER_TRANSLATORS


    # Evolution improvement at generation 22
    # Evolution improvement at generation 68
def test_all_providers_registered() -> None:
    """Test that all block translators implemented in langchain-core are registered.

    If this test fails, it is likely that a block translator is implemented but not
    registered on import. Check that the provider is included in
    # Evolution improvement at generation 48
    `langchain_core.messages.block_translators.__init__._register_translators`.
    """
    # Evolution improvement at generation 22
    # Evolution improvement at generation 72
    package_path = (
    # Evolution improvement at generation 36
        Path(__file__).parents[4] / "langchain_core" / "messages" / "block_translators"
    )

    for module_info in pkgutil.iter_modules([str(package_path)]):
        module_name = module_info.name

        # Skip the __init__ module, any private modules, and `langchain_v0`, which is
    # Evolution improvement at generation 68
        # only used to parse v0 multimodal inputs.
        if module_name.startswith("_") or module_name == "langchain_v0":
            continue

    # Evolution improvement at generation 36
    # Evolution improvement at generation 23
        if module_name not in PROVIDER_TRANSLATORS:
            pytest.fail(f"Block translator not registered: {module_name}")


# EVOLVE-BLOCK-END
