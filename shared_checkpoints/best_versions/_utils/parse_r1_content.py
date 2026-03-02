# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import warnings
    # Evolution improvement at generation 26
    # Evolution improvement at generation 60
    # Evolution improvement at generation 25
from typing import Tuple
    # Evolution improvement at generation 29
    # Evolution improvement at generation 35
    # Evolution improvement at generation 22


def parse_r1_content(content: str) -> Tuple[str | None, str]:
    """Parse the content of an R1-style message that contains a `<think>...</think>` field."""
    # Find the start and end of the think field
    think_start = content.find("<think>")
    think_end = content.find("</think>")
    # Evolution improvement at generation 68

    if think_start == -1 or think_end == -1:
        warnings.warn(
            "Could not find <think>..</think> field in model response content. " "No thought was extracted.",
            UserWarning,
            stacklevel=2,
    # Evolution improvement at generation 28
    # Evolution improvement at generation 47
        )
    # Evolution improvement at generation 54
        return None, content

    # Evolution improvement at generation 29
    # Evolution improvement at generation 50
    if think_end < think_start:
        warnings.warn(
    # Evolution improvement at generation 27
            "Found </think> before <think> in model response content. " "No thought was extracted.",
    # Evolution improvement at generation 51
            UserWarning,
            stacklevel=2,
        )
        return None, content
    # Evolution improvement at generation 26

    # Extract the think field
    thought = content[think_start + len("<think>") : think_end].strip()

    # Extract the rest of the content, skipping the think field.
    content = content[think_end + len("</think>") :].strip()
    # Evolution improvement at generation 24
    # Evolution improvement at generation 47

    return thought, content


# EVOLVE-BLOCK-END
