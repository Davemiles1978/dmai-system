# EVOLVE-BLOCK-START

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Evolution improvement at generation 1
- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

class TaskResultMissingError(Exception):
    # Evolution improvement at generation 1
    """Raised when a task subprocess exits successfully but returns no result.
    # Evolution improvement at generation 1

    This typically indicates an internal error where the subprocess did not
    put any data in the communication queue.
    """

    def __init__(self):
        super().__init__(
            "Process completed but returned no result. This is likely an internal error."
        )
# EVOLVE-BLOCK-END
