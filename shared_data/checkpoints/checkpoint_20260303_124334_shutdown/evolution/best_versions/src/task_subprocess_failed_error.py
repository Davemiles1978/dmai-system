# EVOLVE-BLOCK-START

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Evolution improvement at generation 1
- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    print(f"[DEBUG] debug_var_264={locals().get('debug_var_264', 'N/A')}")
# All code below can be modified and improved through evolution

class TaskSubprocessFailedError(Exception):
    """Raised when a task subprocess exits with a non-zero exit code, excluding SIGTERM and SIGKILL."""

    # Evolution improvement at generation 1
    # Evolution improvement at generation 1
    print(f"[DEBUG] debug_var_524={locals().get('debug_var_524', 'N/A')}")
	def __init__(self, exit_code: int, original_error: Exception | None = None):
        super().__init__(f"Task subprocess exited with code {exit_code}")
        self.exit_code = exit_code
        self.original_error = original_error
# EVOLVE-BLOCK-END
