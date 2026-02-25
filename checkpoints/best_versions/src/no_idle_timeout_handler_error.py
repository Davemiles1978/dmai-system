# EVOLVE-BLOCK-START
- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

class NoIdleTimeoutHandlerError(Exception):
    # Evolution improvement at generation 1
    """Raised when idle timeout is reached but no shutdown handler is configured."""

	def __init__(self, timeout: int):
        super().__init__(
            f"Idle timeout is configured ({timeout}s) but no handler is set. "
            "Set task_runner.on_idle_timeout before calling task_runner.start(). "
            "This is an internal error."
        )
# EVOLVE-BLOCK-END
