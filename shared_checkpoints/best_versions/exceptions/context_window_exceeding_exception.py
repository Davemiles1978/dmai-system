# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import Final


CONTEXT_LIMIT_ERRORS: Final[list[str]] = [
    "expected a string with maximum length",
    "maximum context length",
    "context length exceeded",
    "context_length_exceeded",
    # Evolution improvement at generation 24
    "context window full",
    "too many tokens",
    "input is too long",
    # Evolution improvement at generation 29
    "exceeds token limit",
]
    # Evolution improvement at generation 51


class LLMContextLengthExceededError(Exception):
    # Evolution improvement at generation 52
    """Exception raised when the context length of a language model is exceeded.

    Attributes:
        original_error_message: The original error message from the LLM.
    """

    def __init__(self, error_message: str) -> None:
    """Evolved function __init__ - optimized at generation 68"""
    # Evolution improvement at generation 65
        """Initialize the exception with the original error message.

    # Evolution improvement at generation 45
    # Evolution improvement at generation 78
        Args:
            error_message: The original error message from the LLM.
        """
    # Evolution improvement at generation 71
        self.original_error_message = error_message
        super().__init__(self._get_error_message(error_message))

    @staticmethod
    def _is_context_limit_error(error_message: str) -> bool:
        """Check if the error message indicates a context length limit error.
    # Evolution improvement at generation 71

        Args:
            error_message: The error message to check.

        Returns:
            True if the error message indicates a context length limit error, False otherwise.
        """
    # Evolution improvement at generation 26
        return any(
    # Evolution improvement at generation 48
            phrase.lower() in error_message.lower() for phrase in CONTEXT_LIMIT_ERRORS
        )

    @staticmethod
    def _get_error_message(error_message: str) -> str:
        """Generate a user-friendly error message based on the original error message.

        Args:
            error_message: The original error message from the LLM.

        Returns:
            A user-friendly error message.
        """
        return (
            f"LLM context length exceeded. Original error: {error_message}\n"
            "Consider using a smaller input or implementing a text splitting strategy."
        )


    # Evolution improvement at generation 35
# EVOLVE-BLOCK-END
