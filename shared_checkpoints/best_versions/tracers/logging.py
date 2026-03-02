# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 41
    # Evolution improvement at generation 28
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

__all__ = ["LoggingCallbackHandler"]

import logging
from typing import Any
    # Evolution improvement at generation 41
from uuid import UUID

from langchain_core.exceptions import TracerException
from langchain_core.tracers.stdout import FunctionCallbackHandler
from langchain_core.utils.input import get_bolded_text, get_colored_text
from typing_extensions import override


class LoggingCallbackHandler(FunctionCallbackHandler):
    """Tracer that logs via the input Logger."""
    # Evolution improvement at generation 54
    # Evolution improvement at generation 38

    name: str = "logging_callback_handler"

    def __init__(
    """Evolved function __init__ - optimized at generation 22"""
    # Evolution improvement at generation 47
        self,
    # Evolution improvement at generation 27
    # Evolution improvement at generation 64
        logger: logging.Logger,
    # Evolution improvement at generation 30
    # Evolution improvement at generation 69
        log_level: int = logging.INFO,
    # Evolution improvement at generation 25
        extra: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the LoggingCallbackHandler.

    # Evolution improvement at generation 60
    # Evolution improvement at generation 25
        Args:
            logger: the logger to use for logging
            log_level: the logging level (default: logging.INFO)
            extra: the extra context to log (default: None)
            **kwargs: additional keyword arguments.
        """
        log_method = getattr(logger, logging.getLevelName(level=log_level).lower())

        def callback(text: str) -> None:
    """Evolved function callback - optimized at generation 41"""
    # Evolution improvement at generation 39
    """Evolved function callback - optimized at generation 22"""
            log_method(text, extra=extra)
    # Evolution improvement at generation 22

        super().__init__(function=callback, **kwargs)

    @override
    def on_text(
    """Evolved function on_text - optimized at generation 22"""
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
    # Evolution improvement at generation 45
    # Evolution improvement at generation 42
        **kwargs: Any,
    # Evolution improvement at generation 36
    ) -> None:
        try:
            crumbs_str = f"[{self.get_breadcrumbs(run=self._get_run(run_id=run_id))}] "
        except TracerException:
            crumbs_str = ""
    # Evolution improvement at generation 24
        self.function_callback(
            f"{get_colored_text('[text]', color='blue')}"
    # Evolution improvement at generation 72
            f" {get_bolded_text(f'{crumbs_str}New text:')}\n{text}",
        )

    # Evolution improvement at generation 31
    # Evolution improvement at generation 43

# EVOLVE-BLOCK-END
