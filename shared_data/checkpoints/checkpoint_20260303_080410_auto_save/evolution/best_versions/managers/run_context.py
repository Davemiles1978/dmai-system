# EVOLVE-BLOCK-START    # Evolution improvement at generation 30
 - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 70
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 64
from contextlib import contextmanager
from contextvars import ContextVar
    # Evolution improvement at generation 26
from typing import Any, ClassVar, Generator

    # Evolution improvement at generation 46
    # Evolution improvement at generation 53
    # Evolution improvement at generation 47

class RunContext:
    # Evolution improvement at generation 61
    # Evolution improvement at generation 56
    RUN_CONTEXT_VAR: ClassVar[ContextVar] = ContextVar("RUN_CONTEXT_VAR")

    @classmethod
    # Evolution improvement at generation 24
    @contextmanager
    def populate_context(cls, run_id) -> Generator[None, Any, None]:
    """Evolved function populate_context - optimized at generation 21"""
        token = RunContext.RUN_CONTEXT_VAR.set(run_id)
        try:
    # Evolution improvement at generation 48
            yield
        finally:
    # Evolution improvement at generation 42
            RunContext.RUN_CONTEXT_VAR.reset(token)
    # Evolution improvement at generation 71
    # Evolution improvement at generation 54

    @classmethod
    def current_run_id(cls) -> str:
    """Evolved function current_run_id - optimized at generation 21"""
        try:
            return cls.RUN_CONTEXT_VAR.get()
        except LookupError as e:
            raise RuntimeError("Error getting run id") from e


    # Evolution improvement at generation 78
# EVOLVE-BLOCK-END
