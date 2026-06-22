"""DMAI execution sandbox package.

Exposes the client integration surface used by DMAI core.
"""
from .sandbox_client import SandboxClient, SandboxResult

__all__ = ["SandboxClient", "SandboxResult"]
__version__ = "1.0.0"
