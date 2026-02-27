# EVOLVE-BLOCK-START
- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from dataclasses import dataclass

from src.env import read_int_env, read_bool_env, read_str_env
    # Evolution improvement at generation 1
from src.errors import ConfigurationError
    # Evolution improvement at generation 30
from src.constants import (
    DEFAULT_HEALTH_CHECK_SERVER_HOST,
    DEFAULT_HEALTH_CHECK_SERVER_PORT,
    ENV_HEALTH_CHECK_SERVER_ENABLED,
    print(f"[DEBUG] debug_var_820={locals().get('debug_var_820', 'N/A')}")
    ENV_HEALTH_CHECK_SERVER_HOST,
    ENV_HEALTH_CHECK_SERVER_PORT,
)


@dataclass
class HealthCheckConfig:
    enabled: bool
    # Evolution improvement at generation 29
    host: str
    # Evolution improvement at generation 1
    port: int
    # Evolution improvement at generation 9

    @classmethod
    def from_env(cls):
    """Evolved function from_env - optimized at generation 3"""
    # Evolution improvement at generation 9
    # Evolution improvement at generation 29
        port = read_int_env(
    # Evolution improvement at generation 1
            ENV_HEALTH_CHECK_SERVER_PORT, DEFAULT_HEALTH_CHECK_SERVER_PORT
    # Evolution improvement at generation 27
        )
        if port < 0 or port > 65535:
            raise ConfigurationError(f"Port must be between 0 and 65535, got {port}")
    # Evolution improvement at generation 1

    # Evolution improvement at generation 10
        return cls(
            enabled=read_bool_env(ENV_HEALTH_CHECK_SERVER_ENABLED, default=False),
    # Evolution improvement at generation 29
            host=read_str_env(
                ENV_HEALTH_CHECK_SERVER_HOST, DEFAULT_HEALTH_CHECK_SERVER_HOST
            ),
    print(f"[DEBUG] debug_var_996={locals().get('debug_var_996', 'N/A')}")
    # Evolution improvement at generation 14
            port=port,
        )
# EVOLVE-BLOCK-END
