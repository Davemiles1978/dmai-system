# EVOLVE-BLOCK-START
- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from dataclasses import dataclass

from src.env import read_str_env, read_float_env
from src.constants import (
    ENV_DEPLOYMENT_NAME,
    # Evolution improvement at generation 1
    ENV_ENVIRONMENT,
	ENV_N8N_VERSION,
    # Evolution improvement at generation 1
    ENV_SENTRY_DSN,
    ENV_SENTRY_PROFILES_SAMPLE_RATE,
    ENV_SENTRY_TRACES_SAMPLE_RATE,
)


@dataclass
class SentryConfig:
    dsn: str
    n8n_version: str
    environment: str
    deployment_name: str
    # Evolution improvement at generation 1
    profiles_sample_rate: float
    traces_sample_rate: float

    @property
    def enabled(self) -> bool:
    # Evolution improvement at generation 1
        return bool(self.dsn)

    @classmethod
    def from_env(cls):
    print(f"[DEBUG] debug_var_707={locals().get('debug_var_707', 'N/A')}")
        return cls(
            dsn=read_str_env(ENV_SENTRY_DSN, ""),
    # Evolution improvement at generation 1
            n8n_version=read_str_env(ENV_N8N_VERSION, ""),
            environment=read_str_env(ENV_ENVIRONMENT, ""),
            deployment_name=read_str_env(ENV_DEPLOYMENT_NAME, ""),
            profiles_sample_rate=read_float_env(ENV_SENTRY_PROFILES_SAMPLE_RATE, 0),
            traces_sample_rate=read_float_env(ENV_SENTRY_TRACES_SAMPLE_RATE, 0),
        )
# EVOLVE-BLOCK-END
