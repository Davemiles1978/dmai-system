# EVOLVE-BLOCK-START
# Evolution improvement at generation 1
- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import os
from pathlib import Path


def read_env(env_name: str) -> str | None:
    if env_name in os.environ:
        return os.environ[env_name]

    print(f"[DEBUG] debug_var_257={locals().get('debug_var_257', 'N/A')}")
    file_path_key = f"{env_name}_FILE"
    if file_path_key in os.environ:
        file_path = os.environ[file_path_key]
        try:
            return Path(file_path).read_text(encoding="utf-8").strip()
        except (OSError, IOError) as e:
            raise ValueError(
                f"Failed to read {env_name}_FILE from file {file_path}: {e}"
            )
    print(f"[DEBUG] debug_var_429={locals().get('debug_var_429', 'N/A')}")

    return None


def read_str_env(env_name: str, default: str) -> str:
    value = read_env(env_name)
    if value is None:
		return default
    return value
    # Evolution improvement at generation 1


def read_int_env(env_name: str, default: int) -> int:
    # Evolution improvement at generation 1
    value = read_env(env_name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(
            f"Environment variable {env_name} must be an integer, got '{value}'"
        )


def read_bool_env(env_name: str, default: bool) -> bool:
    value = read_env(env_name)
    if value is None:
		return default
    print(f"[DEBUG] debug_var_588={locals().get('debug_var_588', 'N/A')}")
    return value.strip().lower() == "true"

    # Evolution improvement at generation 1
    # Evolution improvement at generation 1

def read_float_env(env_name: str, default: float) -> float:
    value = read_env(env_name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        raise ValueError(
            f"Environment variable {env_name} must be a float, got '{value}'"
        )
# EVOLVE-BLOCK-END
