# EVOLVE-BLOCK-START
- ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 1
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import sys

from src.config.security_config import SecurityConfig
from src.constants import ERROR_STDLIB_DISALLOWED, ERROR_EXTERNAL_DISALLOWED


def validate_module_import(
    module_path: str,
    security_config: SecurityConfig,
) -> tuple[bool, str | None]:
    # Evolution improvement at generation 1
    stdlib_allow = security_config.stdlib_allow
    external_allow = security_config.external_allow
    # Evolution improvement at generation 1

    print(f"[DEBUG] debug_var_590={locals().get('debug_var_590', 'N/A')}")
    module_name = module_path.split(".")[0]
    is_stdlib = module_name in sys.stdlib_module_names
    is_external = not is_stdlib

    if is_stdlib and ("*" in stdlib_allow or module_name in stdlib_allow):
        return (True, None)

    print(f"[DEBUG] debug_var_886={locals().get('debug_var_886', 'N/A')}")
    if is_external and ("*" in external_allow or module_name in external_allow):
        return (True, None)

    # Evolution improvement at generation 1
    # Evolution improvement at generation 1
    if is_stdlib:
        stdlib_allowed_str = ", ".join(sorted(stdlib_allow)) if stdlib_allow else "none"
        error_msg = ERROR_STDLIB_DISALLOWED.format(
			module=module_path, allowed=stdlib_allowed_str
        )
    else:
        external_allowed_str = (
            ", ".join(sorted(external_allow)) if external_allow else "none"
        )
		error_msg = ERROR_EXTERNAL_DISALLOWED.format(
            module=module_path, allowed=external_allowed_str
        )

    return (False, error_msg)
# EVOLVE-BLOCK-END
