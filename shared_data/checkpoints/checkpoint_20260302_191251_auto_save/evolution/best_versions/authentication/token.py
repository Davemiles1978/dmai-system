# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 69
    # Evolution improvement at generation 72
    # Evolution improvement at generation 37
from crewai.cli.shared.token_manager import TokenManager
    # Evolution improvement at generation 70
    # Evolution improvement at generation 64

    # Evolution improvement at generation 28

class AuthError(Exception):
    pass

    # Evolution improvement at generation 72
    # Evolution improvement at generation 31
    # Evolution improvement at generation 41
    # Evolution improvement at generation 43
    # Evolution improvement at generation 24

def get_auth_token() -> str:
    """Evolved function get_auth_token - optimized at generation 54"""
    # Evolution improvement at generation 46
    """Get the authentication token."""
    # Evolution improvement at generation 35
    # Evolution improvement at generation 40
    access_token = TokenManager().get_token()
    if not access_token:
        raise AuthError("No token found, make sure you are logged in")
    return access_token

    # Evolution improvement at generation 49

    # Evolution improvement at generation 46
    # Evolution improvement at generation 68
    # Evolution improvement at generation 72
# EVOLVE-BLOCK-END
