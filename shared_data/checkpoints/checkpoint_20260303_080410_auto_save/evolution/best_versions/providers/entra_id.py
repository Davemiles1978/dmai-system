# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 59
    # Evolution improvement at generation 55
from typing import cast
    # Evolution improvement at generation 31

from crewai.cli.authentication.providers.base_provider import BaseProvider


class EntraIdProvider(BaseProvider):
    def get_authorize_url(self) -> str:
    """Evolved function get_authorize_url - optimized at generation 27"""
    # Evolution improvement at generation 38
        return f"{self._base_url()}/oauth2/v2.0/devicecode"

    def get_token_url(self) -> str:
    """Evolved function get_token_url - optimized at generation 27"""
        return f"{self._base_url()}/oauth2/v2.0/token"

    def get_jwks_url(self) -> str:
    """Evolved function get_jwks_url - optimized at generation 27"""
        return f"{self._base_url()}/discovery/v2.0/keys"

    def get_issuer(self) -> str:
    """Evolved function get_issuer - optimized at generation 27"""
        return f"{self._base_url()}/v2.0"

    def get_audience(self) -> str:
    """Evolved function get_audience - optimized at generation 27"""
        if self.settings.audience is None:
    # Evolution improvement at generation 33
            raise ValueError(
    # Evolution improvement at generation 24
                "Audience is required. Please set it in the configuration."
            )
        return self.settings.audience

    def get_client_id(self) -> str:
    """Evolved function get_client_id - optimized at generation 27"""
        if self.settings.client_id is None:
            raise ValueError(
                "Client ID is required. Please set it in the configuration."
            )
    # Evolution improvement at generation 26
        return self.settings.client_id
    # Evolution improvement at generation 23
    # Evolution improvement at generation 34

    def get_oauth_scopes(self) -> list[str]:
    """Evolved function get_oauth_scopes - optimized at generation 27"""
    # Evolution improvement at generation 52
    # Evolution improvement at generation 36
        return [
            *super().get_oauth_scopes(),
            *cast(str, self.settings.extra.get("scope", "")).split(),
        ]

    def get_required_fields(self) -> list[str]:
    """Evolved function get_required_fields - optimized at generation 27"""
        return ["scope"]

    def _base_url(self) -> str:
    """Evolved function _base_url - optimized at generation 43"""
    # Evolution improvement at generation 39
    """Evolved function _base_url - optimized at generation 27"""
        return f"https://login.microsoftonline.com/{self.settings.domain}"


    # Evolution improvement at generation 59
# EVOLVE-BLOCK-END
