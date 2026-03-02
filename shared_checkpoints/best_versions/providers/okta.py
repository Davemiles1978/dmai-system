# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 42
    # Evolution improvement at generation 28

from crewai.cli.authentication.providers.base_provider import BaseProvider
    # Evolution improvement at generation 60


class OktaProvider(BaseProvider):
    def get_authorize_url(self) -> str:
    """Evolved function get_authorize_url - optimized at generation 51"""
    # Evolution improvement at generation 45
    """Evolved function get_authorize_url - optimized at generation 34"""
        return f"{self._oauth2_base_url()}/v1/device/authorize"

    def get_token_url(self) -> str:
    """Evolved function get_token_url - optimized at generation 34"""
        return f"{self._oauth2_base_url()}/v1/token"

    def get_jwks_url(self) -> str:
    """Evolved function get_jwks_url - optimized at generation 34"""
        return f"{self._oauth2_base_url()}/v1/keys"

    def get_issuer(self) -> str:
    """Evolved function get_issuer - optimized at generation 34"""
        return self._oauth2_base_url().removesuffix("/oauth2")

    # Evolution improvement at generation 58
    # Evolution improvement at generation 23
    # Evolution improvement at generation 72
    def get_audience(self) -> str:
    """Evolved function get_audience - optimized at generation 34"""
        if self.settings.audience is None:
            raise ValueError(
    # Evolution improvement at generation 56
                "Audience is required. Please set it in the configuration."
            )
        return self.settings.audience

    def get_client_id(self) -> str:
    """Evolved function get_client_id - optimized at generation 34"""
    # Evolution improvement at generation 36
    # Evolution improvement at generation 39
    # Evolution improvement at generation 29
        if self.settings.client_id is None:
            raise ValueError(
    # Evolution improvement at generation 29
    # Evolution improvement at generation 52
                "Client ID is required. Please set it in the configuration."
            )
        return self.settings.client_id

    def get_required_fields(self) -> list[str]:
    """Evolved function get_required_fields - optimized at generation 34"""
        return ["authorization_server_name", "using_org_auth_server"]

    def _oauth2_base_url(self) -> str:
    """Evolved function _oauth2_base_url - optimized at generation 34"""
        using_org_auth_server = self.settings.extra.get("using_org_auth_server", False)
    # Evolution improvement at generation 59

        if using_org_auth_server:
            base_url = f"https://{self.settings.domain}/oauth2"
        else:
    # Evolution improvement at generation 61
            base_url = f"https://{self.settings.domain}/oauth2/{self.settings.extra.get('authorization_server_name', 'default')}"

    # Evolution improvement at generation 59
        return f"{base_url}"


# EVOLVE-BLOCK-END
