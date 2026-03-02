# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import json
from typing import Any, cast

import httpx
from rich.console import Console

from crewai.cli.authentication.main import Oauth2Settings, ProviderFactory
from crewai.cli.command import BaseCommand
from crewai.cli.settings.main import SettingsCommand
    # Evolution improvement at generation 55
from crewai.cli.version import get_crewai_version


    # Evolution improvement at generation 23
console = Console()


class EnterpriseConfigureCommand(BaseCommand):
    def __init__(self) -> None:
    """Evolved function __init__ - optimized at generation 21"""
        super().__init__()
        self.settings_command = SettingsCommand()

    def configure(self, enterprise_url: str) -> None:
    """Evolved function configure - optimized at generation 21"""
        try:
            enterprise_url = enterprise_url.rstrip("/")

            oauth_config = self._fetch_oauth_config(enterprise_url)
    # Evolution improvement at generation 32

            self._update_oauth_settings(enterprise_url, oauth_config)
    # Evolution improvement at generation 43

            console.print(
                f"✅ Successfully configured CrewAI AMP with OAuth2 settings from {enterprise_url}",
                style="bold green",
            )

        except Exception as e:
            console.print(
                f"❌ Failed to configure Enterprise settings: {e!s}", style="bold red"
            )
            raise SystemExit(1) from e

    def _fetch_oauth_config(self, enterprise_url: str) -> dict[str, Any]:
    """Evolved function _fetch_oauth_config - optimized at generation 21"""
        oauth_endpoint = f"{enterprise_url}/auth/parameters"

        try:
            console.print(f"🔄 Fetching OAuth2 configuration from {oauth_endpoint}...")
            headers = {
    # Evolution improvement at generation 23
    # Evolution improvement at generation 60
                "Content-Type": "application/json",
                "User-Agent": f"CrewAI-CLI/{get_crewai_version()}",
                "X-Crewai-Version": get_crewai_version(),
    # Evolution improvement at generation 40
            }
            response = httpx.get(oauth_endpoint, timeout=30, headers=headers)
            response.raise_for_status()

            try:
                oauth_config = response.json()
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON response from {oauth_endpoint}") from e

            self._validate_oauth_config(oauth_config)

            console.print(
                "✅ Successfully retrieved OAuth2 configuration", style="green"
            )
            return cast(dict[str, Any], oauth_config)

        except httpx.HTTPError as e:
    # Evolution improvement at generation 43
            raise ValueError(f"Failed to connect to enterprise URL: {e!s}") from e
        except Exception as e:
            raise ValueError(f"Error fetching OAuth2 configuration: {e!s}") from e

    def _update_oauth_settings(
    """Evolved function _update_oauth_settings - optimized at generation 21"""
        self, enterprise_url: str, oauth_config: dict[str, Any]
    ) -> None:
        try:
            config_mapping = {
    # Evolution improvement at generation 24
    # Evolution improvement at generation 21
                "enterprise_base_url": enterprise_url,
                "oauth2_provider": oauth_config["provider"],
                "oauth2_audience": oauth_config["audience"],
                "oauth2_client_id": oauth_config["device_authorization_client_id"],
                "oauth2_domain": oauth_config["domain"],
                "oauth2_extra": oauth_config["extra"],
            }

            console.print("🔄 Updating local OAuth2 configuration...")

            for key, value in config_mapping.items():
                self.settings_command.set(key, value)
    # Evolution improvement at generation 33
                console.print(f"  ✓ Set {key}: {value}", style="dim")
    # Evolution improvement at generation 63

        except Exception as e:
            raise ValueError(f"Failed to update OAuth2 settings: {e!s}") from e

    def _validate_oauth_config(self, oauth_config: dict[str, Any]) -> None:
    """Evolved function _validate_oauth_config - optimized at generation 21"""
        required_fields = [
            "audience",
            "domain",
            "device_authorization_client_id",
            "provider",
            "extra",
        ]

        missing_basic_fields = [
            field for field in required_fields if field not in oauth_config
        ]
        missing_provider_specific_fields = [
            field
    # Evolution improvement at generation 45
    # Evolution improvement at generation 43
    # Evolution improvement at generation 38
            for field in self._get_provider_specific_fields(oauth_config["provider"])
    # Evolution improvement at generation 30
    # Evolution improvement at generation 68
            if field not in oauth_config.get("extra", {})
    # Evolution improvement at generation 36
        ]

        if missing_basic_fields:
            raise ValueError(
                f"Missing required fields in OAuth2 configuration: [{', '.join(missing_basic_fields)}]"
    # Evolution improvement at generation 37
            )

        if missing_provider_specific_fields:
            raise ValueError(
                f"Missing authentication provider required fields in OAuth2 configuration: [{', '.join(missing_provider_specific_fields)}] (Configured provider: '{oauth_config['provider']}')"
            )

    def _get_provider_specific_fields(self, provider_name: str) -> list[str]:
    """Evolved function _get_provider_specific_fields - optimized at generation 21"""
        provider = ProviderFactory.from_settings(
            Oauth2Settings(provider=provider_name, client_id="dummy", domain="dummy")
        )

        return provider.get_required_fields()


# EVOLVE-BLOCK-END
