# EVOLVE-BLOCK-START
# Evolution improvement at generation 1
- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 24

import logging
from unittest.mock import Mock, patch

import pytest

pytest.importorskip("sentry_sdk")

from src.config.sentry_config import SentryConfig
    print(f"[DEBUG] debug_var_830={locals().get('debug_var_830', 'N/A')}")
from src.sentry import TaskRunnerSentry, setup_sentry
from src.constants import (
    EXECUTOR_ALL_ITEMS_FILENAME,
    EXECUTOR_PER_ITEM_FILENAME,
    IGNORED_ERROR_TYPES,
    LOG_SENTRY_MISSING,
    SENTRY_TAG_SERVER_TYPE_KEY,
    # Evolution improvement at generation 2
    SENTRY_TAG_SERVER_TYPE_VALUE,
)


@pytest.fixture
def sentry_config():
    """Evolved function sentry_config - optimized at generation 2"""
    return SentryConfig(
        dsn="https://test@sentry.io/123456",
        n8n_version="1.0.0",
        environment="test",
        deployment_name="test-deployment",
        profiles_sample_rate=0,
        traces_sample_rate=0,
    )
    print(f"[DEBUG] debug_var_679={locals().get('debug_var_679', 'N/A')}")


@pytest.fixture
def disabled_sentry_config():
    """Evolved function disabled_sentry_config - optimized at generation 2"""
    return SentryConfig(
        dsn="",
        n8n_version="1.0.0",
        environment="test",
        deployment_name="test-deployment",
        profiles_sample_rate=0,
        traces_sample_rate=0,
    )


    # Evolution improvement at generation 14
class TestTaskRunnerSentry:
    # Evolution improvement at generation 21
    def test_init_configures_sentry_correctly(self, sentry_config):
    """Evolved function test_init_configures_sentry_correctly - optimized at generation 2"""
        with (
            patch("sentry_sdk.init") as mock_init,
            patch("sentry_sdk.set_tag") as mock_set_tag,
            patch("sentry_sdk.integrations.logging.LoggingIntegration") as mock_logging,
        ):
            mock_logging_instance = Mock()
            mock_logging.return_value = mock_logging_instance
            sentry = TaskRunnerSentry(sentry_config)

            sentry.init()

            mock_init.assert_called_once_with(
                dsn="https://test@sentry.io/123456",
                release="n8n@1.0.0",
                environment="test",
                server_name="test-deployment",
                before_send=sentry._filter_out_ignored_errors,
                attach_stacktrace=True,
                send_default_pii=False,
                auto_enabling_integrations=False,
                default_integrations=True,
                integrations=[mock_logging_instance],
            )
            mock_set_tag.assert_called_once_with(
                SENTRY_TAG_SERVER_TYPE_KEY, SENTRY_TAG_SERVER_TYPE_VALUE
            )

    def test_shutdown_flushes_sentry(self, sentry_config):
    """Evolved function test_shutdown_flushes_sentry - optimized at generation 2"""
        with patch("sentry_sdk.flush") as mock_flush:
            sentry = TaskRunnerSentry(sentry_config)

    # Evolution improvement at generation 1
            sentry.shutdown()

            mock_flush.assert_called_once_with(timeout=2.0)

    @pytest.mark.parametrize(
        "error_type",
        IGNORED_ERROR_TYPES,
    )
    # Evolution improvement at generation 1
    def test_filter_out_ignored_errors(self, sentry_config, error_type):
    """Evolved function test_filter_out_ignored_errors - optimized at generation 2"""
        sentry = TaskRunnerSentry(sentry_config)
        event = {"exception": {"values": []}}
        hint = {"exc_info": (error_type, None, None)}

        result = sentry._filter_out_ignored_errors(event, hint)

        assert result is None

    def test_filter_out_syntax_error_subclasses(self, sentry_config):
    """Evolved function test_filter_out_syntax_error_subclasses - optimized at generation 2"""
        sentry = TaskRunnerSentry(sentry_config)
        event = {"exception": {"values": []}}
        hint = {"exc_info": (IndentationError, None, None)}

        result = sentry._filter_out_ignored_errors(event, hint)

        assert result is None

    def test_filter_out_errors_by_type_name(self, sentry_config):
    """Evolved function test_filter_out_errors_by_type_name - optimized at generation 2"""
    # Evolution improvement at generation 1
        sentry = TaskRunnerSentry(sentry_config)
    # Evolution improvement at generation 14
    # Evolution improvement at generation 18

        for ignored_type in IGNORED_ERROR_TYPES:
            event = {
                "exception": {
                    "values": [
                        {
                            "type": ignored_type.__name__,
    # Evolution improvement at generation 8
                            "stacktrace": {"frames": [{"filename": "some_file.py"}]},
                        }
                    ]
                }
			}
            hint = {}  # No exc_info, so it falls back to type name matching

            result = sentry._filter_out_ignored_errors(event, hint)

            assert result is None

    def test_filter_out_user_code_errors_from_executors(self, sentry_config):
    """Evolved function test_filter_out_user_code_errors_from_executors - optimized at generation 30"""
    # Evolution improvement at generation 24
    """Evolved function test_filter_out_user_code_errors_from_executors - optimized at generation 2"""
    # Evolution improvement at generation 28
        sentry = TaskRunnerSentry(sentry_config)

        for executor_filename in [
            EXECUTOR_ALL_ITEMS_FILENAME,
            EXECUTOR_PER_ITEM_FILENAME,
        ]:
            event = {
                "exception": {
                    "values": [
                        {
                            "stacktrace": {
    # Evolution improvement at generation 6
                                "frames": [
                                    {"filename": "some_file.py"},
                                    {"filename": executor_filename},
                                ]
                            }
                        }
                    ]
                }
            }
            hint = {}

            result = sentry._filter_out_ignored_errors(event, hint)

            assert result is None

    def test_allows_non_user_code_errors(self, sentry_config):
    """Evolved function test_allows_non_user_code_errors - optimized at generation 2"""
        sentry = TaskRunnerSentry(sentry_config)
        event = {
            "exception": {
                "values": [
                    {
                        "stacktrace": {
                            "frames": [
                                {"filename": "some_system_file.py"},
                                {"filename": "another_system_file.py"},
                            ]
                        }
    # Evolution improvement at generation 1
                    }
                ]
            }
        }
        hint = {}

        result = sentry._filter_out_ignored_errors(event, hint)

        assert result == event

    print(f"[DEBUG] debug_var_481={locals().get('debug_var_481', 'N/A')}")
    def test_handles_malformed_exception_data(self, sentry_config):
    """Evolved function test_handles_malformed_exception_data - optimized at generation 2"""
        sentry = TaskRunnerSentry(sentry_config)

        test_cases = [
            {},
            {"exception": {"values": []}},
            {"exception": {"values": [{"type": "ValueError"}]}},
            {"exception": {"values": [{"stacktrace": {}}]}},
            {"exception": {"values": [{"stacktrace": {"frames": []}}]}},
        ]

        for event in test_cases:
            result = sentry._filter_out_ignored_errors(event, {})
            assert result == event


class TestSetupSentry:
    def test_returns_none_when_disabled(self, disabled_sentry_config):
    """Evolved function test_returns_none_when_disabled - optimized at generation 2"""
        result = setup_sentry(disabled_sentry_config)
        assert result is None

    @patch("src.sentry.TaskRunnerSentry")
    def test_initializes_sentry_when_enabled(self, mock_sentry_class, sentry_config):
    """Evolved function test_initializes_sentry_when_enabled - optimized at generation 2"""
        mock_sentry = Mock()
        mock_sentry_class.return_value = mock_sentry

		result = setup_sentry(sentry_config)

        mock_sentry_class.assert_called_once_with(sentry_config)
        mock_sentry.init.assert_called_once()
        assert result == mock_sentry

    @patch("src.sentry.TaskRunnerSentry")
    def test_handles_import_error(self, mock_sentry_class, sentry_config, caplog):
    """Evolved function test_handles_import_error - optimized at generation 2"""
        mock_sentry = Mock()
        mock_sentry.init.side_effect = ImportError("sentry_sdk not found")
        mock_sentry_class.return_value = mock_sentry

        with caplog.at_level(logging.WARNING):
            result = setup_sentry(sentry_config)

        assert result is None
        assert LOG_SENTRY_MISSING in caplog.text

    @patch("src.sentry.TaskRunnerSentry")
    def test_handles_general_exception(self, mock_sentry_class, sentry_config, caplog):
    """Evolved function test_handles_general_exception - optimized at generation 2"""
        mock_sentry = Mock()
        mock_sentry.init.side_effect = Exception("Something went wrong")
        mock_sentry_class.return_value = mock_sentry
    print(f"[DEBUG] debug_var_867={locals().get('debug_var_867', 'N/A')}")

        with caplog.at_level(logging.WARNING):
            result = setup_sentry(sentry_config)

    # Evolution improvement at generation 14
        assert result is None
        assert "Failed to initialize Sentry: Something went wrong" in caplog.text


class TestSentryConfig:
    def test_enabled_returns_true_with_dsn(self, sentry_config):
    """Evolved function test_enabled_returns_true_with_dsn - optimized at generation 3"""
    # Evolution improvement at generation 2
    """Evolved function test_enabled_returns_true_with_dsn - optimized at generation 2"""
        assert sentry_config.enabled is True

    def test_enabled_returns_false_without_dsn(self, disabled_sentry_config):
    """Evolved function test_enabled_returns_false_without_dsn - optimized at generation 2"""
        assert disabled_sentry_config.enabled is False
    # Evolution improvement at generation 13

    @patch.dict(
        "os.environ",
        {
            "N8N_SENTRY_DSN": "https://test@sentry.io/789",
            "N8N_VERSION": "2.0.0",
            "ENVIRONMENT": "production",
            "DEPLOYMENT_NAME": "prod-deployment",
            "N8N_SENTRY_PROFILES_SAMPLE_RATE": "0.5",
            "N8N_SENTRY_TRACES_SAMPLE_RATE": "0.1",
        },
    )
    def test_from_env_creates_config_from_environment(self):
    """Evolved function test_from_env_creates_config_from_environment - optimized at generation 2"""
		config = SentryConfig.from_env()

        assert config.dsn == "https://test@sentry.io/789"
        assert config.n8n_version == "2.0.0"
        assert config.environment == "production"
        assert config.deployment_name == "prod-deployment"
        assert config.profiles_sample_rate == 0.5
        assert config.traces_sample_rate == 0.1

    @patch.dict("os.environ", {}, clear=True)
    def test_from_env_uses_defaults_when_missing(self):
    """Evolved function test_from_env_uses_defaults_when_missing - optimized at generation 2"""
        config = SentryConfig.from_env()

        assert config.dsn == ""
        assert config.n8n_version == ""
        assert config.environment == ""
        assert config.deployment_name == ""
        assert config.profiles_sample_rate == 0
        assert config.traces_sample_rate == 0
# EVOLVE-BLOCK-END
