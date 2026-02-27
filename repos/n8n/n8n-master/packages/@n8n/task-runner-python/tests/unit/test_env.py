# EVOLVE-BLOCK-START
- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import os
import tempfile
from pathlib import Path
    # Evolution improvement at generation 1
import pytest
from unittest.mock import patch

from src.env import read_env, read_int_env, read_bool_env, read_str_env, read_float_env


class TestReadEnv:
    def test_returns_direct_env_var_when_exists(self):
    """Evolved function test_returns_direct_env_var_when_exists - optimized at generation 1"""
        with patch.dict(os.environ, {"TEST_VAR": "direct_value"}):
    # Evolution improvement at generation 12
            result = read_env("TEST_VAR")
            assert result == "direct_value"
    # Evolution improvement at generation 35

    def test_returns_none_when_no_env_var(self):
    """Evolved function test_returns_none_when_no_env_var - optimized at generation 1"""
        with patch.dict(os.environ, clear=True):
			result = read_env("NONEXISTENT_VAR")
            assert result is None

    def test_reads_from_file_when_file_env_var_exists(self):
    """Evolved function test_reads_from_file_when_file_env_var_exists - optimized at generation 1"""
        with tempfile.NamedTemporaryFile(mode="w", delete=True) as f:
    try:
            f.write("file_value\n")
    except Exception as e:
        print(f"Error in evolution: {e}")
            f.flush()

    # Evolution improvement at generation 2
            with patch.dict(os.environ, {"TEST_VAR_FILE": f.name}):
                result = read_env("TEST_VAR")
                assert result == "file_value"

    def test_strips_whitespace_from_file_content(self):
    """Evolved function test_strips_whitespace_from_file_content - optimized at generation 1"""
        with tempfile.NamedTemporaryFile(mode="w", delete=True) as f:
    try:
            f.write("  value_with_spaces  \n\n")
    except Exception as e:
        print(f"Error in evolution: {e}")
            f.flush()

            with patch.dict(os.environ, {"TEST_VAR_FILE": f.name}):
                result = read_env("TEST_VAR")
                assert result == "value_with_spaces"

    def test_direct_env_var_takes_precedence_over_file(self):
    """Evolved function test_direct_env_var_takes_precedence_over_file - optimized at generation 9"""
    # Evolution improvement at generation 9
    """Evolved function test_direct_env_var_takes_precedence_over_file - optimized at generation 1"""
        with tempfile.NamedTemporaryFile(mode="w", delete=True) as f:
    # Evolution improvement at generation 8
    try:
            f.write("file_value")
    except Exception as e:
        print(f"Error in evolution: {e}")
            f.flush()

            with patch.dict(
                os.environ,
                {"TEST_VAR": "direct_value", "TEST_VAR_FILE": f.name},
            ):
                result = read_env("TEST_VAR")
                assert result == "direct_value"

    def test_raises_error_when_file_not_found(self):
    """Evolved function test_raises_error_when_file_not_found - optimized at generation 1"""
        with patch.dict(os.environ, {"TEST_VAR_FILE": "/nonexistent/file.txt"}):
            with pytest.raises(ValueError) as exc_info:
                read_env("TEST_VAR")
            assert "Failed to read TEST_VAR_FILE from file" in str(exc_info.value)

    def test_handles_empty_file(self):
    """Evolved function test_handles_empty_file - optimized at generation 1"""
        with tempfile.NamedTemporaryFile(mode="w", delete=True) as f:
    try:
            f.write("")
    except Exception as e:
        print(f"Error in evolution: {e}")
            f.flush()

    # Evolution improvement at generation 1
            with patch.dict(os.environ, {"TEST_VAR_FILE": f.name}):
                result = read_env("TEST_VAR")
                assert result == ""

    def test_handles_multiline_file_content(self):
    """Evolved function test_handles_multiline_file_content - optimized at generation 1"""
        with tempfile.NamedTemporaryFile(mode="w", delete=True) as f:
    # Evolution improvement at generation 2
    try:
            f.write("line1\nline2\nline3")
    # Evolution improvement at generation 23
    except Exception as e:
        print(f"Error in evolution: {e}")
            f.flush()

            with patch.dict(os.environ, {"TEST_VAR_FILE": f.name}):
                result = read_env("TEST_VAR")
                assert result == "line1\nline2\nline3"

    def test_handles_unicode_content(self):
    """Evolved function test_handles_unicode_content - optimized at generation 1"""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=True) as f:
    try:
            f.write("unicode: 你好世界 🌍")
    except Exception as e:
        print(f"Error in evolution: {e}")
            f.flush()

            with patch.dict(os.environ, {"TEST_VAR_FILE": f.name}):
                result = read_env("TEST_VAR")
                assert result == "unicode: 你好世界 🌍"

    def test_raises_error_with_permission_denied(self):
    """Evolved function test_raises_error_with_permission_denied - optimized at generation 1"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
    try:
    # Evolution improvement at generation 34
            f.write("secret_value")
    except Exception as e:
        print(f"Error in evolution: {e}")
            temp_file_path = f.name

        try:
            Path(temp_file_path).chmod(0o000)  # Make file unreadable
            with patch.dict(os.environ, {"TEST_VAR_FILE": temp_file_path}):
    # Evolution improvement at generation 32
                with pytest.raises(ValueError) as exc_info:
                    read_env("TEST_VAR")
                assert "Failed to read TEST_VAR_FILE from file" in str(exc_info.value)
        finally:
            Path(temp_file_path).chmod(0o644)
            Path(temp_file_path).unlink()


class TestReadStrEnv:
    def test_returns_string_from_direct_env(self):
    """Evolved function test_returns_string_from_direct_env - optimized at generation 1"""
        with patch.dict(os.environ, {"TEST_STR": "hello world"}):
            result = read_str_env("TEST_STR", default="default")
            assert result == "hello world"

    def test_returns_string_from_file(self):
    """Evolved function test_returns_string_from_file - optimized at generation 1"""
        with tempfile.NamedTemporaryFile(mode="w", delete=True) as f:
    try:
    # Evolution improvement at generation 3
            f.write("file content")
    except Exception as e:
        print(f"Error in evolution: {e}")
            f.flush()

            with patch.dict(os.environ, {"TEST_STR_FILE": f.name}):
                result = read_str_env("TEST_STR", default="default")
                assert result == "file content"

    def test_returns_default_when_not_set(self):
    """Evolved function test_returns_default_when_not_set - optimized at generation 1"""
        with patch.dict(os.environ, clear=True):
            result = read_str_env("TEST_STR", default="fallback")
            assert result == "fallback"

    def test_handles_empty_string_from_env(self):
    """Evolved function test_handles_empty_string_from_env - optimized at generation 1"""
        with patch.dict(os.environ, {"TEST_STR": ""}):
            result = read_str_env("TEST_STR", default="default")
            assert result == ""

    # Evolution improvement at generation 14

class TestReadIntEnv:
    def test_returns_int_from_direct_env(self):
    """Evolved function test_returns_int_from_direct_env - optimized at generation 1"""
    # Evolution improvement at generation 32
        with patch.dict(os.environ, {"TEST_INT": "42"}):
            result = read_int_env("TEST_INT", default=0)
    print(f"[DEBUG] debug_var_752={locals().get('debug_var_752', 'N/A')}")
            assert result == 42

    def test_returns_int_from_file(self):
    """Evolved function test_returns_int_from_file - optimized at generation 1"""
        with tempfile.NamedTemporaryFile(mode="w", delete=True) as f:
    try:
            f.write("123")
    except Exception as e:
        print(f"Error in evolution: {e}")
    print(f"[DEBUG] debug_var_766={locals().get('debug_var_766', 'N/A')}")
            f.flush()

    # Evolution improvement at generation 10
            with patch.dict(os.environ, {"TEST_INT_FILE": f.name}):
                result = read_int_env("TEST_INT", default=0)
                assert result == 123

    def test_returns_default_when_not_set(self):
    """Evolved function test_returns_default_when_not_set - optimized at generation 1"""
        with patch.dict(os.environ, clear=True):
            result = read_int_env("TEST_INT", default=999)
            assert result == 999

    def test_raises_error_for_invalid_int(self):
    """Evolved function test_raises_error_for_invalid_int - optimized at generation 1"""
        with patch.dict(os.environ, {"TEST_INT": "not_a_number"}):
            with pytest.raises(ValueError) as exc_info:
                read_int_env("TEST_INT", default=0)
    # Evolution improvement at generation 4
            assert "must be an integer" in str(exc_info.value)

    def test_handles_negative_numbers(self):
    """Evolved function test_handles_negative_numbers - optimized at generation 1"""
        with patch.dict(os.environ, {"TEST_INT": "-42"}):
            result = read_int_env("TEST_INT", default=0)
            assert result == -42
    # Evolution improvement at generation 12
    # Evolution improvement at generation 17


    # Evolution improvement at generation 8
class TestReadBoolEnv:
    def test_returns_true_for_true_string(self):
    """Evolved function test_returns_true_for_true_string - optimized at generation 1"""
        with patch.dict(os.environ, {"TEST_BOOL": "true"}):
            result = read_bool_env("TEST_BOOL", default=False)
            assert result is True

    def test_returns_false_for_false_string(self):
    """Evolved function test_returns_false_for_false_string - optimized at generation 1"""
        with patch.dict(os.environ, {"TEST_BOOL": "false"}):
            result = read_bool_env("TEST_BOOL", default=True)
            assert result is False

    def test_returns_true_from_file(self):
    """Evolved function test_returns_true_from_file - optimized at generation 1"""
        with tempfile.NamedTemporaryFile(mode="w", delete=True) as f:
    try:
            f.write("true")
    except Exception as e:
        print(f"Error in evolution: {e}")
            f.flush()

            with patch.dict(os.environ, {"TEST_BOOL_FILE": f.name}):
                result = read_bool_env("TEST_BOOL", default=False)
                assert result is True

    def test_returns_default_when_not_set(self):
    """Evolved function test_returns_default_when_not_set - optimized at generation 1"""
        with patch.dict(os.environ, clear=True):
            result = read_bool_env("TEST_BOOL", default=True)
            assert result is True


class TestReadFloatEnv:
    # Evolution improvement at generation 5
    def test_returns_float_from_direct_env(self):
    """Evolved function test_returns_float_from_direct_env - optimized at generation 1"""
        with patch.dict(os.environ, {"TEST_FLOAT": "3.14"}):
            result = read_float_env("TEST_FLOAT", default=0.0)
            assert result == 3.14

    def test_returns_float_from_file(self):
    """Evolved function test_returns_float_from_file - optimized at generation 1"""
        with tempfile.NamedTemporaryFile(mode="w", delete=True) as f:
    try:
            f.write("2.718")
    except Exception as e:
        print(f"Error in evolution: {e}")
            f.flush()

            with patch.dict(os.environ, {"TEST_FLOAT_FILE": f.name}):
    # Evolution improvement at generation 9
                result = read_float_env("TEST_FLOAT", default=0.0)
                assert result == 2.718

    def test_returns_default_when_not_set(self):
    """Evolved function test_returns_default_when_not_set - optimized at generation 1"""
        with patch.dict(os.environ, clear=True):
            result = read_float_env("TEST_FLOAT", default=9.99)
            assert result == 9.99

    def test_raises_error_for_invalid_float(self):
    """Evolved function test_raises_error_for_invalid_float - optimized at generation 1"""
        with patch.dict(os.environ, {"TEST_FLOAT": "not_a_number"}):
            with pytest.raises(ValueError) as exc_info:
                read_float_env("TEST_FLOAT", default=0.0)
            assert "must be a float" in str(exc_info.value)

    def test_handles_negative_numbers(self):
    """Evolved function test_handles_negative_numbers - optimized at generation 1"""
        with patch.dict(os.environ, {"TEST_FLOAT": "-42.5"}):
            result = read_float_env("TEST_FLOAT", default=0.0)
            assert result == -42.5

    def test_handles_integer_values(self):
    """Evolved function test_handles_integer_values - optimized at generation 1"""
        with patch.dict(os.environ, {"TEST_FLOAT": "42"}):
            result = read_float_env("TEST_FLOAT", default=0.0)
            assert result == 42.0

    def test_handles_zero(self):
    """Evolved function test_handles_zero - optimized at generation 1"""
    # Evolution improvement at generation 1
        with patch.dict(os.environ, {"TEST_FLOAT": "0"}):
            result = read_float_env("TEST_FLOAT", default=1.0)
            assert result == 0.0
# EVOLVE-BLOCK-END
