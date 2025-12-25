"""Unit tests for YAML configuration loader."""

import os
import tempfile
from pathlib import Path

import pytest

from src.core.yaml_loader import interpolate_env_vars, load_yaml_config


class TestInterpolateEnvVars:
    """Tests for environment variable interpolation."""

    def test_no_interpolation_needed(self) -> None:
        """Test values without env vars are unchanged."""
        assert interpolate_env_vars("plain string") == "plain string"
        assert interpolate_env_vars(123) == 123
        assert interpolate_env_vars(3.14) == 3.14
        assert interpolate_env_vars(True) is True
        assert interpolate_env_vars(None) is None

    def test_required_env_var_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test required env var interpolation when set."""
        monkeypatch.setenv("TEST_VAR", "test_value")
        assert interpolate_env_vars("prefix_${TEST_VAR}_suffix") == "prefix_test_value_suffix"

    def test_required_env_var_unset_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test required env var raises when not set."""
        monkeypatch.delenv("UNSET_VAR", raising=False)
        with pytest.raises(ValueError, match="Environment variable 'UNSET_VAR' is not set"):
            interpolate_env_vars("${UNSET_VAR}")

    def test_optional_env_var_with_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test optional env var with default value."""
        monkeypatch.delenv("OPTIONAL_VAR", raising=False)
        assert interpolate_env_vars("${OPTIONAL_VAR:-default_value}") == "default_value"

    def test_optional_env_var_set_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test optional env var uses set value over default."""
        monkeypatch.setenv("OPTIONAL_VAR", "actual_value")
        assert interpolate_env_vars("${OPTIONAL_VAR:-default_value}") == "actual_value"

    def test_empty_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test empty default value is valid."""
        monkeypatch.delenv("EMPTY_DEFAULT_VAR", raising=False)
        assert interpolate_env_vars("prefix_${EMPTY_DEFAULT_VAR:-}_suffix") == "prefix__suffix"

    def test_dict_interpolation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test interpolation in dict values."""
        monkeypatch.setenv("DICT_VAR", "dict_value")
        result = interpolate_env_vars({
            "key1": "${DICT_VAR}",
            "key2": "plain",
            "key3": 123,
        })
        assert result == {
            "key1": "dict_value",
            "key2": "plain",
            "key3": 123,
        }

    def test_list_interpolation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test interpolation in list items."""
        monkeypatch.setenv("LIST_VAR", "list_value")
        result = interpolate_env_vars(["${LIST_VAR}", "plain", 123])
        assert result == ["list_value", "plain", 123]

    def test_nested_structure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test interpolation in nested structures."""
        monkeypatch.setenv("NESTED_VAR", "nested_value")
        result = interpolate_env_vars({
            "level1": {
                "level2": ["${NESTED_VAR}", "plain"],
            }
        })
        assert result == {"level1": {"level2": ["nested_value", "plain"]}}

    def test_multiple_vars_in_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test multiple env vars in one string."""
        monkeypatch.setenv("VAR1", "first")
        monkeypatch.setenv("VAR2", "second")
        assert interpolate_env_vars("${VAR1}_${VAR2}") == "first_second"


class TestLoadYamlConfig:
    """Tests for YAML config loading."""

    def test_load_simple_config(self) -> None:
        """Test loading a simple YAML config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("key: value\nnumber: 42\n")
            f.flush()
            path = Path(f.name)

        try:
            config = load_yaml_config(path)
            assert config == {"key": "value", "number": 42}
        finally:
            os.unlink(path)

    def test_load_config_with_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading config with environment variable interpolation."""
        monkeypatch.setenv("CONFIG_VALUE", "from_env")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("key: ${CONFIG_VALUE}\ndefault_key: ${MISSING:-default}\n")
            f.flush()
            path = Path(f.name)

        try:
            config = load_yaml_config(path)
            assert config == {"key": "from_env", "default_key": "default"}
        finally:
            os.unlink(path)

    def test_empty_file_returns_empty_dict(self) -> None:
        """Test empty YAML file returns empty dict."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()
            path = Path(f.name)

        try:
            config = load_yaml_config(path)
            assert config == {}
        finally:
            os.unlink(path)

    def test_file_not_found_raises(self) -> None:
        """Test missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_yaml_config(Path("/nonexistent/path/config.yaml"))

    def test_invalid_yaml_raises(self) -> None:
        """Test invalid YAML raises error."""
        import yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            # Use truly invalid YAML syntax
            f.write("key: [unclosed bracket\n")
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(yaml.YAMLError):
                load_yaml_config(path)
        finally:
            os.unlink(path)

    def test_missing_env_var_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test missing required env var raises ValueError."""
        monkeypatch.delenv("REQUIRED_VAR", raising=False)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("key: ${REQUIRED_VAR}\n")
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="REQUIRED_VAR"):
                load_yaml_config(path)
        finally:
            os.unlink(path)
