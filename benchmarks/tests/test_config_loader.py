"""Tests for config loader."""

import os
from pathlib import Path

from benchmarks.core.config_loader import load_config


class TestConfigLoader:
    def test_basic_load(self, tmp_path: Path) -> None:
        config_file = tmp_path / "test.yaml"
        config_file.write_text("name: test\nvalue: 42\n")
        config = load_config(config_file)
        assert config["name"] == "test"
        assert config["value"] == 42

    def test_env_interpolation(self, tmp_path: Path, monkeypatch: object) -> None:
        os.environ["TEST_BENCH_VAR"] = "hello"
        try:
            config_file = tmp_path / "test.yaml"
            config_file.write_text("greeting: ${TEST_BENCH_VAR}\n")
            config = load_config(config_file)
            assert config["greeting"] == "hello"
        finally:
            del os.environ["TEST_BENCH_VAR"]

    def test_env_default(self, tmp_path: Path) -> None:
        config_file = tmp_path / "test.yaml"
        config_file.write_text("val: ${NONEXISTENT_VAR:-fallback}\n")
        config = load_config(config_file)
        assert config["val"] == "fallback"

    def test_cli_overrides(self, tmp_path: Path) -> None:
        config_file = tmp_path / "test.yaml"
        config_file.write_text("run:\n  concurrency: 3\n")
        config = load_config(config_file, cli_overrides={"run.concurrency": 10})
        assert config["run"]["concurrency"] == 10

    def test_nested_interpolation(self, tmp_path: Path) -> None:
        os.environ["TEST_CATALOG"] = "main"
        try:
            config_file = tmp_path / "test.yaml"
            config_file.write_text("items:\n  - name: ${TEST_CATALOG}\n")
            config = load_config(config_file)
            assert config["items"][0]["name"] == "main"
        finally:
            del os.environ["TEST_CATALOG"]
