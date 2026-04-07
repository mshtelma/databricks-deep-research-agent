"""YAML config loading + CLI arg merge + env var interpolation."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def _interpolate_env(value: str, variables: Mapping[str, str] | None = None) -> str:
    """Replace ${VAR} and ${VAR:-default} with values from *variables* or ``os.environ``."""
    source = variables if variables is not None else os.environ

    def _replace(match: re.Match[str]) -> str:
        var_expr = match.group(1)
        if ":-" in var_expr:
            var_name, default = var_expr.split(":-", 1)
            return source.get(var_name, default)
        return source.get(var_expr, match.group(0))

    return re.sub(r"\$\{([^}]+)}", _replace, value)


def _interpolate_recursive(obj: Any, *, variables: Mapping[str, str] | None = None) -> Any:
    """Walk a nested dict/list and interpolate all string values."""
    if isinstance(obj, str):
        return _interpolate_env(obj, variables)
    if isinstance(obj, dict):
        return {k: _interpolate_recursive(v, variables=variables) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate_recursive(item, variables=variables) for item in obj]
    return obj


def load_config(
    config_path: Path,
    cli_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load YAML config, apply env interpolation and CLI overrides.

    Parameters
    ----------
    config_path:
        Path to the YAML config file.
    cli_overrides:
        Flat dict of key=value overrides. Supports dotted keys
        (e.g., ``{"run.concurrency": 5}``).
    """
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}")

    config = _interpolate_recursive(raw)

    if cli_overrides:
        for dotted_key, value in cli_overrides.items():
            parts = dotted_key.split(".")
            target = config
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value

    return config
