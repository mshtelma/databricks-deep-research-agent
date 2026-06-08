"""Plan ``imperative-wishing-lynx.md`` — shell-app wheel bundling.

Verifies that ``ShellAppExporter`` now ships the framework as a local wheel
bundled inside the generated zip, and removes the runtime dependency on
``pip install git+https://...``.

The primary resolver reads from
``deep_research/services/deployment/_framework_wheel/`` (hatch
``force-include`` package data). The dev fallback walks up to
``databricks-deep-research-app/wheels/``. Both paths are exercised here
against a synthetic wheel file under ``tmp_path`` so the assertions don't
depend on ``make build-framework`` having been run first.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from deep_research.services.deployment import shell_app
from deep_research.services.deployment.shell_app import (
    ShellAppExporter,
    ShellAppWheelMissingError,
    _parse_framework_wheel_version,
    _resolve_framework_wheel,
)

_VALID_CONFIG: dict[str, Any] = {
    "mode": "shell_app",
    "app_name": "dr-shell-wheel",
    "target": "dev",
}


def _agent_revision() -> tuple[MagicMock, MagicMock]:
    agent = MagicMock(id=uuid4(), name="WheelBundleAgent")
    revision = MagicMock(
        rev_id=uuid4(),
        definition={
            "name": "wheel-bundle",
            "version": 1,
            "tools": [],
            "root": {"type": "sequence", "children": []},
        },
    )
    return agent, revision


class TestResolveFrameworkWheel:
    def test_resolves_from_primary_package_data_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The primary location is ``_framework_wheel/`` next to shell_app.py.

        We construct a fake module file under tmp_path with a sibling
        ``_framework_wheel`` directory and patch __file__ so the resolver
        walks the synthetic tree instead of the real source tree.
        """
        fake_module = tmp_path / "shell_app.py"
        fake_module.write_text("# placeholder")
        wheel_dir = tmp_path / "_framework_wheel"
        wheel_dir.mkdir()
        wheel = wheel_dir / "databricks_deep_research-9.9.9-py3-none-any.whl"
        wheel_payload = b"PK\x03\x04 fake wheel content"
        wheel.write_bytes(wheel_payload)

        monkeypatch.setattr(shell_app, "__file__", str(fake_module))

        name, payload = _resolve_framework_wheel()
        assert name == "databricks_deep_research-9.9.9-py3-none-any.whl"
        assert payload == wheel_payload

    def test_raises_when_multiple_wheels_in_package_data(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_module = tmp_path / "shell_app.py"
        fake_module.write_text("# placeholder")
        wheel_dir = tmp_path / "_framework_wheel"
        wheel_dir.mkdir()
        (wheel_dir / "databricks_deep_research-1.0.0-py3-none-any.whl").write_bytes(b"a")
        (wheel_dir / "databricks_deep_research-2.0.0-py3-none-any.whl").write_bytes(b"b")

        monkeypatch.setattr(shell_app, "__file__", str(fake_module))

        with pytest.raises(ShellAppWheelMissingError) as exc_info:
            _resolve_framework_wheel()
        assert "exactly one" in str(exc_info.value)

    def test_raises_when_no_wheel_anywhere(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty _framework_wheel/ + no parent ``wheels/`` → clear error."""
        fake_module = tmp_path / "shell_app.py"
        fake_module.write_text("# placeholder")
        (tmp_path / "_framework_wheel").mkdir()  # exists but empty

        monkeypatch.setattr(shell_app, "__file__", str(fake_module))

        with pytest.raises(ShellAppWheelMissingError) as exc_info:
            _resolve_framework_wheel()
        assert "make build-framework" in str(exc_info.value)


class TestWheelVersionParser:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("databricks_deep_research-0.2.0-py3-none-any.whl", "0.2.0"),
            ("databricks_deep_research-1.5.0rc1-py3-none-any.whl", "1.5.0rc1"),
            ("not-a-framework-wheel.whl", "unknown"),
            ("databricks_deep_research_app-1.0.0-py3-none-any.whl", "unknown"),
        ],
    )
    def test_parse_version(self, filename: str, expected: str) -> None:
        assert _parse_framework_wheel_version(filename) == expected


class TestShellAppZipBundlesWheel:
    """End-to-end translate() assertions. Relies on the real
    _framework_wheel/ directory being populated (true in dev + CI after
    `make build-framework`)."""

    @pytest.mark.asyncio
    async def test_zip_contains_framework_wheel_under_wheels_dir(self) -> None:
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        artifact = await translator.translate(agent, revision, _VALID_CONFIG)
        with zipfile.ZipFile(io.BytesIO(artifact.payload)) as zf:
            wheel_entries = [
                n for n in zf.namelist()
                if n.startswith("wheels/databricks_deep_research-")
                and n.endswith(".whl")
            ]
        assert len(wheel_entries) == 1, (
            f"expected exactly one framework wheel bundled at wheels/; "
            f"got {wheel_entries}"
        )

    @pytest.mark.asyncio
    async def test_zip_wheel_bytes_match_source(self) -> None:
        """The wheel bytes inside the zip must be a byte-for-byte copy of
        the source file — guarantees no truncation / encoding mishap."""
        source_name, source_bytes = _resolve_framework_wheel()
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        artifact = await translator.translate(agent, revision, _VALID_CONFIG)
        with zipfile.ZipFile(io.BytesIO(artifact.payload)) as zf:
            zip_wheel_bytes = zf.read(f"wheels/{source_name}")
        assert zip_wheel_bytes == source_bytes

    @pytest.mark.asyncio
    async def test_pyproject_path_source_matches_bundled_wheel(self) -> None:
        """pyproject.toml's [tool.uv.sources] path must match the actual
        filename of the bundled wheel — otherwise `uv run` won't find it."""
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        artifact = await translator.translate(agent, revision, _VALID_CONFIG)
        with zipfile.ZipFile(io.BytesIO(artifact.payload)) as zf:
            pyproject = zf.read("pyproject.toml").decode("utf-8")
            wheel_entries = [
                n for n in zf.namelist()
                if n.startswith("wheels/databricks_deep_research-")
            ]
        assert len(wheel_entries) == 1
        wheel_relpath = wheel_entries[0]  # "wheels/databricks_deep_research-X.Y.Z-..."
        assert f'path = "{wheel_relpath}"' in pyproject

    @pytest.mark.asyncio
    async def test_pyproject_has_no_git_url(self) -> None:
        """Defense in depth: the deployed shell app must NEVER pip-install
        from GitHub. If a git URL appears, the wheel-bundling refactor has
        regressed and the deployed app's startup will fall back to network
        fetch."""
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        artifact = await translator.translate(agent, revision, _VALID_CONFIG)
        with zipfile.ZipFile(io.BytesIO(artifact.payload)) as zf:
            pyproject = zf.read("pyproject.toml").decode("utf-8")
        assert "git+https://" not in pyproject
        assert "github.com" not in pyproject
        assert "@main" not in pyproject

    @pytest.mark.asyncio
    async def test_translate_remains_byte_deterministic_with_wheel(self) -> None:
        """The wheel entry uses the same fixed 1980-01-01 timestamp as the
        other zip entries — regeneration from the same inputs must remain
        byte-identical (W7 invariant)."""
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        first = await translator.translate(agent, revision, _VALID_CONFIG)
        second = await translator.translate(agent, revision, _VALID_CONFIG)
        assert first.payload == second.payload
        assert first.metadata["sha256"] == second.metadata["sha256"]

    @pytest.mark.asyncio
    async def test_legacy_framework_git_tag_is_ignored(self) -> None:
        """Supplying ``framework_git_tag`` is allowed for backwards
        compatibility but must NOT change the rendered pyproject (the value
        is logged + ignored)."""
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        with_tag = await translator.translate(
            agent, revision, {**_VALID_CONFIG, "framework_git_tag": "v9.9.9"}
        )
        without_tag = await translator.translate(agent, revision, _VALID_CONFIG)
        assert with_tag.payload == without_tag.payload
