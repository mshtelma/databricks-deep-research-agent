"""Registered-tool catalog build + Designer-save validation."""

from __future__ import annotations

from typing import Any

import pytest

from deep_research.agent.tools import registered_catalog
from deep_research.agent_designer.semantic_validation import (
    semantic_validation_errors,
)


def catalog_probe(x: int) -> int:
    """Import target for catalog tests."""
    return x + 1


@pytest.fixture(autouse=True)
def _reset_catalog() -> Any:
    registered_catalog.reset_registered_tool_catalog()
    yield
    registered_catalog.reset_registered_tool_catalog()


def _patch_entries(monkeypatch: pytest.MonkeyPatch, entries: list[str]) -> None:
    class _Tools:
        registered_tools = entries

    class _Cfg:
        tools = _Tools()

    monkeypatch.setattr(registered_catalog, "get_app_config", lambda: _Cfg())


class TestCatalogBuild:
    def test_builds_from_config_entries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_entries(
            monkeypatch,
            ["tests.unit.agent_designer.test_registered_tools:catalog_probe"],
        )
        catalog = registered_catalog.get_registered_tool_catalog()
        key = "tests.unit.agent_designer.test_registered_tools.catalog_probe"
        assert key in catalog
        assert catalog[key].definition.name == "catalog_probe"

    def test_bad_entries_are_skipped_not_fatal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_entries(monkeypatch, ["not-a-target", "ghost.module:fn"])
        assert dict(registered_catalog.get_registered_tool_catalog()) == {}


class TestRegisteredValidation:
    def _definition(self, key: str) -> dict[str, Any]:
        return {
            "tools": [{"name": "fc", "kind": "registered", "config": {"key": key}}],
            "root": {"id": "s", "type": "sequence", "label": "s", "children": []},
        }

    def test_unknown_key_rejected_at_save(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_entries(monkeypatch, [])
        errors = semantic_validation_errors(self._definition("ghost.key"))
        assert any("not in the registered catalog" in e.message for e in errors)

    def test_known_key_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_entries(
            monkeypatch,
            ["tests.unit.agent_designer.test_registered_tools:catalog_probe"],
        )
        key = "tests.unit.agent_designer.test_registered_tools.catalog_probe"
        errors = semantic_validation_errors(self._definition(key))
        assert not [e for e in errors if "registered catalog" in e.message]
