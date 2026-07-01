"""Unit tests for the shell-app runtime-requirements value object + metadata contract.

These lock the single-source-of-truth behavior: ``translate`` decides the runtime
requirements once and serializes them; the deploy path reconstructs them verbatim.
The Brave gate (``uses_brave``) must be impossible to bypass via inconsistent
inputs or stale/old metadata.
"""

from __future__ import annotations

from deep_research.services.deployment.shell_app_runtime import (
    BRAVE_SECRET_RESOURCE_NAME,
    SQL_WAREHOUSE_RESOURCE_NAME,
    ShellAppRuntimeBindings,
)


class TestBuildInvariants:
    def test_brave_scope_key_dropped_when_not_uses_brave(self) -> None:
        req = ShellAppRuntimeBindings.build(
            requires_web_search=True,
            uses_brave=False,
            requires_sql_warehouse=False,
            brave_secret_scope="deep-research-secrets",
            brave_secret_key="BRAVE_API_KEY",
            storage_warehouse_id=None,
        )
        assert req.uses_brave is False
        assert req.brave_secret_scope is None
        assert req.brave_secret_key is None

    def test_brave_scope_key_kept_when_uses_brave(self) -> None:
        req = ShellAppRuntimeBindings.build(
            requires_web_search=True,
            uses_brave=True,
            requires_sql_warehouse=False,
            brave_secret_scope="my-scope",
            brave_secret_key="MY_KEY",
            storage_warehouse_id=None,
        )
        assert req.uses_brave is True
        assert req.brave_secret_scope == "my-scope"
        assert req.brave_secret_key == "MY_KEY"

    def test_warehouse_dropped_when_not_required(self) -> None:
        req = ShellAppRuntimeBindings.build(
            requires_web_search=False,
            uses_brave=False,
            requires_sql_warehouse=False,
            brave_secret_scope=None,
            brave_secret_key=None,
            storage_warehouse_id="wh-123",
        )
        assert req.requires_sql_warehouse is False
        assert req.storage_warehouse_id is None

    def test_blank_identifiers_normalized_to_none(self) -> None:
        req = ShellAppRuntimeBindings.build(
            requires_web_search=True,
            uses_brave=True,
            requires_sql_warehouse=True,
            brave_secret_scope="  ",
            brave_secret_key="",
            storage_warehouse_id="   ",
        )
        assert req.brave_secret_scope is None
        assert req.brave_secret_key is None
        assert req.storage_warehouse_id is None


class TestMetadataRoundTrip:
    def _round_trip(self, req: ShellAppRuntimeBindings) -> ShellAppRuntimeBindings:
        return ShellAppRuntimeBindings.from_metadata(req.to_metadata())

    def test_brave_pinned_round_trip(self) -> None:
        req = ShellAppRuntimeBindings.build(
            requires_web_search=True,
            uses_brave=True,
            requires_sql_warehouse=False,
            brave_secret_scope="deep-research-secrets",
            brave_secret_key="BRAVE_API_KEY",
            storage_warehouse_id=None,
        )
        assert self._round_trip(req) == req

    def test_inherited_web_round_trip(self) -> None:
        req = ShellAppRuntimeBindings.build(
            requires_web_search=True,
            uses_brave=False,
            requires_sql_warehouse=False,
            brave_secret_scope=None,
            brave_secret_key=None,
            storage_warehouse_id=None,
            databricks_web_search_endpoint="databricks-gemini-3-1-flash-lite",
        )
        restored = self._round_trip(req)
        assert restored == req
        assert restored.uses_brave is False
        assert restored.brave_secret_scope is None

    def test_sql_warehouse_round_trip(self) -> None:
        req = ShellAppRuntimeBindings.build(
            requires_web_search=False,
            uses_brave=False,
            requires_sql_warehouse=True,
            brave_secret_scope=None,
            brave_secret_key=None,
            storage_warehouse_id="wh-123",
        )
        assert self._round_trip(req) == req

    def test_to_metadata_emits_legacy_configured_keys(self) -> None:
        req = ShellAppRuntimeBindings.build(
            requires_web_search=True,
            uses_brave=True,
            requires_sql_warehouse=False,
            brave_secret_scope="s",
            brave_secret_key="k",
            storage_warehouse_id=None,
        )
        md = req.to_metadata()
        assert md["uses_brave"] == "true"
        assert md["brave_secret_scope_configured"] == "true"
        assert md["brave_secret_key_configured"] == "true"
        assert md["storage_warehouse_id_configured"] == "false"


class TestFromMetadataSafety:
    def test_empty_metadata_yields_no_brave(self) -> None:
        req = ShellAppRuntimeBindings.from_metadata({})
        assert req.uses_brave is False
        assert req.requires_web_search is False
        assert req.brave_secret_scope is None
        assert req.brave_secret_key is None

    def test_none_metadata_yields_no_brave(self) -> None:
        req = ShellAppRuntimeBindings.from_metadata(None)
        assert req.uses_brave is False
        assert req.brave_secret_scope is None

    def test_legacy_artifact_without_uses_brave_does_not_bind_brave(self) -> None:
        # An artifact produced before ``uses_brave`` existed: web search true, the
        # key absent, and a stray scope/key present. Must still yield no Brave.
        req = ShellAppRuntimeBindings.from_metadata(
            {
                "requires_web_search": "true",
                "brave_secret_scope": "deep-research-secrets",
                "brave_secret_key": "BRAVE_API_KEY",
            }
        )
        assert req.requires_web_search is True
        assert req.uses_brave is False
        assert req.brave_secret_scope is None
        assert req.brave_secret_key is None

    def test_resource_names_default_when_absent(self) -> None:
        req = ShellAppRuntimeBindings.from_metadata({"uses_brave": "false"})
        assert req.brave_secret_resource_name == BRAVE_SECRET_RESOURCE_NAME
        assert req.sql_warehouse_resource_name == SQL_WAREHOUSE_RESOURCE_NAME
