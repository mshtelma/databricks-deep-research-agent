"""Tests for the deploy preflight: sentinel guard + drift detection."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from deep_research.deployment.preflight import (
    BUNDLE_VAR_ORDER,
    PENDING_SENTINEL,
    WAREHOUSE_VAR_ORDER,
    PreflightError,
    ResourceIdentity,
    _load_dab_app_resources,
    assert_no_drift,
    assert_no_pending_sentinels,
    diff_resources,
)

# ---- assert_no_pending_sentinels --------------------------------------------


def test_sentinel_guard_blocks_pending_branch() -> None:
    with pytest.raises(PreflightError, match="postgres_branch"):
        assert_no_pending_sentinels(
            {
                "postgres_branch": PENDING_SENTINEL,
                "postgres_database_resource": "projects/x/branches/y/databases/z",
            }
        )


def test_sentinel_guard_blocks_pending_database() -> None:
    with pytest.raises(PreflightError, match="postgres_database_resource"):
        assert_no_pending_sentinels(
            {
                "postgres_branch": "projects/x/branches/y",
                "postgres_database_resource": PENDING_SENTINEL,
            }
        )


def test_sentinel_guard_blocks_missing_key() -> None:
    # A missing var is just as bad as one set to "pending"; defaults to
    # sentinel in the guard.
    with pytest.raises(PreflightError, match="postgres_database_resource"):
        assert_no_pending_sentinels({"postgres_branch": "projects/x/branches/y"})


def test_sentinel_guard_passes_when_all_resolved() -> None:
    assert_no_pending_sentinels(
        {
            "postgres_branch": "projects/deep-research-ais/branches/production",
            "postgres_database_resource": (
                "projects/deep-research-ais/branches/production/databases/db-1"
            ),
            "storage_warehouse_id": "abc1234567890def",
        }
    )


def test_sentinel_guard_blocks_pending_storage_warehouse_id() -> None:
    # Layer 1 of the layered tool-context validation: the warehouse id
    # leaking through as 'pending' would deploy an app whose
    # workflow_runner_factory leaves schema_cache + sql_executor unset,
    # causing a misleading mid-stream "missing declared tools" error for
    # any workflow that declares table_*. The guard must catch this at
    # deploy time alongside the postgres bindings.
    with pytest.raises(PreflightError, match="storage_warehouse_id"):
        assert_no_pending_sentinels(
            {
                "postgres_branch": "projects/x/branches/y",
                "postgres_database_resource": "projects/x/branches/y/databases/z",
                "storage_warehouse_id": PENDING_SENTINEL,
            }
        )


def test_sentinel_guard_blocks_missing_storage_warehouse_id() -> None:
    # Missing key is treated identically to the 'pending' sentinel — both
    # would leak past the guard otherwise.
    with pytest.raises(PreflightError, match="storage_warehouse_id"):
        assert_no_pending_sentinels(
            {
                "postgres_branch": "projects/x/branches/y",
                "postgres_database_resource": "projects/x/branches/y/databases/z",
            }
        )


def test_sentinel_guard_covers_all_bundle_vars() -> None:
    # If BUNDLE_VAR_ORDER or WAREHOUSE_VAR_ORDER grow, this test fails to
    # remind us to update the sentinel guard's expectations.
    assert set(BUNDLE_VAR_ORDER) == {
        "postgres_branch",
        "postgres_database_resource",
    }
    assert set(WAREHOUSE_VAR_ORDER) == {"storage_warehouse_id"}


# ---- diff_resources / DriftReport -------------------------------------------


def _ident(name: str, kind: str = "secret", target: str = "scope/key") -> ResourceIdentity:
    return ResourceIdentity(name=name, kind=kind, target=target)


def test_diff_clean_when_identities_match() -> None:
    desired = [_ident("a"), _ident("b", kind="postgres", target="branchX::dbY")]
    live = [_ident("a"), _ident("b", kind="postgres", target="branchX::dbY")]

    report = diff_resources(desired=desired, live=live)

    assert not report.has_drift
    assert report.extra_in_live == []
    assert report.missing_from_live == []


def test_diff_reports_extra_in_live() -> None:
    desired = [_ident("a")]
    live = [
        _ident("a"),
        _ident("lakebase-postgres", kind="postgres", target="branchX::dbY"),
    ]

    report = diff_resources(desired=desired, live=live)

    assert report.has_drift
    assert report.extra_in_live == [
        _ident("lakebase-postgres", kind="postgres", target="branchX::dbY")
    ]
    assert report.missing_from_live == []
    msg = report.format()
    assert "lakebase-postgres" in msg
    assert "NOT in DAB" in msg


def test_diff_reports_missing_from_live() -> None:
    desired = [
        _ident("a"),
        _ident("lakebase-postgres", kind="postgres", target="branchX::dbY"),
    ]
    live = [_ident("a")]

    report = diff_resources(desired=desired, live=live)

    assert report.has_drift
    assert report.missing_from_live == [
        _ident("lakebase-postgres", kind="postgres", target="branchX::dbY")
    ]


def test_diff_distinguishes_target_change() -> None:
    # Same name + kind but different target (e.g. branch path moved) is drift.
    desired = [_ident("lakebase-postgres", kind="postgres", target="branchA::dbX")]
    live = [_ident("lakebase-postgres", kind="postgres", target="branchB::dbX")]

    report = diff_resources(desired=desired, live=live)

    assert report.has_drift
    assert len(report.extra_in_live) == 1
    assert len(report.missing_from_live) == 1


# ---- ResourceIdentity parsing -----------------------------------------------


def test_resource_identity_from_dab_secret() -> None:
    raw = {
        "name": "brave-api-key",
        "secret": {"scope": "deep-research-secrets", "key": "BRAVE_API_KEY"},
    }

    ident = ResourceIdentity.from_dab_resource(raw)

    assert ident == ResourceIdentity(
        name="brave-api-key", kind="secret", target="deep-research-secrets/BRAVE_API_KEY"
    )


def test_resource_identity_from_dab_postgres() -> None:
    raw = {
        "name": "lakebase-postgres",
        "postgres": {
            "branch": "projects/p/branches/b",
            "database": "projects/p/branches/b/databases/db-1",
            "permission": "CAN_CONNECT_AND_CREATE",
        },
    }

    ident = ResourceIdentity.from_dab_resource(raw)

    assert ident.name == "lakebase-postgres"
    assert ident.kind == "postgres"
    assert "branches/b" in ident.target
    assert "databases/db-1" in ident.target


def test_resource_identity_from_sdk_postgres() -> None:
    resource = SimpleNamespace(
        name="lakebase-postgres",
        postgres=SimpleNamespace(
            branch="projects/p/branches/b",
            database="projects/p/branches/b/databases/db-1",
            permission="CAN_CONNECT_AND_CREATE",
        ),
        secret=None,
        serving_endpoint=None,
        sql_warehouse=None,
        uc_securable=None,
        genie_space=None,
        job=None,
        experiment=None,
        app=None,
        database=None,
    )

    ident = ResourceIdentity.from_sdk_resource(resource)

    assert ident.name == "lakebase-postgres"
    assert ident.kind == "postgres"
    assert "branches/b" in ident.target


def test_resource_identity_dab_and_sdk_match_for_postgres() -> None:
    # The drift check round-trips DAB JSON ↔ SDK objects; identity must
    # be identical on both sides.
    raw = {
        "name": "lakebase-postgres",
        "postgres": {
            "branch": "projects/p/branches/b",
            "database": "projects/p/branches/b/databases/db-1",
            "permission": "CAN_CONNECT_AND_CREATE",
        },
    }
    sdk = SimpleNamespace(
        name=raw["name"],
        postgres=SimpleNamespace(**raw["postgres"]),
        secret=None,
        serving_endpoint=None,
        sql_warehouse=None,
        uc_securable=None,
        genie_space=None,
        job=None,
        experiment=None,
        app=None,
        database=None,
    )

    assert ResourceIdentity.from_dab_resource(raw) == ResourceIdentity.from_sdk_resource(sdk)


# ---- _load_dab_app_resources / assert_no_drift -----------------------------


def _write_bundle_tf_json(path: Path, resources: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "resource": {
                    "databricks_app": {
                        "deep_research_agent": {
                            "name": "deep-research-agent-ais",
                            "resources": resources,
                        }
                    }
                }
            }
        )
    )


def test_load_dab_app_resources_raises_when_file_missing(tmp_path: Path) -> None:
    with pytest.raises(PreflightError, match="not found"):
        _load_dab_app_resources(tmp_path / "nope.json")


def test_load_dab_app_resources_raises_when_layout_wrong(tmp_path: Path) -> None:
    path = tmp_path / "bundle.tf.json"
    path.write_text(json.dumps({"resource": {}}))

    with pytest.raises(PreflightError, match="deep_research_agent"):
        _load_dab_app_resources(path)


def test_load_dab_app_resources_returns_identities(tmp_path: Path) -> None:
    path = tmp_path / "bundle.tf.json"
    _write_bundle_tf_json(
        path,
        [
            {
                "name": "brave-api-key",
                "secret": {"scope": "s", "key": "k"},
            },
            {
                "name": "lakebase-postgres",
                "postgres": {
                    "branch": "projects/p/branches/b",
                    "database": "projects/p/branches/b/databases/db-1",
                    "permission": "CAN_CONNECT_AND_CREATE",
                },
            },
        ],
    )

    idents = _load_dab_app_resources(path)

    assert {i.name for i in idents} == {"brave-api-key", "lakebase-postgres"}


def test_assert_no_drift_raises_when_live_has_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "bundle.tf.json"
    _write_bundle_tf_json(
        path,
        [{"name": "brave-api-key", "secret": {"scope": "s", "key": "k"}}],
    )

    fake_app = SimpleNamespace(
        resources=[
            SimpleNamespace(
                name="brave-api-key",
                secret=SimpleNamespace(scope="s", key="k", permission="READ"),
                serving_endpoint=None,
                postgres=None,
                sql_warehouse=None,
                uc_securable=None,
                genie_space=None,
                job=None,
                experiment=None,
                app=None,
                database=None,
            ),
            SimpleNamespace(
                name="lakebase-postgres",
                secret=None,
                serving_endpoint=None,
                postgres=SimpleNamespace(
                    branch="projects/p/branches/b",
                    database="projects/p/branches/b/databases/db-1",
                    permission="CAN_CONNECT_AND_CREATE",
                ),
                sql_warehouse=None,
                uc_securable=None,
                genie_space=None,
                job=None,
                experiment=None,
                app=None,
                database=None,
            ),
        ]
    )

    class _FakeClient:
        def __init__(self, profile: str) -> None:
            self.apps = SimpleNamespace(get=lambda _: fake_app)

    monkeypatch.setattr(
        "deep_research.deployment.preflight.WorkspaceClient", _FakeClient
    )

    with pytest.raises(PreflightError, match="lakebase-postgres"):
        assert_no_drift(
            profile="ais",
            app_name="deep-research-agent-ais",
            bundle_tf_json=path,
        )


def test_assert_no_drift_passes_when_aligned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "bundle.tf.json"
    _write_bundle_tf_json(
        path,
        [
            {"name": "brave-api-key", "secret": {"scope": "s", "key": "k"}},
            {
                "name": "lakebase-postgres",
                "postgres": {
                    "branch": "projects/p/branches/b",
                    "database": "projects/p/branches/b/databases/db-1",
                    "permission": "CAN_CONNECT_AND_CREATE",
                },
            },
        ],
    )

    fake_app = SimpleNamespace(
        resources=[
            SimpleNamespace(
                name="brave-api-key",
                secret=SimpleNamespace(scope="s", key="k", permission="READ"),
                serving_endpoint=None,
                postgres=None,
                sql_warehouse=None,
                uc_securable=None,
                genie_space=None,
                job=None,
                experiment=None,
                app=None,
                database=None,
            ),
            SimpleNamespace(
                name="lakebase-postgres",
                secret=None,
                serving_endpoint=None,
                postgres=SimpleNamespace(
                    branch="projects/p/branches/b",
                    database="projects/p/branches/b/databases/db-1",
                    permission="CAN_CONNECT_AND_CREATE",
                ),
                sql_warehouse=None,
                uc_securable=None,
                genie_space=None,
                job=None,
                experiment=None,
                app=None,
                database=None,
            ),
        ]
    )

    class _FakeClient:
        def __init__(self, profile: str) -> None:
            self.apps = SimpleNamespace(get=lambda _: fake_app)

    monkeypatch.setattr(
        "deep_research.deployment.preflight.WorkspaceClient", _FakeClient
    )

    assert_no_drift(
        profile="ais",
        app_name="deep-research-agent-ais",
        bundle_tf_json=path,
    )
