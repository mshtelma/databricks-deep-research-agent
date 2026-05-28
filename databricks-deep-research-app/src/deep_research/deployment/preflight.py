"""Preflight checks for ``make deploy`` — drift + sentinel guards.

These checks BLOCK; they never auto-correct. Auto-correcting the live app
state is exactly the pattern (Apps-API write side outside DAB) that
created the historical lakebase-postgres drift this module exists to
prevent.

Three guards are provided:

* :func:`resolve_postgres_vars_or_fail` — wraps the SDK resolver in
  :mod:`deep_research.deployment.app_postgres_resource` with a clear
  remediation message that points at ``make bootstrap-postgres``.
* :func:`assert_no_pending_sentinels` — refuses to deploy if the DAB
  ``postgres_branch`` or ``postgres_database_resource`` variables are
  still set to their ``"pending"`` defaults.
* :func:`assert_no_drift` — compares the live app's resource list
  against the resource list rendered by DAB. Drift means an out-of-band
  writer (a manual Apps-API mutation, a previous deploy with the
  retired ``configure_app_postgres_resource()`` write side) has changed
  the live app. The deploy must be aborted so terraform's destructive
  reconciliation never runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.apps import App, AppResource

from deep_research.deployment.app_postgres_resource import (
    BUNDLE_VAR_ORDER,
    resolve_postgres_bundle_vars,
)

logger = logging.getLogger(__name__)

PENDING_SENTINEL = "pending"

# Bundle-var keys for the SQL Warehouse used by the framework's text-table
# tools. Resolved by :func:`resolve_warehouse_id_or_fail` and asserted
# non-pending by :func:`assert_no_pending_sentinels` (the same guard as the
# postgres binding vars).
WAREHOUSE_VAR_ORDER: tuple[str, ...] = ("storage_warehouse_id",)


class PreflightError(RuntimeError):
    """A preflight gate refused to let the deploy proceed."""


@dataclass(frozen=True)
class ResourceIdentity:
    """The minimal identity tuple used to compare app resources."""

    name: str
    kind: str
    target: str

    @classmethod
    def from_dab_resource(cls, raw: Mapping[str, Any]) -> ResourceIdentity:
        """Build identity from a ``bundle.tf.json`` resource dict.

        DAB renders each app resource as ``{"name": ..., "secret": {...}}`` or
        ``{"name": ..., "serving_endpoint": {...}}`` etc.; exactly one of the
        non-``name`` keys is populated and identifies the kind. The inner
        block's most-stable identifier (``name``/``scope+key``/``branch+database``)
        is the target.
        """
        name = str(raw.get("name", ""))
        for kind in (
            "secret",
            "serving_endpoint",
            "postgres",
            "sql_warehouse",
            "uc_securable",
            "genie_space",
            "job",
            "experiment",
            "app",
            "database",
        ):
            inner = raw.get(kind)
            if inner:
                return cls(
                    name=name,
                    kind=kind,
                    target=_resource_target(kind, inner),
                )
        # Fall back to "name" alone if no kind block is recognized — this is a
        # diagnostic for unknown resource types; identity uniqueness still
        # depends on the name.
        return cls(name=name, kind="unknown", target="")

    @classmethod
    def from_sdk_resource(cls, resource: AppResource) -> ResourceIdentity:
        """Build identity from the live ``client.apps.get(...).resources`` list."""
        name = resource.name or ""
        for kind in (
            "secret",
            "serving_endpoint",
            "postgres",
            "sql_warehouse",
            "uc_securable",
            "genie_space",
            "job",
            "experiment",
            "app",
            "database",
        ):
            inner = getattr(resource, kind, None)
            if inner is not None:
                return cls(
                    name=name,
                    kind=kind,
                    target=_resource_target(kind, inner),
                )
        return cls(name=name, kind="unknown", target="")


def _resource_target(kind: str, block: Any) -> str:
    """Return a stable identifier string for a resource's target block.

    Accepts either a dict (DAB) or an SDK dataclass (Apps API). Falls back to
    ``repr`` for unknown shapes.
    """
    def _get(key: str) -> str:
        if isinstance(block, Mapping):
            return str(block.get(key, ""))
        value = getattr(block, key, "")
        return str(value if value is not None else "")

    if kind == "secret":
        return f"{_get('scope')}/{_get('key')}"
    if kind == "serving_endpoint":
        return _get("name")
    if kind == "postgres":
        return f"{_get('branch')}::{_get('database')}"
    if kind == "sql_warehouse":
        return _get("id") or _get("name")
    if kind == "uc_securable":
        return _get("securable_full_name") or _get("name")
    if kind == "genie_space":
        return _get("space_id")
    if kind == "job":
        return _get("id")
    if kind == "experiment":
        return _get("path") or _get("id")
    if kind == "database":
        return _get("name") or _get("database_name")
    return repr(block)


@dataclass
class DriftReport:
    """Difference between live app resources and DAB-desired resources."""

    extra_in_live: list[ResourceIdentity] = field(default_factory=list)
    missing_from_live: list[ResourceIdentity] = field(default_factory=list)

    @property
    def has_drift(self) -> bool:
        return bool(self.extra_in_live or self.missing_from_live)

    def format(self) -> str:
        lines: list[str] = []
        if self.extra_in_live:
            lines.append("Live app has resources NOT in DAB desired state:")
            for ident in self.extra_in_live:
                lines.append(f"  + {ident.name} ({ident.kind}) -> {ident.target}")
        if self.missing_from_live:
            lines.append("DAB desired state has resources NOT in live app:")
            for ident in self.missing_from_live:
                lines.append(f"  - {ident.name} ({ident.kind}) -> {ident.target}")
        return "\n".join(lines)


def resolve_postgres_vars_or_fail(
    *,
    profile: str,
    endpoint_name: str,
    database_name: str,
    target: str = "",
) -> dict[str, str]:
    """Resolve ``postgres_branch`` and ``postgres_database_resource``.

    Raises :class:`PreflightError` with an explicit ``make bootstrap-postgres``
    hint if the project / branch / database cannot be discovered.
    """
    try:
        client = WorkspaceClient(profile=profile)
        return resolve_postgres_bundle_vars(
            client=client,
            endpoint_name=endpoint_name,
            database_name=database_name,
        )
    except Exception as exc:  # noqa: BLE001 — surface details in the message
        hint_target = target or "<target>"
        raise PreflightError(
            "Could not resolve DAB postgres binding vars "
            f"(endpoint={endpoint_name!r}, database={database_name!r}): "
            f"{exc!r}. If this is a brand-new target, run "
            f"`make bootstrap-postgres TARGET={hint_target}` first to "
            "provision the Postgres project + database."
        ) from exc


def assert_no_pending_sentinels(bundle_vars: Mapping[str, str]) -> None:
    """Raise if any required bundle var is still the ``pending`` sentinel.

    ``bundle_vars`` is the resolved dict to be passed to
    ``databricks bundle deploy``. Required keys come from
    :data:`BUNDLE_VAR_ORDER` (postgres binding) and
    :data:`WAREHOUSE_VAR_ORDER` (text-table SQL warehouse).
    """
    leaked_postgres = [
        key
        for key in BUNDLE_VAR_ORDER
        if bundle_vars.get(key, PENDING_SENTINEL) == PENDING_SENTINEL
    ]
    leaked_warehouse = [
        key
        for key in WAREHOUSE_VAR_ORDER
        if bundle_vars.get(key, PENDING_SENTINEL) == PENDING_SENTINEL
    ]
    if leaked_postgres:
        raise PreflightError(
            "Refusing to deploy: the following DAB variables are still "
            f"set to the 'pending' sentinel: {leaked_postgres!r}. The "
            "lakebase-postgres app resource would be deployed with an invalid "
            "binding. Run `make bootstrap-postgres TARGET=<name>` to resolve "
            "real values."
        )
    if leaked_warehouse:
        raise PreflightError(
            "Refusing to deploy: the following DAB variables are still "
            f"set to the 'pending' sentinel: {leaked_warehouse!r}. The "
            "deployed app would silently boot without text-table SQL "
            "wiring (workflow_runner_factory leaves schema_cache and "
            "sql_executor unset), and any workflow that declares a table_* "
            "tool would fail mid-stream as a misleading 'missing declared "
            "tools' WorkflowError. Pin the workspace's SQL Warehouse via "
            "STORAGE_WAREHOUSE_ID, the bundle var "
            "`storage_warehouse_id`, or expose exactly one STARTED "
            "warehouse for auto-discovery."
        )


def resolve_warehouse_id_or_fail(
    *,
    profile: str,
    target: str = "",
) -> dict[str, str]:
    """Resolve ``storage_warehouse_id`` for the deploy.

    Resolution order (each step short-circuits on success):

    1. Explicit env var ``STORAGE_WAREHOUSE_ID`` (CI / per-developer override).
    2. Explicit env var ``TABLE_TOOLS_WAREHOUSE_ID`` (legacy alias used by
       ``workflow_runner_factory._resolve_table_warehouse_id``).
    3. ``WorkspaceClient(profile=profile).warehouses.list()`` — pick the
       single ``STARTED`` warehouse, else raise with the candidate list so
       the operator can pin one in the bundle.

    Returns a single-key dict ``{"storage_warehouse_id": <id>}`` shaped to
    feed straight into ``assert_no_pending_sentinels``.
    """
    env_id = os.environ.get("STORAGE_WAREHOUSE_ID") or os.environ.get(
        "TABLE_TOOLS_WAREHOUSE_ID"
    )
    if env_id:
        return {"storage_warehouse_id": env_id}

    try:
        client = WorkspaceClient(profile=profile)
        warehouses = list(client.warehouses.list())
    except Exception as exc:  # noqa: BLE001 - surface details in the message
        hint_target = target or "<target>"
        raise PreflightError(
            "Could not enumerate SQL Warehouses to resolve "
            f"storage_warehouse_id ({exc!r}). Either set "
            "STORAGE_WAREHOUSE_ID in your environment, pin "
            "`storage_warehouse_id` per target in databricks.yml, or "
            f"verify `--profile {profile!r}` has warehouse list access "
            f"(target={hint_target})."
        ) from exc

    started = [w for w in warehouses if _warehouse_is_started(w)]
    if len(started) == 1:
        wid = _warehouse_id(started[0])
        if not wid:
            raise PreflightError(
                "Single STARTED warehouse has no id attribute — refusing to "
                "use it. Set STORAGE_WAREHOUSE_ID explicitly."
            )
        return {"storage_warehouse_id": wid}

    if len(started) == 0:
        candidates = ", ".join(
            f"{_warehouse_name(w) or '<unnamed>'} ({_warehouse_id(w) or '<no-id>'})"
            for w in warehouses
        ) or "<none>"
        raise PreflightError(
            "No STARTED SQL Warehouse found in the workspace. Start one or "
            "set STORAGE_WAREHOUSE_ID explicitly. "
            f"All warehouses: [{candidates}]"
        )

    candidates = ", ".join(
        f"{_warehouse_name(w) or '<unnamed>'} ({_warehouse_id(w) or '<no-id>'})"
        for w in started
    )
    raise PreflightError(
        f"{len(started)} STARTED SQL Warehouses found — refusing to "
        "auto-pick. Set STORAGE_WAREHOUSE_ID explicitly or pin "
        "`storage_warehouse_id` per target in databricks.yml. Started "
        f"warehouses: [{candidates}]"
    )


def _warehouse_id(warehouse: Any) -> str:
    return str(getattr(warehouse, "id", "") or "")


def _warehouse_name(warehouse: Any) -> str:
    return str(getattr(warehouse, "name", "") or "")


def _warehouse_is_started(warehouse: Any) -> bool:
    state = getattr(warehouse, "state", None)
    if state is None:
        return False
    name = getattr(state, "value", None) or getattr(state, "name", None) or state
    return str(name).upper() == "RUNNING" or str(name).upper() == "STARTED"


def _load_dab_app_resources(bundle_tf_json: Path) -> list[ResourceIdentity]:
    """Read the DAB-generated ``bundle.tf.json`` and extract app resources."""
    if not bundle_tf_json.exists():
        raise PreflightError(
            f"DAB-generated bundle file not found: {bundle_tf_json}. "
            "Run `databricks bundle deploy -t <target> --plan` (or any "
            "deploy command) so DAB renders the desired state."
        )
    payload = json.loads(bundle_tf_json.read_text())
    try:
        app = payload["resource"]["databricks_app"]["deep_research_agent"]
        raw_resources = app.get("resources", [])
    except KeyError as exc:
        raise PreflightError(
            f"{bundle_tf_json} did not contain "
            "resource.databricks_app.deep_research_agent — bundle layout "
            "may have changed."
        ) from exc
    return [ResourceIdentity.from_dab_resource(r) for r in raw_resources]


def _load_live_app_resources(
    *, profile: str, app_name: str
) -> list[ResourceIdentity]:
    """Fetch the live app's resource list via the Apps API."""
    client = WorkspaceClient(profile=profile)
    app: App = client.apps.get(app_name)
    return [ResourceIdentity.from_sdk_resource(r) for r in (app.resources or [])]


def diff_resources(
    *,
    desired: Iterable[ResourceIdentity],
    live: Iterable[ResourceIdentity],
) -> DriftReport:
    """Compute a :class:`DriftReport` between desired and live identities."""
    desired_set = set(desired)
    live_set = set(live)
    return DriftReport(
        extra_in_live=sorted(live_set - desired_set, key=lambda i: (i.kind, i.name)),
        missing_from_live=sorted(
            desired_set - live_set, key=lambda i: (i.kind, i.name)
        ),
    )


def assert_no_drift(
    *,
    profile: str,
    app_name: str,
    bundle_tf_json: Path,
) -> None:
    """Block if live app has resources not in DAB desired state (or vice versa).

    Use BEFORE ``databricks bundle deploy`` so terraform never reaches an
    apply phase where it would try to reconcile (i.e. delete) an
    out-of-band-added resource.
    """
    desired = _load_dab_app_resources(bundle_tf_json)
    live = _load_live_app_resources(profile=profile, app_name=app_name)
    report = diff_resources(desired=desired, live=live)
    if report.has_drift:
        raise PreflightError(
            "Live app drifted from DAB desired state:\n"
            + report.format()
            + "\n\nResolve by aligning databricks.yml with the live app, "
            "or, if the live mutation was unintended, remove the offending "
            "resource(s) via the Apps API before re-running deploy. Do NOT "
            "let terraform auto-correct: removing an actively-bound Postgres "
            "resource currently returns an opaque 'failed to update app' "
            "from the provider."
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight guards for `make deploy`: resolve postgres binding "
            "vars, assert no 'pending' sentinels, and (optionally) assert "
            "no drift between live app and DAB desired state."
        )
    )
    parser.add_argument(
        "--profile",
        default=os.environ.get("DATABRICKS_CONFIG_PROFILE"),
        required=os.environ.get("DATABRICKS_CONFIG_PROFILE") is None,
    )
    parser.add_argument("--target", required=True, help="DAB target name")
    parser.add_argument(
        "--endpoint-name",
        default=os.environ.get("ENDPOINT_NAME"),
        required=os.environ.get("ENDPOINT_NAME") is None,
    )
    parser.add_argument(
        "--database-name",
        default=os.environ.get("LAKEBASE_DATABASE", "deep_research"),
    )
    parser.add_argument(
        "--app-name",
        default=None,
        help=(
            "Live Databricks App name. Required to run the drift check. "
            "When omitted, only the var resolution + sentinel guard runs."
        ),
    )
    parser.add_argument(
        "--bundle-tf-json",
        type=Path,
        default=None,
        help=(
            "Path to DAB-generated bundle.tf.json. Required to run the "
            "drift check (e.g. "
            ".databricks/bundle/<target>/terraform/bundle.tf.json)."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Exits 1 with a one-line message on PreflightError."""
    args = _build_arg_parser().parse_args(argv)
    try:
        bundle_vars = resolve_postgres_vars_or_fail(
            profile=args.profile,
            endpoint_name=args.endpoint_name,
            database_name=args.database_name,
            target=args.target,
        )
        warehouse_vars = resolve_warehouse_id_or_fail(
            profile=args.profile,
            target=args.target,
        )
        bundle_vars = {**bundle_vars, **warehouse_vars}
        assert_no_pending_sentinels(bundle_vars)
        if args.app_name and args.bundle_tf_json:
            assert_no_drift(
                profile=args.profile,
                app_name=args.app_name,
                bundle_tf_json=args.bundle_tf_json,
            )
        elif args.app_name or args.bundle_tf_json:
            print(
                "INFO: drift check skipped — both --app-name and "
                "--bundle-tf-json are required to run it.",
                file=sys.stderr,
            )
        # Echo the resolved vars so the Makefile can capture them.
        print(
            " ".join(
                f"--var {k}={bundle_vars[k]}"
                for k in (*BUNDLE_VAR_ORDER, *WAREHOUSE_VAR_ORDER)
            )
        )
        return 0
    except PreflightError as exc:
        print(f"PREFLIGHT FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
