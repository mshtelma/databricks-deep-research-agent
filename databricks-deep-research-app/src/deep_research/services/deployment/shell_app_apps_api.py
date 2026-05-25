"""Core live-deploy helper for the Shell App "Deploy here" feature.

US-402: implements ``_deploy_via_apps_api`` which uploads the rendered
source tree to a user-scoped workspace path and creates/updates a
Databricks App via the SDK Apps API.

All ``databricks.sdk`` symbols are **lazy-imported** inside the function body
so unit tests can ``unittest.mock.patch`` them without requiring the SDK at
module import time.

Top-level imports are restricted to stdlib + translator types only.
"""

from __future__ import annotations

import asyncio
import io
import logging
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deep_research.core.config import get_settings
from deep_research.models.agent_deployment import AgentDeployment
from deep_research.services.deployment.apps_collision import (
    AppOwnershipCheck,
    generate_suggested_name,
    resolve_apps_already_exists,
)
from deep_research.services.deployment.apps_logs import fetch_app_log_tail
from deep_research.services.deployment.github_tag_probe import probe_framework_tag
from deep_research.services.deployment.translator import Artifact, DeploymentResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Size thresholds (R9 in the plan)
# ---------------------------------------------------------------------------
_MAX_SINGLE_FILE_BYTES: int = 500 * 1024 * 1024  # 500 MB
_MAX_TOTAL_BYTES: int = 50 * 1024 * 1024  # 50 MB

# ---------------------------------------------------------------------------
# Reachability probe settings
# ---------------------------------------------------------------------------
_PROBE_POLL_INTERVAL_SEC: float = 5.0
_PROBE_TIMEOUT_SEC: float = 300.0  # 5 minutes
_APP_NAME_PREFIX = "dr-shell-"
_APP_NAME_MAX_LENGTH = 30
_BRAVE_SECRET_RESOURCE_NAME = "brave-api-key"
# OBO scopes granted to deployed shell apps. Must include both vector-search
# scopes — endpoints-only is insufficient because the SDK calls query_index
# against the index path, not the endpoint. Keep in sync with the bundle
# template at templates/agent-shell-app/databricks.yml.j2 and with the main
# DRE bundle at databricks-deep-research-app/databricks.yml.
_APP_USER_API_SCOPES: tuple[str, ...] = (
    "sql",
    "serving.serving-endpoints",
    "vectorsearch.vector-search-endpoints",
    "vectorsearch.vector-search-indexes",
    "dashboards.genie",
)
_APP_RUNNING_STATES: frozenset[str] = frozenset({"RUNNING", "ACTIVE"})
_APP_FAILED_STATES: frozenset[str] = frozenset(
    {
        "ERROR",
        "FAILED",
        "CRASHED",
        "CRASH_LOOP_BACKOFF",
    }
)


@dataclass(frozen=True)
class ReachabilityProbeResult:
    """Outcome of polling a Databricks App after a deploy-here action."""

    reached: bool
    timed_out: bool
    failed: bool = False
    last_state: str | None = None
    last_message: str | None = None


@dataclass(frozen=True)
class ShellAppRuntimeBindings:
    """Databricks Apps runtime bindings required by the generated shell app."""

    requires_web_search: bool
    brave_secret_scope: str | None
    brave_secret_key: str | None
    brave_secret_resource_name: str = _BRAVE_SECRET_RESOURCE_NAME


def _fallback_app_name(deployment_id: str) -> str:
    """Build a Databricks Apps-compatible fallback name from a deployment id."""
    suffix = deployment_id.replace("-", "")[: _APP_NAME_MAX_LENGTH - len(_APP_NAME_PREFIX)]
    return f"{_APP_NAME_PREFIX}{suffix}"


def _resolve_app_name(config: dict[str, Any], deployment_id: str) -> str:
    """Use the requested app name; fall back to a bounded generated name.

    The create-deployment API validates ``config.app_name`` before rows are
    written. This fallback exists for defensive compatibility with older or
    synthetic rows that might be missing the field.
    """
    configured = config.get("app_name")
    if isinstance(configured, str) and configured:
        return configured
    return _fallback_app_name(deployment_id)


def _metadata_bool(metadata: dict[str, Any] | None, key: str) -> bool:
    value = (metadata or {}).get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _resolve_runtime_bindings(
    artifact: Artifact,
    config: dict[str, Any],
    settings: Any,
) -> ShellAppRuntimeBindings:
    """Resolve runtime resource/env bindings from artifact metadata + config."""
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    requires_web_search = _metadata_bool(metadata, "requires_web_search")
    scope = config.get("brave_secret_scope") or settings.deploy_here_brave_secret_scope
    key = config.get("brave_secret_key") or settings.deploy_here_brave_secret_key
    raw_resource_name = metadata.get("brave_secret_resource_name")
    resource_name = (
        raw_resource_name
        if isinstance(raw_resource_name, str)
        else _BRAVE_SECRET_RESOURCE_NAME
    )
    return ShellAppRuntimeBindings(
        requires_web_search=requires_web_search,
        brave_secret_scope=str(scope).strip() if scope else None,
        brave_secret_key=str(key).strip() if key else None,
        brave_secret_resource_name=resource_name,
    )


async def _deploy_via_apps_api(
    artifact: Artifact,
    config: dict[str, Any],
    deployment: AgentDeployment,
    workspace_client: Any,  # WorkspaceClient — kept Any to avoid hard SDK import dep
) -> DeploymentResult:
    """Upload rendered shell-app source tree to workspace and create/update the App.

    Steps:
      1. Validate ``artifact.payload`` is bytes.
      2. Extract zip into a temp directory.
      3. Pre-flight size check (per-file 500 MB, total-tree 50 MB).
      3a. Framework Git-tag preflight (S4a) — probe GitHub before any mutation.
      4. Resolve deployer e-mail via ``workspace_client.current_user.me()``.
      5. Compute workspace upload path.
      6. Upload files (creating parent dirs as needed).
      7. App create-or-update (AlreadyExists → owner-aware resolution via S3).
      8. Reachability probe (poll until RUNNING or timeout → S4b log tail).
      9. Inline rollback on failure after upload started.
    """
    # ------------------------------------------------------------------
    # Step 1 — validate payload type
    # ------------------------------------------------------------------
    if not isinstance(artifact.payload, bytes):
        return DeploymentResult(
            success=False,
            error_message=(
                f"_deploy_via_apps_api expected artifact.payload to be bytes, "
                f"got {type(artifact.payload).__name__}"
            ),
        )

    # ------------------------------------------------------------------
    # Step 2 — extract zip to tempdir
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory(prefix="dr-shell-app-") as workdir_str:
        workdir = Path(workdir_str)
        try:
            with zipfile.ZipFile(io.BytesIO(artifact.payload)) as zf:
                zf.extractall(workdir)
        except zipfile.BadZipFile as exc:
            return DeploymentResult(
                success=False,
                error_message=f"artifact.payload is not a valid zip: {exc}",
            )

        # ------------------------------------------------------------------
        # Step 3 — pre-flight size check
        # ------------------------------------------------------------------
        all_files = [p for p in workdir.rglob("*") if p.is_file()]
        total_bytes = 0
        for file_path in all_files:
            size = file_path.stat().st_size
            if size > _MAX_SINGLE_FILE_BYTES:
                return DeploymentResult(
                    success=False,
                    error_message=(f"File {file_path.name} exceeds 500 MB per-file limit"),
                    external_resource_ids={
                        "error_kind": "artifact_too_large",
                        "limit": "500MB per file",
                        "actual": size,
                    },
                )
            total_bytes += size

        if total_bytes > _MAX_TOTAL_BYTES:
            return DeploymentResult(
                success=False,
                error_message=(f"Total source tree size {total_bytes} bytes exceeds 50 MB limit"),
                external_resource_ids={
                    "error_kind": "artifact_too_large",
                    "limit": "50MB total",
                    "actual": total_bytes,
                },
            )

        # ------------------------------------------------------------------
        # Step 3a — framework Git-ref preflight (S4a)
        # Probe GitHub before any workspace mutation so a missing ref
        # surfaces as a clean error_kind rather than a pip-install failure.
        # ------------------------------------------------------------------
        settings = get_settings()
        runtime_bindings = _resolve_runtime_bindings(artifact, config, settings)
        if (
            runtime_bindings.requires_web_search
            and (
                not runtime_bindings.brave_secret_scope
                or not runtime_bindings.brave_secret_key
            )
        ):
            return DeploymentResult(
                success=False,
                error_message=(
                    "Shell app workflow uses web_search but Brave secret "
                    "scope/key is not configured."
                ),
                external_resource_ids={
                    "error_kind": "validation_failed",
                    "requires_web_search": True,
                },
            )
        logger.info(
            "DEPLOY_HERE_RUNTIME_BINDINGS deployment=%s requires_web_search=%s "
            "env_vars=%s resource_names=%s brave_secret_scope_configured=%s "
            "brave_secret_key_configured=%s",
            deployment.id,
            runtime_bindings.requires_web_search,
            ["MLFLOW_TRACKING_URI"]
            + (["BRAVE_API_KEY"] if runtime_bindings.requires_web_search else []),
            [runtime_bindings.brave_secret_resource_name]
            if runtime_bindings.requires_web_search
            else [],
            bool(runtime_bindings.brave_secret_scope),
            bool(runtime_bindings.brave_secret_key),
        )
        if settings.deploy_here_framework_tag_preflight:
            git_tag: str = config.get("framework_git_tag", "")
            git_url: str = settings.framework_git_url
            tag_probe = await probe_framework_tag(
                git_url=git_url,
                git_tag=git_tag,
                github_token=settings.github_api_token,
            )
            if not tag_probe.reachable:
                logger.info(
                    "DEPLOYMENT_HERE_STAGE deployment=%s stage=tag_preflight outcome=err "
                    "git_tag=%s git_url=%s note=%s",
                    deployment.id,
                    git_tag,
                    git_url,
                    tag_probe.note,
                )
                return DeploymentResult(
                    success=False,
                    error_message=(f"Framework Git ref {git_tag!r} not found at {git_url}"),
                    external_resource_ids={
                        "error_kind": tag_probe.error_kind,
                        "git_tag": git_tag,
                        "git_url": git_url,
                        "probe_note": tag_probe.note,
                    },
                )
            # M3 — reject branches when tag-only is required. Branches can be
            # force-pushed under deployed apps and silently change framework
            # code on the next pip install. Tags are immutable by convention.
            if (
                settings.deploy_here_require_tag_only
                and tag_probe.ref_kind == "branch"
            ):
                logger.info(
                    "DEPLOYMENT_HERE_STAGE deployment=%s stage=tag_preflight outcome=err "
                    "git_tag=%s git_url=%s ref_kind=branch reason=tag_required",
                    deployment.id,
                    git_tag,
                    git_url,
                )
                return DeploymentResult(
                    success=False,
                    error_message=(
                        f"Framework Git ref {git_tag!r} resolves to a branch, "
                        "but tag-only deploys are required. Tag the commit and "
                        "redeploy against the tag (branches can be force-pushed "
                        "and silently change framework code under your app)."
                    ),
                    external_resource_ids={
                        "error_kind": "framework_ref_is_branch",
                        "git_tag": git_tag,
                        "git_url": git_url,
                        "ref_kind": "branch",
                    },
                )

        # ------------------------------------------------------------------
        # Step 4 — resolve deployer email
        # ------------------------------------------------------------------
        def _get_me() -> Any:
            return workspace_client.current_user.me()

        try:
            current_user = await asyncio.to_thread(_get_me)
            deployer_email: str = current_user.user_name or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not resolve current user email, falling back to 'unknown': %s",
                exc,
            )
            deployer_email = "unknown"

        # ------------------------------------------------------------------
        # Step 5 — compute workspace path and app name
        # ------------------------------------------------------------------
        deployment_id = str(deployment.id)
        workspace_path = f"/Workspace/Shared/deep-research-agent/shell-apps/{deployment_id}"
        from deep_research.core.databricks_auth import (  # noqa: PLC0415
            get_databricks_auth,
        )

        source_client = get_databricks_auth().get_client()
        app_name = _resolve_app_name(config, deployment_id)
        logger.info(
            "DEPLOY_HERE_START deployment=%s app_name=%s workspace_path=%s "
            "file_count=%s total_bytes=%s",
            deployment_id,
            app_name,
            workspace_path,
            len(all_files),
            total_bytes,
        )

        # ------------------------------------------------------------------
        # Steps 6-9 — upload + create/update with inline rollback
        # ------------------------------------------------------------------
        upload_started = False
        try:
            # Step 6 — upload files
            upload_started = True
            logger.info(
                "DEPLOY_HERE_UPLOAD_START deployment=%s app_name=%s workspace_path=%s",
                deployment_id,
                app_name,
                workspace_path,
            )
            await _upload_source_tree(
                workspace_client=source_client,
                workdir=workdir,
                all_files=all_files,
                workspace_path=workspace_path,
            )
            logger.info(
                "DEPLOY_HERE_UPLOAD_OK deployment=%s app_name=%s workspace_path=%s",
                deployment_id,
                app_name,
                workspace_path,
            )

            # Step 7 — app create-or-update (S3 owner-aware collision)
            logger.info(
                "DEPLOY_HERE_APPS_CREATE_OR_UPDATE_START deployment=%s app_name=%s "
                "deployer_email=%s",
                deployment_id,
                app_name,
                deployer_email,
            )
            app, collision_result = await _create_or_update_app(
                workspace_client=workspace_client,
                app_name=app_name,
                workspace_path=workspace_path,
                deployer_email=deployer_email,
                settings=settings,
                runtime_bindings=runtime_bindings,
            )
            if collision_result is not None:
                # Collision with another user's app — clean up the uploaded
                # files and surface the collision error.
                await _cleanup_workspace_path(source_client, workspace_path)
                return collision_result

        except _PermissionDeniedError as exc:
            # PermissionDenied — do NOT rollback (no point), return error
            logger.warning(
                "DEPLOY_HERE_PERMISSION_DENIED deployment=%s app_name=%s "
                "workspace_path=%s exc_class=%s exc_msg=%s",
                deployment_id,
                app_name,
                workspace_path,
                type(exc).__name__,
                str(exc),
            )
            if upload_started:
                await _cleanup_workspace_path(source_client, workspace_path)
            return DeploymentResult(
                success=False,
                error_message=str(exc),
                external_resource_ids={"error_kind": "missing_workspace_permission"},
            )
        except Exception as exc:  # noqa: BLE001
            # Rollback: delete uploaded files if upload already started
            logger.warning(
                "DEPLOY_HERE_FAILED deployment=%s app_name=%s workspace_path=%s "
                "exc_class=%s exc_msg=%s inferred_error_kind=%s",
                deployment_id,
                app_name,
                workspace_path,
                type(exc).__name__,
                str(exc),
                _infer_error_kind(exc),
            )
            if upload_started:
                await _cleanup_workspace_path(source_client, workspace_path)
            return DeploymentResult(
                success=False,
                error_message=f"{type(exc).__name__}: {exc}",
                external_resource_ids={
                    "error_kind": _infer_error_kind(exc),
                },
            )

        # ------------------------------------------------------------------
        # Step 8 — reachability probe (S4b: timeout → log tail + failure)
        # ------------------------------------------------------------------
        reachability = _coerce_reachability_result(
            await _probe_app_reachability_with_timeout(
                workspace_client=workspace_client,
                app_name=app_name,
            )
        )

        if reachability.timed_out or reachability.failed:
            error_kind = "reachability_failed" if reachability.failed else "reachability_timeout"
            error_message = (
                f"App reached failed state {reachability.last_state}."
                if reachability.failed
                else "App did not reach RUNNING within timeout."
            )
            tail = await fetch_app_log_tail(
                workspace_client=workspace_client,
                app_name=app_name,
            )
            logger.info(
                "DEPLOYMENT_HERE_STAGE deployment=%s stage=%s "
                "app_name=%s last_state=%s last_message=%s logs_available=%s",
                deployment.id,
                error_kind,
                app_name,
                reachability.last_state,
                _clip_for_log(reachability.last_message),
                tail is not None,
            )
            return DeploymentResult(
                success=False,
                error_message=error_message,
                external_resource_ids={
                    "error_kind": error_kind,
                    "app_name": app_name,
                    "deployment_path": workspace_path,
                    "last_state": reachability.last_state,
                    "last_status_message": reachability.last_message,
                    "last_logs": tail.text if tail else None,
                    "logs_truncated": tail.truncated if tail else None,
                    "logs_source": tail.source if tail else None,
                },
            )

        reached = reachability.reached

        # ------------------------------------------------------------------
        # Step 10 — return success
        # ------------------------------------------------------------------
        app_url: str | None = getattr(app, "url", None)
        logger.info(
            "DEPLOY_HERE_SUCCESS deployment=%s app_name=%s app_url=%s reachability_status=%s",
            deployment_id,
            app_name,
            app_url,
            "running" if reached else "provisioning",
        )
        return DeploymentResult(
            success=True,
            endpoint_name=app_name,
            external_resource_ids={
                "app_name": app_name,
                "app_url": app_url,
                "deployment_path": workspace_path,
                "reachability_status": "running" if reached else "provisioning",
            },
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _PermissionDeniedError(Exception):
    """Raised internally when the SDK returns a PermissionDenied error."""


async def _upload_source_tree(
    workspace_client: Any,
    workdir: Path,
    all_files: list[Path],
    workspace_path: str,
) -> None:
    """Upload all files from ``workdir`` to ``workspace_path`` in the workspace."""
    # Lazy-import SDK symbol
    from databricks.sdk.service.workspace import ImportFormat  # noqa: PLC0415

    # Collect unique parent directories and create them
    parent_dirs: set[str] = set()
    for file_path in all_files:
        relative = file_path.relative_to(workdir)
        # Build the workspace parent dir for this file
        ws_file_dir = workspace_path
        parts = relative.parts
        for part in parts[:-1]:
            ws_file_dir = f"{ws_file_dir}/{part}"
        if ws_file_dir != workspace_path:
            parent_dirs.add(ws_file_dir)

    # Create parent dirs (including workspace_path itself)
    dirs_to_create = sorted({workspace_path} | parent_dirs)
    for d in dirs_to_create:

        def _mkdirs(path: str = d) -> None:
            workspace_client.workspace.mkdirs(path=path)

        try:
            await asyncio.to_thread(_mkdirs)
        except Exception as exc:  # noqa: BLE001
            # mkdirs is idempotent; AlreadyExists is fine
            if not _is_already_exists_error(exc):
                if _is_permission_denied_error(exc):
                    raise _PermissionDeniedError(str(exc)) from exc
                raise

    # Upload each file
    for file_path in all_files:
        relative = file_path.relative_to(workdir)
        ws_file_path = workspace_path + "/" + "/".join(relative.parts)
        content = file_path.read_bytes()

        def _upload(
            path: str = ws_file_path,
            data: bytes = content,
        ) -> None:
            workspace_client.workspace.upload(
                path=path,
                content=data,
                format=ImportFormat.RAW,
                overwrite=True,
            )

        try:
            await asyncio.to_thread(_upload)
        except Exception as exc:  # noqa: BLE001
            if _is_permission_denied_error(exc):
                raise _PermissionDeniedError(str(exc)) from exc
            raise


async def _create_or_update_app(
    workspace_client: Any,
    app_name: str,
    workspace_path: str,
    deployer_email: str,
    settings: Any,
    runtime_bindings: ShellAppRuntimeBindings,
) -> tuple[Any, DeploymentResult | None]:
    """Create the Databricks App, updating if it already exists.

    Returns ``(app_object, None)`` on success, or ``(None, DeploymentResult)``
    when a collision prevents deploy (``error_kind="app_name_collision"``).
    Raises ``_PermissionDeniedError`` on PermissionDenied.

    Uses :func:`resolve_apps_already_exists` (S3) for owner-aware collision
    resolution and bounds retries on the race-deleted path to 1.
    """
    from databricks.sdk.service.apps import (  # noqa: PLC0415
        App,
        AppDeployment,
        AppDeploymentMode,
        AppResource,
        AppResourceSecret,
        AppResourceSecretSecretPermission,
        EnvVar,
    )

    def _app_resources() -> list[Any] | None:
        if not runtime_bindings.requires_web_search:
            return None
        return [
            AppResource(
                name=runtime_bindings.brave_secret_resource_name,
                description="Brave Search API key for web_search tools",
                secret=AppResourceSecret(
                    scope=runtime_bindings.brave_secret_scope or "",
                    key=runtime_bindings.brave_secret_key or "",
                    permission=AppResourceSecretSecretPermission.READ,
                ),
            )
        ]

    def _app_definition() -> Any:
        return App(
            name=app_name,
            resources=_app_resources(),
            user_api_scopes=list(_APP_USER_API_SCOPES),
        )

    def _deployment_env_vars() -> list[Any]:
        env_vars = [EnvVar(name="MLFLOW_TRACKING_URI", value="databricks")]
        if runtime_bindings.requires_web_search:
            env_vars.append(
                EnvVar(
                    name="BRAVE_API_KEY",
                    value_from=runtime_bindings.brave_secret_resource_name,
                )
            )
        return env_vars

    def _do_create() -> Any:
        return workspace_client.apps.create(_app_definition()).result()

    def _do_update() -> Any:
        return workspace_client.apps.update(name=app_name, app=_app_definition())

    def _do_deploy() -> Any:
        return workspace_client.apps.deploy(
            app_name=app_name,
            app_deployment=AppDeployment(
                source_code_path=workspace_path,
                mode=AppDeploymentMode.SNAPSHOT,
                env_vars=_deployment_env_vars(),
            ),
        ).result()

    def _do_get_app() -> Any:
        return workspace_client.apps.get(name=app_name)

    async def _deploy_and_get_app() -> Any:
        await asyncio.to_thread(_do_deploy)
        return await asyncio.to_thread(_do_get_app)

    # Bounded retry loop: at most 2 iterations (create + one retry on
    # race-deleted path).
    for attempt in range(2):
        try:
            logger.info(
                "DEPLOY_HERE_APPS_CREATE_ATTEMPT app_name=%s attempt=%s workspace_path=%s",
                app_name,
                attempt + 1,
                workspace_path,
            )
            await asyncio.to_thread(_do_create)
            app = await _deploy_and_get_app()
            logger.info(
                "DEPLOY_HERE_APPS_CREATE_DEPLOY_OK app_name=%s attempt=%s "
                "requires_web_search=%s env_vars=%s resources=%s",
                app_name,
                attempt + 1,
                runtime_bindings.requires_web_search,
                [env.name for env in _deployment_env_vars()],
                [resource.name for resource in (_app_resources() or [])],
            )
            return app, None
        except Exception as exc:  # noqa: BLE001
            if _is_permission_denied_error(exc):
                logger.warning(
                    "DEPLOY_HERE_APPS_PERMISSION_DENIED app_name=%s attempt=%s "
                    "exc_class=%s exc_msg=%s",
                    app_name,
                    attempt + 1,
                    type(exc).__name__,
                    str(exc),
                )
                raise _PermissionDeniedError(str(exc)) from exc
            if not _is_already_exists_error(exc):
                logger.warning(
                    "DEPLOY_HERE_APPS_CREATE_FAILED app_name=%s attempt=%s exc_class=%s exc_msg=%s",
                    app_name,
                    attempt + 1,
                    type(exc).__name__,
                    str(exc),
                )
                raise

            # AlreadyExists — check ownership.
            logger.info(
                "DEPLOY_HERE_APPS_ALREADY_EXISTS app_name=%s attempt=%s",
                app_name,
                attempt + 1,
            )
            check: AppOwnershipCheck = await resolve_apps_already_exists(
                workspace_client=workspace_client,
                app_name=app_name,
                deployer_email=deployer_email,
            )

            if check.failure_reason == "race_deleted" and attempt == 0:
                # App was deleted between our create and the get — retry create.
                logger.info(
                    "DEPLOY_HERE_APPS_RACE_DELETED_RETRY app_name=%s attempt=%s",
                    app_name,
                    attempt + 1,
                )
                continue

            if check.deployer_can_redeploy:
                try:
                    await asyncio.to_thread(_do_update)
                    app = await _deploy_and_get_app()
                    logger.info(
                        "DEPLOY_HERE_APPS_REDEPLOY_OK app_name=%s owner=%s "
                        "requires_web_search=%s env_vars=%s resources=%s",
                        app_name,
                        check.existing_owner,
                        runtime_bindings.requires_web_search,
                        [env.name for env in _deployment_env_vars()],
                        [resource.name for resource in (_app_resources() or [])],
                    )
                    return app, None
                except Exception as update_exc:  # noqa: BLE001
                    if _is_permission_denied_error(update_exc):
                        logger.warning(
                            "DEPLOY_HERE_APPS_REDEPLOY_PERMISSION_DENIED "
                            "app_name=%s owner=%s exc_class=%s exc_msg=%s",
                            app_name,
                            check.existing_owner,
                            type(update_exc).__name__,
                            str(update_exc),
                        )
                        raise _PermissionDeniedError(str(update_exc)) from update_exc
                    logger.warning(
                        "DEPLOY_HERE_APPS_REDEPLOY_FAILED app_name=%s owner=%s "
                        "exc_class=%s exc_msg=%s",
                        app_name,
                        check.existing_owner,
                        type(update_exc).__name__,
                        str(update_exc),
                    )
                    raise

            # Collision: someone else owns this app.
            suggested = generate_suggested_name(app_name=app_name, deployer_email=deployer_email)
            disclosed_owner: str | None = (
                check.existing_owner if settings.deploy_here_disclose_owner else None
            )
            logger.info(
                "DEPLOY_HERE_APPS_COLLISION app_name=%s disclosed_owner=%s "
                "suggested_name=%s failure_reason=%s",
                app_name,
                disclosed_owner,
                suggested,
                check.failure_reason,
            )
            return None, DeploymentResult(
                success=False,
                error_message=(
                    f"App name {app_name!r} is owned by {disclosed_owner or 'another user'}."
                ),
                external_resource_ids={
                    "error_kind": "app_name_collision",
                    "existing_owner": disclosed_owner,
                    "suggested_name": suggested,
                },
            )

    # Should not be reached but return a collision result if somehow we exit
    # the loop without returning.
    return None, DeploymentResult(
        success=False,
        error_message=f"App {app_name!r} already exists and could not be claimed.",
        external_resource_ids={"error_kind": "app_name_collision"},
    )


async def _probe_app_reachability_with_timeout(
    workspace_client: Any,
    app_name: str,
) -> ReachabilityProbeResult:
    """Poll until App status is ready or timeout elapsed.

    Returns a :class:`ReachabilityProbeResult` with the last observed status.
    Databricks Apps can accept a deployment and then fail while the app process
    starts (for example, package build failures). Treating terminal failure
    states as a distinct result avoids waiting the full timeout after the app
    has already failed. The Apps API may report a healthy app as either
    ``RUNNING`` or ``ACTIVE`` depending on the response surface.
    """
    settings = get_settings()
    timeout_sec = settings.deploy_here_reachability_timeout_seconds
    deadline = time.monotonic() + timeout_sec
    last_state: str | None = None
    last_message: str | None = None
    last_logged_state: str | None = None

    def _do_get() -> Any:
        return workspace_client.apps.get(name=app_name)

    while time.monotonic() < deadline:
        try:
            app = await asyncio.to_thread(_do_get)
            compute_status = getattr(app, "compute_status", None)
            state: str | None = None
            if compute_status is not None:
                state = _normalize_status_value(getattr(compute_status, "state", None))
                message = getattr(compute_status, "message", None)
                last_message = str(message) if message else None
            last_state = state or last_state
            if state is not None and state != last_logged_state:
                logger.info(
                    "DEPLOY_HERE_REACHABILITY_POLL app_name=%s state=%s message=%s",
                    app_name,
                    state,
                    _clip_for_log(last_message),
                )
                last_logged_state = state
            if state in _APP_RUNNING_STATES:
                logger.info(
                    "DEPLOY_HERE_REACHABILITY_READY app_name=%s state=%s message=%s",
                    app_name,
                    state,
                    _clip_for_log(last_message),
                )
                return ReachabilityProbeResult(
                    reached=True,
                    timed_out=False,
                    last_state=state,
                    last_message=last_message,
                )
            if state in _APP_FAILED_STATES:
                logger.info(
                    "DEPLOY_HERE_REACHABILITY_FAILED app_name=%s state=%s message=%s",
                    app_name,
                    state,
                    _clip_for_log(last_message),
                )
                return ReachabilityProbeResult(
                    reached=False,
                    timed_out=False,
                    failed=True,
                    last_state=state,
                    last_message=last_message,
                )
        except Exception as exc:  # noqa: BLE001
            logger.info(
                "DEPLOY_HERE_REACHABILITY_PROBE_ERROR app_name=%s exc_class=%s exc_msg=%s",
                app_name,
                type(exc).__name__,
                _clip_for_log(str(exc)),
            )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        await asyncio.sleep(min(_PROBE_POLL_INTERVAL_SEC, remaining))

    return ReachabilityProbeResult(
        reached=False,
        timed_out=True,
        last_state=last_state,
        last_message=last_message,
    )


def _coerce_reachability_result(
    raw: ReachabilityProbeResult | tuple[bool, bool],
) -> ReachabilityProbeResult:
    """Normalize old tuple-shaped test doubles to ``ReachabilityProbeResult``."""
    if isinstance(raw, ReachabilityProbeResult):
        return raw
    reached, timed_out = raw
    return ReachabilityProbeResult(reached=reached, timed_out=timed_out)


def _normalize_status_value(value: object | None) -> str | None:
    """Return a comparable Databricks status value from SDK enum/string shapes."""
    if value is None:
        return None
    raw = getattr(value, "value", value)
    text = str(raw)
    if "." in text:
        text = text.rsplit(".", maxsplit=1)[-1]
    return text.upper()


def _clip_for_log(value: object | None, *, limit: int = 500) -> str | None:
    """Keep diagnostic log fields single-line and bounded."""
    if value is None:
        return None
    text = str(value).replace("\n", "\\n").replace("\r", "\\r")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...<truncated>"


async def _cleanup_workspace_path(
    workspace_client: Any,
    workspace_path: str,
) -> None:
    """Best-effort cleanup: delete workspace_path recursively.

    Treats NotFound/404 as success. Swallows other cleanup errors but logs them.
    """
    try:
        await delete_workspace_source_tree(workspace_client, workspace_path)
    except Exception as exc:  # noqa: BLE001
        if _is_not_found_error(exc):
            logger.debug("Cleanup: workspace path %s already gone (NotFound)", workspace_path)
        else:
            logger.warning(
                "Cleanup of workspace path %s failed (swallowed): %s",
                workspace_path,
                exc,
            )


async def delete_workspace_source_tree(
    workspace_client: Any,
    workspace_path: str,
) -> None:
    """Delete an uploaded source tree using the Workspace API."""

    def _delete() -> None:
        workspace_client.workspace.delete(path=workspace_path, recursive=True)

    await asyncio.to_thread(_delete)


# ---------------------------------------------------------------------------
# SDK error-class heuristics (no hard imports required)
# ---------------------------------------------------------------------------


def _is_not_found_error(exc: BaseException) -> bool:
    """Heuristic 404/NotFound detector for upstream SDK exceptions."""
    cls_name = type(exc).__name__.lower()
    if "notfound" in cls_name or "doesnotexist" in cls_name:
        return True
    msg = str(exc).lower()
    return "404" in msg or "not found" in msg or "does not exist" in msg


def _is_already_exists_error(exc: BaseException) -> bool:
    """Heuristic AlreadyExists/Conflict detector for upstream SDK exceptions."""
    cls_name = type(exc).__name__.lower()
    if "alreadyexists" in cls_name or "conflict" in cls_name:
        return True
    msg = str(exc).lower()
    return "already exists" in msg or "409" in msg


def _is_permission_denied_error(exc: BaseException) -> bool:
    """Heuristic PermissionDenied detector for upstream SDK exceptions."""
    cls_name = type(exc).__name__.lower()
    if "permissiondenied" in cls_name or "forbidden" in cls_name or "unauthorized" in cls_name:
        return True
    msg = str(exc).lower()
    return "403" in msg or "permission denied" in msg or "unauthorized" in msg


def _infer_error_kind(exc: BaseException) -> str:
    """Map an exception to a short error_kind string for external_resource_ids."""
    if _is_permission_denied_error(exc):
        return "missing_workspace_permission"
    if _is_not_found_error(exc):
        return "resource_not_found"
    if _is_already_exists_error(exc):
        return "resource_already_exists"
    return "unknown_error"
