"""ShellAppExporter: Mode 2 — standalone Databricks App with chat UI.

Generates a downloadable zip from the 8-file template under
``templates/agent-shell-app/`` (Jinja-rendered .j2 files + verbatim copies of
.py / .yaml / .sh / .html / .md). The agent's ``WorkflowDefinition`` AST is
serialized into ``agent.yaml`` and embedded in the zip.

Plan reference: agent-designer-deployment.md Section E (Shell-app). The
framework is no longer installed from GitHub at app-startup time; instead a
locally-built ``databricks_deep_research-*.whl`` is bundled into the zip at
``wheels/`` and referenced from the generated ``pyproject.toml`` via
``[tool.uv.sources]``. See plan imperative-wishing-lynx.md.

Phase 2-B ships the zip artifact + recorded metadata; live deploy via
``w.apps.create`` lands in Phase 3 alongside MLflow agent live deploy.
"""
# Method args (deployment) carry context required by DeploymentTranslator
# but are not all used in Phase 2-B's stub deploy/deactivate paths.
# ruff: noqa: ARG002
from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import os
import re
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import yaml
from jinja2 import StrictUndefined, Template

from deep_research.models.agent_deployment import AgentDeployment, DeploymentMode
from deep_research.services.deployment._paths import resolve_package_data_dir
from deep_research.services.deployment.translator import (
    Artifact,
    DeploymentCleanupError,
    DeploymentCleanupExhaustedError,
    DeploymentResult,
    ValidationError,
    ValidationResult,
)

if TYPE_CHECKING:
    from deep_research.services.deployment.auth import WorkspaceClientResolver

logger = logging.getLogger(__name__)
_DEFAULT_BRAVE_SECRET_SCOPE = "deep-research-secrets"
_DEFAULT_BRAVE_SECRET_KEY = "BRAVE_API_KEY"

# Historical: shell-apps used to install the framework via a git URL pinned
# in their generated pyproject.toml; the value was validated against this
# whitelist before being rendered verbatim. The bundled-wheel path (see
# ``_resolve_framework_wheel`` below) makes the git ref obsolete. The regex
# stays defined for backwards-compatible type checking of any incoming
# ``framework_git_tag`` value (we log + ignore it now) and for MLflow
# agent-deploy mode in ``mlflow_deploy.py`` which still uses git URLs.
_GIT_REF_WHITELIST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$")

# Matches the version segment in a PEP 427 framework wheel filename:
#   ``databricks_deep_research-<version>-py3-none-any.whl``
_FRAMEWORK_WHEEL_RE = re.compile(
    r"^databricks_deep_research-(?P<version>[^-]+)-py3-none-any\.whl$"
)


class ShellAppWheelMissingError(RuntimeError):
    """Raised when the bundled framework wheel cannot be located.

    Surfaced at ``validate()`` time so the caller sees a clear error before
    the zip-build path starts; never raised from ``translate()``.
    """


def _resolve_framework_wheel() -> tuple[str, bytes]:
    """Return ``(filename, bytes)`` for the bundled framework wheel.

    Primary location: ``_framework_wheel/`` next to this file. Hatch
    ``force-include`` copies it there in both source-tree and installed-wheel
    layouts (see ``pyproject.toml`` ``[tool.hatch.build.targets.wheel.force-include]``).
    Populated by ``make build-framework``.

    Source-tree dev fallback: if the primary location is empty (devs running
    ``make dev`` without rebuilding), walk up to find the repo's
    ``databricks-deep-research-app/wheels/databricks_deep_research-*.whl``
    and use the newest match. Logs at INFO so the dev knows the fallback fired.

    Raises ``ShellAppWheelMissingError`` if neither path resolves.
    """
    primary_dir = Path(__file__).resolve().parent / "_framework_wheel"
    if primary_dir.is_dir():
        matches = [
            p for p in primary_dir.iterdir()
            if p.is_file() and _FRAMEWORK_WHEEL_RE.match(p.name)
        ]
        if len(matches) == 1:
            return matches[0].name, matches[0].read_bytes()
        if len(matches) > 1:
            raise ShellAppWheelMissingError(
                "Expected exactly one framework wheel in package data "
                f"({primary_dir}), found {len(matches)}: "
                f"{sorted(p.name for p in matches)}. Run `make build-framework`."
            )

    # Source-tree fallback: walk up to find <repo>/databricks-deep-research-app/wheels/.
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        candidate_dir = ancestor / "databricks-deep-research-app" / "wheels"
        if candidate_dir.is_dir():
            fallback_matches = sorted(
                (
                    p for p in candidate_dir.iterdir()
                    if p.is_file() and _FRAMEWORK_WHEEL_RE.match(p.name)
                ),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if fallback_matches:
                logger.info(
                    "SHELL_APP_FRAMEWORK_WHEEL_FALLBACK_SOURCE_TREE path=%s "
                    "(primary _framework_wheel/ empty; ran from source tree without "
                    "make build-framework)",
                    fallback_matches[0],
                )
                return fallback_matches[0].name, fallback_matches[0].read_bytes()
            break  # Found wheels dir but no framework wheel — stop searching.

    raise ShellAppWheelMissingError(
        "Framework wheel not found in package data "
        f"({primary_dir}) or in source-tree fallback "
        "(databricks-deep-research-app/wheels/). Run "
        "`make build-framework` in databricks-deep-research-app/ "
        "to build and stage it."
    )


def _parse_framework_wheel_version(filename: str) -> str:
    """Extract version from a framework wheel filename, or "unknown"."""
    match = _FRAMEWORK_WHEEL_RE.match(filename)
    return match.group("version") if match else "unknown"


def _is_not_found_error(exc: BaseException) -> bool:
    """Heuristic 404/NotFound detector for upstream SDK exceptions.

    Mirrors ``shell_app_apps_api._is_not_found_error`` — kept local to avoid
    a cross-module import between two sibling modules.
    """
    cls_name = type(exc).__name__.lower()
    if "notfound" in cls_name or "doesnotexist" in cls_name:
        return True
    msg = str(exc).lower()
    return "404" in msg or "not found" in msg or "does not exist" in msg


def _is_permission_denied_error(exc: BaseException) -> bool:
    """Detect PermissionDenied (HTTP 403) from the Databricks SDK.

    Strict-first: prefers ``isinstance`` against the canonical SDK class so
    we don't fight string formatting. Falls back to a narrow type-name /
    message check for wrapped or legacy paths where the original SDK
    exception was re-raised inside another class.

    Does NOT match a bare ``"403"`` in the message — that's too broad and
    would false-match unrelated errors that happen to mention the substring.
    """
    try:
        from databricks.sdk.errors.platform import PermissionDenied  # noqa: PLC0415

        if isinstance(exc, PermissionDenied):
            return True
    except Exception:  # noqa: BLE001
        # SDK class not importable for some reason — fall through to duck-type.
        pass
    cls_name = type(exc).__name__.lower()
    if "permissiondenied" in cls_name or "forbidden" in cls_name:
        return True
    msg = str(exc).lower()
    return "permission_denied" in msg or "permission denied" in msg


_TEMPLATE_DIR = resolve_package_data_dir(Path(__file__), "agent-shell-app")

# Files copied verbatim into the zip. Keys are template-relative paths;
# values are output paths inside the zip (often the same).
_VERBATIM_FILES: tuple[tuple[str, str], ...] = (
    ("app.py", "app.py"),
    ("entrypoint.sh", "entrypoint.sh"),
    ("static/index.html", "static/index.html"),
)

# .j2 templates rendered with str-format-style Jinja (StrictUndefined catches
# missing substitutions at render time). Values are output paths in the zip.
_JINJA_FILES: tuple[tuple[str, str], ...] = (
    ("app.yaml", "app.yaml"),
    ("databricks.yml.j2", "databricks.yml"),
    ("pyproject.toml.j2", "pyproject.toml"),
    ("agent.yaml.j2", "agent.yaml"),
    ("README.md", "README.md"),  # README is also Jinja-rendered (uses {{var}})
)

# Entries in the generated zip that MUST be executable. The Apps API and
# `databricks bundle deploy` both expect entrypoint.sh to have the +x bit
# set on the uploaded source. ZipInfo.external_attr encodes the file mode
# in the high 16 bits.
_EXEC_ENTRIES: frozenset[str] = frozenset({"entrypoint.sh"})
_BRAVE_SECRET_RESOURCE_NAME = "brave-api-key"


def _zip_mode_bits(dst: str) -> int:
    """Return external_attr value (mode bits in high 16 bits) for a zip entry."""
    mode = 0o755 if dst in _EXEC_ENTRIES else 0o644
    return mode << 16


def _load_template(name: str) -> str:
    return (_TEMPLATE_DIR / name).read_text("utf-8")


def _render(template: str, **context: Any) -> str:
    return Template(template, undefined=StrictUndefined).render(**context)


def _definition_uses_web_search(definition: dict[str, Any]) -> bool:
    """Return True when the workflow references the built-in web_search tool."""

    def _walk(value: Any) -> bool:
        if isinstance(value, dict):
            if value.get("kind") == "web_search":
                return True
            if value.get("tool") == "web_search" or value.get("ref") == "web_search":
                return True
            for key, child in value.items():
                if key == "tools" and _walk_tool_refs(child):
                    return True
                if key != "tools" and _walk(child):
                    return True
            return False
        if isinstance(value, list):
            return any(_walk(child) for child in value)
        return False

    def _walk_tool_refs(value: Any) -> bool:
        if isinstance(value, str):
            return value == "web_search"
        return _walk(value)

    return _walk(definition)


def _resolve_brave_secret_config(
    config: dict[str, Any],
    *,
    include_defaults: bool,
) -> tuple[str | None, str | None]:
    """Resolve Brave secret location without constructing full app Settings.

    Shell-app export can run in contexts that do not have database settings
    loaded, and non-web workflows do not need Brave bindings at all. Read only
    the specific deploy-here env vars needed for this binding.
    """
    scope = config.get("brave_secret_scope")
    key = config.get("brave_secret_key")
    if include_defaults:
        scope = (
            scope
            or os.environ.get("DEPLOY_HERE_BRAVE_SECRET_SCOPE")
            or _DEFAULT_BRAVE_SECRET_SCOPE
        )
        key = (
            key
            or os.environ.get("DEPLOY_HERE_BRAVE_SECRET_KEY")
            or _DEFAULT_BRAVE_SECRET_KEY
        )
    return (
        str(scope).strip() if scope else None,
        str(key).strip() if key else None,
    )


def _preview(value: Any, *, max_length: int = 200) -> str:
    """Return bounded diagnostic text for logs."""
    text = " ".join(str(value or "").split())
    if len(text) <= max_length:
        return text
    return text[: max_length - 15].rstrip() + " ...(truncated)"


def _root_child_summary(definition: dict[str, Any]) -> list[str]:
    root = definition.get("root")
    if not isinstance(root, dict):
        return []
    children = root.get("children")
    if not isinstance(children, list):
        return []
    summary: list[str] = []
    for child in children:
        if not isinstance(child, dict):
            continue
        node_id = child.get("id") or "<unnamed>"
        node_type = child.get("type") or "<unknown>"
        label = child.get("label") or ""
        summary.append(f"{node_id}:{node_type}:{label}")
    return summary


def _first_planner_guidance(definition: dict[str, Any]) -> str:
    def _walk(node: Any) -> str:
        if not isinstance(node, dict):
            return ""
        if node.get("type") == "plan_and_execute":
            config = node.get("config")
            if isinstance(config, dict):
                guidance = config.get("planner_guidance")
                if isinstance(guidance, str) and guidance.strip():
                    return guidance
        config = node.get("config")
        if isinstance(config, dict):
            body_guidance = _walk(config.get("body"))
            if body_guidance:
                return body_guidance
        children = node.get("children")
        if isinstance(children, list):
            for child in children:
                child_guidance = _walk(child)
                if child_guidance:
                    return child_guidance
        return ""

    return _walk(definition.get("root"))


class ShellAppExporter:
    """Translator for ``DeploymentMode.SHELL_APP`` (standalone Databricks App).

    Produces a downloadable zip. Phase 2-B does not call the Databricks Apps
    REST API directly; the user runs ``databricks bundle deploy`` against the
    extracted zip.
    """

    mode: ClassVar[DeploymentMode] = DeploymentMode.SHELL_APP

    async def validate(
        self,
        agent: Any,
        revision: Any,
        config: dict[str, Any],
    ) -> ValidationResult:
        """Reject configs missing required fields and ASTs containing custom
        tools (custom tools require app context unavailable in standalone
        shell apps; see plan Section M per-mode feature handling)."""
        errors: list[ValidationError] = []

        app_name = config.get("app_name", "")
        if (
            not isinstance(app_name, str)
            or not app_name.startswith("dr-shell-")
            or len(app_name) > 30
        ):
            errors.append(
                ValidationError(
                    message="app_name must start with 'dr-shell-' and be 30 chars or fewer",
                    path="config.app_name",
                )
            )

        # framework_git_tag is no longer rendered into the generated shell-app
        # pyproject.toml — the framework now ships as a bundled wheel (see
        # _resolve_framework_wheel below + plan imperative-wishing-lynx.md).
        # We keep accepting the field on the request payload for backwards
        # compatibility but the value is ignored at translate() time. The
        # whitelist still runs as a typing guard so malformed values don't
        # silently slip through into logs / metadata.
        git_tag = config.get("framework_git_tag", "")
        if git_tag and (
            not isinstance(git_tag, str) or not _GIT_REF_WHITELIST.fullmatch(git_tag)
        ):
            errors.append(
                ValidationError(
                    message=(
                        "framework_git_tag, when supplied, must be a valid "
                        "Git ref. Allowed: alphanumerics, '.', '_', '-', '/'. "
                        "Must not begin with '.', '-', or '/' and must be 256 "
                        "chars or fewer. Disallowed characters: '@', '#', '?', "
                        "whitespace, and quotes. Note: this field is ignored — "
                        "the framework is now installed from a bundled wheel."
                    ),
                    path="config.framework_git_tag",
                )
            )

        # The shell-app deploy bundles the locally-built framework wheel into
        # the generated zip. Surface a clear error if the wheel hasn't been
        # built yet so the user sees actionable feedback before the zip-build
        # path starts.
        try:
            _resolve_framework_wheel()
        except ShellAppWheelMissingError as exc:
            errors.append(
                ValidationError(
                    message=str(exc),
                    path="server.framework_wheel",
                )
            )

        # Reject AST containing custom tool kinds. AgentRevision.definition
        # is a JSONB dict at this point (see AgentV2 model). The 'tools' key
        # is the top-level tool list; each entry has a 'kind' string field.
        definition = revision.definition or {}
        uses_web_search = _definition_uses_web_search(definition)
        for tool in definition.get("tools", []) or []:
            if isinstance(tool, dict) and tool.get("kind") == "custom":
                errors.append(
                    ValidationError(
                        message=(
                            "Custom tools are not supported in shell-app "
                            "deployments (require app context). Either replace "
                            "the custom tool with a built-in equivalent or "
                            "deploy in-app instead."
                        ),
                        path=f"definition.tools[{tool.get('name', '<unnamed>')!r}]",
                    )
                )

        if uses_web_search:
            scope, key = _resolve_brave_secret_config(config, include_defaults=True)
            if not scope or not key:
                errors.append(
                    ValidationError(
                        message=(
                            "Shell-app workflows using web_search require a "
                            "Databricks secret binding for BRAVE_API_KEY. Set "
                            "brave_secret_scope/brave_secret_key or configure "
                            "DEPLOY_HERE_BRAVE_SECRET_SCOPE and "
                            "DEPLOY_HERE_BRAVE_SECRET_KEY."
                        ),
                        path="config.brave_secret_scope",
                    )
                )

        return ValidationResult(valid=not errors, errors=errors)

    async def translate(
        self,
        agent: Any,
        revision: Any,
        config: dict[str, Any],
    ) -> Artifact:
        """Render the 8 template files + zip them into an in-memory artifact."""
        app_name: str = config["app_name"]
        # framework_git_tag is accepted on the payload for backwards compat
        # but no longer rendered into the generated pyproject.toml. The
        # framework now ships as a bundled wheel; see _resolve_framework_wheel.
        legacy_git_tag = config.get("framework_git_tag") or ""
        if legacy_git_tag:
            logger.warning(
                "SHELL_APP_FRAMEWORK_GIT_TAG_IGNORED app_name=%s framework_git_tag=%r "
                "(field is ignored — framework installs from bundled wheel)",
                app_name,
                legacy_git_tag,
            )
        target: str = config.get("target", "dev")
        definition = revision.definition or {}
        uses_web_search = _definition_uses_web_search(definition)
        brave_secret_scope, brave_secret_key = _resolve_brave_secret_config(
            config,
            include_defaults=uses_web_search,
        )

        framework_wheel_filename, framework_wheel_bytes = _resolve_framework_wheel()
        framework_wheel_version = _parse_framework_wheel_version(
            framework_wheel_filename
        )

        # Serialize the workflow definition into a YAML string that we splice
        # into agent.yaml.j2 below. ``default_flow_style=False`` keeps it
        # human-readable (block style).
        definition_yaml = yaml.safe_dump(
            definition, default_flow_style=False, sort_keys=False
        )

        context = {
            "app_name": app_name,
            "framework_wheel_filename": framework_wheel_filename,
            "target": target,
            "agent_name": getattr(agent, "name", "Untitled Agent"),
            "agent_id": str(agent.id),
            "revision_id": str(revision.rev_id),
            "definition_yaml": definition_yaml,
            "requires_web_search": uses_web_search,
            "brave_secret_scope": brave_secret_scope,
            "brave_secret_key": brave_secret_key,
            "brave_secret_resource_name": _BRAVE_SECRET_RESOURCE_NAME,
        }

        logger.info(
            "SHELL_APP_TRANSLATE_RUNTIME_REQUIREMENTS app_name=%s requires_web_search=%s "
            "brave_secret_scope_configured=%s brave_secret_key_configured=%s",
            app_name,
            uses_web_search,
            bool(brave_secret_scope),
            bool(brave_secret_key),
        )
        planner_guidance = _first_planner_guidance(definition)
        logger.info(
            "SHELL_APP_TRANSLATE_WORKFLOW_SUMMARY app_name=%s agent_id=%s revision_id=%s "
            "workflow_name=%s workflow_description=%s root_children=%s "
            "planner_guidance_present=%s planner_guidance=%s",
            app_name,
            str(agent.id),
            str(revision.rev_id),
            _preview(definition.get("name")),
            _preview(definition.get("description")),
            _root_child_summary(definition),
            bool(planner_guidance),
            _preview(planner_guidance),
        )

        # Build the zip in memory. Use a fixed timestamp (1980-01-01, the
        # zip-epoch minimum) for every entry so regeneration from the same
        # inputs is byte-deterministic — required for the integrity check
        # in the /export-zip route (W7). Reproducible-build convention.
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for src, dst in _VERBATIM_FILES:
                info = zipfile.ZipInfo(dst, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = _zip_mode_bits(dst)
                zf.writestr(info, _load_template(src))
            for src, dst in _JINJA_FILES:
                info = zipfile.ZipInfo(dst, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = _zip_mode_bits(dst)
                zf.writestr(info, _render(_load_template(src), **context))
            # Bundle the framework wheel at wheels/<filename>. The generated
            # pyproject.toml's [tool.uv.sources] block points at this path so
            # the deployed app installs the framework from a local file
            # instead of git+https. Stays byte-deterministic via the fixed
            # 1980-01-01 timestamp and stable wheel filename.
            wheel_dst = f"wheels/{framework_wheel_filename}"
            wheel_info = zipfile.ZipInfo(wheel_dst, date_time=(1980, 1, 1, 0, 0, 0))
            wheel_info.compress_type = zipfile.ZIP_DEFLATED
            wheel_info.external_attr = _zip_mode_bits(wheel_dst)
            zf.writestr(wheel_info, framework_wheel_bytes)

        payload = buf.getvalue()
        digest = hashlib.sha256(payload).hexdigest()

        return Artifact(
            mode=DeploymentMode.SHELL_APP,
            payload=payload,
            metadata={
                "app_name": app_name,
                "framework_wheel_filename": framework_wheel_filename,
                "framework_wheel_version": framework_wheel_version,
                "requires_web_search": str(uses_web_search).lower(),
                "brave_secret_resource_name": _BRAVE_SECRET_RESOURCE_NAME,
                "brave_secret_scope_configured": str(bool(brave_secret_scope)).lower(),
                "brave_secret_key_configured": str(bool(brave_secret_key)).lower(),
                "sha256": digest,
                "size_bytes": str(len(payload)),
            },
        )

    async def deploy(
        self,
        artifact: Artifact,
        config: dict[str, Any],
        deployment: AgentDeployment,
    ) -> DeploymentResult:
        """Phase 2-B stub: record the SHA256 + app_name in
        ``external_resource_ids``. Phase 3 will replace this body with
        ``WorkspaceClient.apps.create`` + sync of the zip contents.
        """
        if not isinstance(artifact.payload, bytes):
            return DeploymentResult(
                success=False,
                error_message="ShellAppExporter artifact payload must be bytes",
            )
        return DeploymentResult(
            success=True,
            endpoint_name=config["app_name"],
            external_resource_ids={
                "app_name": config["app_name"],
                "shell_app_zip_sha256": hashlib.sha256(artifact.payload).hexdigest(),
                "framework_wheel_filename": artifact.metadata.get(
                    "framework_wheel_filename", ""
                ),
                "framework_wheel_version": artifact.metadata.get(
                    "framework_wheel_version", ""
                ),
                "size_bytes": str(len(artifact.payload)),
            },
        )

    async def deploy_inline(
        self,
        artifact: Artifact,
        config: dict[str, Any],
        deployment: AgentDeployment,
        workspace_client: Any,
    ) -> DeploymentResult:
        """Inline-synchronous deploy using a request-scoped WorkspaceClient.

        Delegates to ``_deploy_via_apps_api`` (US-402) which handles the full
        upload + App create/update + reachability probe lifecycle.
        """
        from deep_research.services.deployment.shell_app_apps_api import (  # noqa: PLC0415
            _deploy_via_apps_api,
        )

        return await _deploy_via_apps_api(artifact, config, deployment, workspace_client)

    async def deactivate(
        self,
        deployment: AgentDeployment,
        *,
        client_resolver: WorkspaceClientResolver | None = None,
    ) -> None:
        """Tear down the live Databricks App and uploaded workspace files.

        Idempotent: 404/NotFound from the SDK is treated as success (the resource
        is already gone). Any other upstream failure is raised as
        ``DeploymentCleanupError`` so the API layer can escalate the row to
        ``cleanup_failed`` after ``MAX_CLEANUP_ATTEMPTS``.

        When ``client_resolver`` is supplied (user-initiated DELETE), the
        Apps / workspace SDK calls run with the user's OBO-scoped client so
        that resources the user originally created (via OBO at deploy time)
        can be deleted by the same identity. With a ``None`` resolver
        (janitor / orphan-detection), falls back to the parent-app SP via
        ``get_databricks_auth().get_client()`` — same behaviour as before.
        """
        external = deployment.external_resource_ids or {}
        app_name: str | None = external.get("app_name")
        deployment_path: str | None = external.get("deployment_path")

        if not app_name and not deployment_path:
            # Deployment was never live — nothing to tear down.
            return

        from deep_research.core.databricks_auth import (  # noqa: PLC0415
            get_databricks_auth,
        )

        sp_client = get_databricks_auth().get_client()
        if client_resolver is not None:
            user_client = client_resolver.resolve(
                purpose="shell_app.deactivate",
                deployment_id=deployment.id,
            )
        else:
            user_client = sp_client

        failures: list[tuple[str, Exception]] = []

        # --- Delete the Databricks App (App is OBO-owned: use user_client) ---
        if app_name:
            try:
                await asyncio.to_thread(user_client.apps.delete, app_name)
                logger.info(
                    "SHELL_APP_DEACTIVATE_APP_DELETED app_name=%s", app_name
                )
            except Exception as exc:  # noqa: BLE001
                if _is_not_found_error(exc):
                    logger.info(
                        "SHELL_APP_DEACTIVATE_APP_ALREADY_GONE app_name=%s", app_name
                    )
                else:
                    logger.exception(
                        "SHELL_APP_DEACTIVATE_APP_DELETE_FAILED app_name=%s", app_name
                    )
                    failures.append(("apps.delete", exc))

        # --- Delete the workspace source tree ---
        # Files are SP-uploaded today (shell_app_apps_api.py:310 hard-codes SP for
        # the upload), so the OBO identity often lacks delete permission on them.
        # Try the user_client first (correct for OBO-uploaded files in a future
        # consistent-identity world), then fall back to sp_client on
        # PermissionDenied (correct for today's SP-uploaded files). If BOTH
        # identities are denied, raise DeploymentCleanupExhaustedError so the
        # service layer marks cleanup_failed once and proceeds to delete the
        # parent agent — the user gets a single decisive force-delete instead
        # of three identical deterministic failures.
        if deployment_path:
            from deep_research.services.deployment.shell_app_apps_api import (  # noqa: PLC0415
                delete_workspace_source_tree,
            )

            try:
                await delete_workspace_source_tree(user_client, deployment_path)
                logger.info(
                    "SHELL_APP_DEACTIVATE_WS_DELETED path=%s actor=user",
                    deployment_path,
                )
            except Exception as exc:  # noqa: BLE001
                if _is_not_found_error(exc):
                    logger.info(
                        "SHELL_APP_DEACTIVATE_WS_ALREADY_GONE path=%s",
                        deployment_path,
                    )
                elif (
                    _is_permission_denied_error(exc)
                    and user_client is not sp_client
                ):
                    logger.warning(
                        "SHELL_APP_DEACTIVATE_WS_OBO_DENIED_FALLBACK_SP "
                        "path=%s deployment_id=%s",
                        deployment_path,
                        deployment.id,
                    )
                    try:
                        await delete_workspace_source_tree(
                            sp_client, deployment_path
                        )
                        logger.info(
                            "SHELL_APP_DEACTIVATE_WS_DELETED path=%s actor=sp",
                            deployment_path,
                        )
                    except Exception as exc2:  # noqa: BLE001
                        if _is_not_found_error(exc2):
                            logger.info(
                                "SHELL_APP_DEACTIVATE_WS_ALREADY_GONE "
                                "path=%s actor=sp",
                                deployment_path,
                            )
                        elif _is_permission_denied_error(exc2):
                            logger.error(
                                "SHELL_APP_DEACTIVATE_WS_PERMDENIED_BOTH_IDENTITIES "
                                "path=%s deployment_id=%s "
                                "obo_error=%s sp_error=%s",
                                deployment_path,
                                deployment.id,
                                type(exc).__name__,
                                type(exc2).__name__,
                            )
                            raise DeploymentCleanupExhaustedError(
                                "Shell-app deactivate: workspace.delete denied "
                                f"by both OBO and SP. path={deployment_path}",
                                resource="workspace.delete",
                                upstream_error_type=type(exc2).__name__,
                            ) from exc2
                        else:
                            logger.exception(
                                "SHELL_APP_DEACTIVATE_WS_DELETE_FAILED "
                                "path=%s actor=sp",
                                deployment_path,
                            )
                            failures.append(("workspace.delete", exc2))
                else:
                    logger.exception(
                        "SHELL_APP_DEACTIVATE_WS_DELETE_FAILED path=%s actor=user",
                        deployment_path,
                    )
                    failures.append(("workspace.delete", exc))

        if failures:
            detail = ", ".join(
                f"{resource} raised {type(exc).__name__}"
                for resource, exc in failures
            )
            raise DeploymentCleanupError(
                f"Shell-app deactivate failed: {detail}",
                resource=", ".join(resource for resource, _ in failures),
                upstream_error_type=type(failures[0][1]).__name__,
            )
