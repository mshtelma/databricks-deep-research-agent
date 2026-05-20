"""ShellAppExporter: Mode 2 — standalone Databricks App with chat UI.

Generates a downloadable zip from the 8-file template under
``templates/agent-shell-app/`` (Jinja-rendered .j2 files + verbatim copies of
.py / .yaml / .sh / .html / .md). The agent's ``WorkflowDefinition`` AST is
serialized into ``agent.yaml`` and embedded in the zip.

Plan reference: agent-designer-deployment.md Section E (Shell-app), with
GitHub-pinning instead of PyPI per user override (Section C.1, plan tag
``git+https://github.com/mshtelma/databricks-deep-research-agent.git@<tag>``).

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
import zipfile
from pathlib import Path
from typing import Any, ClassVar

import yaml
from jinja2 import StrictUndefined, Template

from deep_research.models.agent_deployment import AgentDeployment, DeploymentMode
from deep_research.services.deployment._paths import resolve_package_data_dir
from deep_research.services.deployment.translator import (
    Artifact,
    DeploymentCleanupError,
    DeploymentResult,
    ValidationError,
    ValidationResult,
)

logger = logging.getLogger(__name__)
_DEFAULT_BRAVE_SECRET_SCOPE = "deep-research-secrets"
_DEFAULT_BRAVE_SECRET_KEY = "BRAVE_API_KEY"


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

        git_tag = config.get("framework_git_tag", "")
        if not isinstance(git_tag, str) or not git_tag.strip():
            errors.append(
                ValidationError(
                    message="framework_git_tag is required (Git ref)",
                    path="config.framework_git_tag",
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
        git_tag: str = config["framework_git_tag"]
        target: str = config.get("target", "dev")
        definition = revision.definition or {}
        uses_web_search = _definition_uses_web_search(definition)
        brave_secret_scope, brave_secret_key = _resolve_brave_secret_config(
            config,
            include_defaults=uses_web_search,
        )

        # Serialize the workflow definition into a YAML string that we splice
        # into agent.yaml.j2 below. ``default_flow_style=False`` keeps it
        # human-readable (block style).
        definition_yaml = yaml.safe_dump(
            definition, default_flow_style=False, sort_keys=False
        )

        context = {
            "app_name": app_name,
            "git_tag": git_tag,
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

        payload = buf.getvalue()
        digest = hashlib.sha256(payload).hexdigest()

        return Artifact(
            mode=DeploymentMode.SHELL_APP,
            payload=payload,
            metadata={
                "app_name": app_name,
                "framework_git_tag": git_tag,
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
                "framework_git_tag": config["framework_git_tag"],
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

    async def deactivate(self, deployment: AgentDeployment) -> None:
        """Tear down the live Databricks App and uploaded workspace files.

        Idempotent: 404/NotFound from the SDK is treated as success (the resource
        is already gone). Any other upstream failure is raised as
        ``DeploymentCleanupError`` so the API layer can escalate the row to
        ``cleanup_failed`` after ``MAX_CLEANUP_ATTEMPTS``.
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

        client = get_databricks_auth().get_client()

        failures: list[tuple[str, Exception]] = []

        # --- Delete the Databricks App ---
        if app_name:
            try:
                await asyncio.to_thread(client.apps.delete, app_name)
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
        if deployment_path:
            try:
                from deep_research.services.deployment.shell_app_apps_api import (  # noqa: PLC0415
                    delete_workspace_source_tree,
                )

                await delete_workspace_source_tree(client, deployment_path)
                logger.info(
                    "SHELL_APP_DEACTIVATE_WS_DELETED path=%s", deployment_path
                )
            except Exception as exc:  # noqa: BLE001
                if _is_not_found_error(exc):
                    logger.info(
                        "SHELL_APP_DEACTIVATE_WS_ALREADY_GONE path=%s",
                        deployment_path,
                    )
                else:
                    logger.exception(
                        "SHELL_APP_DEACTIVATE_WS_DELETE_FAILED path=%s",
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
