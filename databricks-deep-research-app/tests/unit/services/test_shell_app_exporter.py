"""Unit tests for ShellAppExporter (US-210).

Plan reference: agent-designer-deployment.md Section E (Shell-app).

Verifies the zip artifact contents (8 files), Jinja substitution of the
immutable Git tag, custom-tool rejection, and the Phase 2-B stub deploy()
recording the SHA256 + app_name in external_resource_ids.
"""
from __future__ import annotations

import io
import zipfile
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from deep_research.models.agent_deployment import AgentDeployment, DeploymentMode
from deep_research.services.deployment import (
    Artifact,
    DeploymentResult,
    DeploymentTranslator,
    ShellAppExporter,
    ValidationResult,
)


def _valid_config(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "mode": "shell_app",
        "app_name": "dr-shell-research",
        "framework_git_tag": "v0.3.0",
        "target": "dev",
    }
    base.update(overrides)
    return base


def _agent_revision(
    *,
    custom_tool: bool = False,
    web_search: bool = False,
    table_tools: bool = False,
) -> tuple[MagicMock, MagicMock]:
    agent = MagicMock(id=uuid4(), name="Deep Research Agent")
    tools: list[dict[str, object]] = []
    if custom_tool:
        tools.append({"name": "mytool", "kind": "custom", "config": {}})
    if web_search:
        tools.append({"name": "web_search", "kind": "web_search", "config": {}})
    if table_tools:
        tools.extend(
            [
                {
                    "name": "table_read",
                    "kind": "table_read",
                    "config": {"table_name": "main.officeqa_benchmark.treasury_chunks"},
                },
                {
                    "name": "table_load",
                    "kind": "table_load",
                    "config": {"table_name": "main.officeqa_benchmark.treasury_chunks"},
                },
            ]
        )
    revision = MagicMock(
        rev_id=uuid4(),
        definition={
            "name": "deep-research",
            "version": 1,
            "tools": tools,
            "root": {"type": "sequence", "children": []},
        },
    )
    return agent, revision


class TestProtocolConformance:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(ShellAppExporter(), DeploymentTranslator)

    def test_mode_classvar(self) -> None:
        assert ShellAppExporter.mode == DeploymentMode.SHELL_APP


class TestValidate:
    @pytest.mark.asyncio
    async def test_valid_when_all_required_fields_present(self) -> None:
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        result = await translator.validate(agent, revision, _valid_config())
        assert isinstance(result, ValidationResult)
        assert result.valid is True
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_invalid_when_app_name_missing_prefix(self) -> None:
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        result = await translator.validate(
            agent, revision, _valid_config(app_name="my-app")
        )
        assert result.valid is False
        assert any("app_name" in e.message for e in result.errors)

    @pytest.mark.asyncio
    async def test_framework_git_tag_now_optional(self) -> None:
        """framework_git_tag is no longer required — the framework ships as a
        bundled wheel (see plan imperative-wishing-lynx.md). An absent/empty
        value must NOT block validation."""
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        result = await translator.validate(
            agent, revision, _valid_config(framework_git_tag="")
        )
        # The only acceptable failure here is wheel-not-found (CI without a
        # built framework wheel). framework_git_tag itself must not contribute.
        git_tag_errors = [e for e in result.errors if "framework_git_tag" in e.message]
        assert git_tag_errors == [], (
            f"empty framework_git_tag was rejected; messages: "
            f"{[e.message for e in result.errors]}"
        )

    # ----------------------------------------------------------------------
    # Phase 3 C1 — framework_git_tag whitelist regex. Rendered verbatim into
    # the shell-app's pyproject.toml pip URL; rogue characters could fragment
    # the URL or alter pip's parsing. Reject anything outside the
    # alphanumerics + ``.``, ``_``, ``-``, ``/`` set.
    # ----------------------------------------------------------------------

    @pytest.mark.parametrize(
        "tag",
        [
            "main@evil",          # @ — pip URL revision/auth separator
            "v1.0#evil",          # # — pip URL fragment separator
            "main?evil",          # ? — query string
            "main hack",          # whitespace
            "main\nrm -rf /",     # newline + shell-like content
            "main\"; pip install evil",  # quote + semicolon
            ".hidden",            # starts with dot (git rejects)
            "-flag",              # starts with hyphen (could look like pip flag)
            "/abs/path",          # starts with slash
            "branch:with:colons", # colon
            "x" * 257,            # length overflow
        ],
    )
    @pytest.mark.asyncio
    async def test_rejects_malicious_framework_git_tag(self, tag: str) -> None:
        """C1 regression — adversarial ``framework_git_tag`` inputs must be
        rejected by the whitelist regex before the value reaches pyproject."""
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        result = await translator.validate(
            agent, revision, _valid_config(framework_git_tag=tag)
        )
        assert result.valid is False, (
            f"Expected validator to reject framework_git_tag={tag!r} but it accepted"
        )
        assert any(
            "framework_git_tag" in e.message and (
                "disallowed characters" in e.message
                or "Git ref" in e.message
            )
            for e in result.errors
        ), f"Expected whitelist rejection message; got: {[e.message for e in result.errors]}"

    @pytest.mark.parametrize(
        "tag",
        [
            "v1.2.3",
            "main",
            "develop",
            "release/2026.05.24",
            "feat_x-2",
            "0123456789abcdef0123456789abcdef01234567",  # 40-char sha
        ],
    )
    @pytest.mark.asyncio
    async def test_accepts_valid_framework_git_tag(self, tag: str) -> None:
        """C1 sanity — common, safe git-ref shapes pass the whitelist."""
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        result = await translator.validate(
            agent, revision, _valid_config(framework_git_tag=tag)
        )
        # No git-tag-related error in the result.
        git_tag_errors = [
            e for e in result.errors if "framework_git_tag" in e.message
        ]
        assert git_tag_errors == [], (
            f"Whitelist falsely rejected safe tag {tag!r}: {git_tag_errors}"
        )

    @pytest.mark.asyncio
    async def test_rejects_custom_tool_kind(self) -> None:
        translator = ShellAppExporter()
        agent, revision = _agent_revision(custom_tool=True)
        result = await translator.validate(agent, revision, _valid_config())
        assert result.valid is False
        assert any(
            "custom tools are not supported" in e.message.lower()
            for e in result.errors
        )

    @pytest.mark.asyncio
    async def test_rejects_table_tools_without_storage_warehouse_id(self) -> None:
        translator = ShellAppExporter()
        agent, revision = _agent_revision(table_tools=True)
        with patch(
            "deep_research.services.deployment.shell_app._resolve_storage_warehouse_id",
            return_value=None,
        ):
            result = await translator.validate(agent, revision, _valid_config())

        assert result.valid is False
        assert any("SQL Warehouse id" in e.message for e in result.errors)

    @pytest.mark.asyncio
    async def test_web_search_uses_default_brave_secret_config(self) -> None:
        translator = ShellAppExporter()
        agent, revision = _agent_revision(web_search=True)
        result = await translator.validate(agent, revision, _valid_config())
        assert result.valid is True


class TestTranslate:
    @pytest.mark.asyncio
    async def test_artifact_payload_is_bytes(self) -> None:
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        artifact = await translator.translate(agent, revision, _valid_config())
        assert isinstance(artifact, Artifact)
        assert isinstance(artifact.payload, bytes)
        assert artifact.mode == DeploymentMode.SHELL_APP

    @pytest.mark.asyncio
    async def test_zip_contains_all_8_template_files(self) -> None:
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        artifact = await translator.translate(agent, revision, _valid_config())
        with zipfile.ZipFile(io.BytesIO(artifact.payload)) as zf:
            names = set(zf.namelist())
        expected = {
            "app.py",
            "app.yaml",
            "databricks.yml",
            "pyproject.toml",
            "entrypoint.sh",
            "static/index.html",
            "agent.yaml",
            "README.md",
        }
        assert expected.issubset(names), f"missing: {expected - names}"

    @pytest.mark.asyncio
    async def test_pyproject_references_bundled_wheel(self) -> None:
        """The generated pyproject.toml MUST point at the bundled wheel via
        [tool.uv.sources] and MUST NOT contain a git+https URL — the deployed
        shell app installs the framework from a local file at app startup.
        Plan: imperative-wishing-lynx.md."""
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        artifact = await translator.translate(agent, revision, _valid_config())
        with zipfile.ZipFile(io.BytesIO(artifact.payload)) as zf:
            pyproject = zf.read("pyproject.toml").decode("utf-8")
            wheel_entries = [
                n for n in zf.namelist()
                if n.startswith("wheels/databricks_deep_research-") and n.endswith(".whl")
            ]
        assert "git+https://" not in pyproject, (
            "shell-app pyproject must not contain a git URL after the "
            "wheel-bundling refactor"
        )
        assert "[tool.uv.sources]" in pyproject
        assert 'databricks-deep-research = { path = "wheels/' in pyproject
        assert len(wheel_entries) == 1, (
            f"shell-app zip must bundle exactly one framework wheel under wheels/; "
            f"got {wheel_entries}"
        )
        # The path declared in pyproject must match the actual zip entry.
        wheel_filename = wheel_entries[0].removeprefix("wheels/")
        assert f'path = "wheels/{wheel_filename}"' in pyproject

    @pytest.mark.asyncio
    async def test_pyproject_allows_direct_git_dependency_reference(self) -> None:
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        artifact = await translator.translate(agent, revision, _valid_config())
        with zipfile.ZipFile(io.BytesIO(artifact.payload)) as zf:
            pyproject = zf.read("pyproject.toml").decode("utf-8")

        assert "[tool.hatch.metadata]" in pyproject
        assert "allow-direct-references = true" in pyproject

    @pytest.mark.asyncio
    async def test_pyproject_disables_self_packaging_for_uv_run(self) -> None:
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        artifact = await translator.translate(agent, revision, _valid_config())
        with zipfile.ZipFile(io.BytesIO(artifact.payload)) as zf:
            pyproject = zf.read("pyproject.toml").decode("utf-8")

        assert "[tool.uv]" in pyproject
        assert "package = false" in pyproject
        assert "[tool.hatch.build.targets.wheel]" in pyproject
        assert '"app.py"' in pyproject
        assert '"agent.yaml"' in pyproject
        assert '"static"' in pyproject

    @pytest.mark.asyncio
    async def test_databricks_yml_references_app_name(self) -> None:
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        artifact = await translator.translate(
            agent, revision, _valid_config(app_name="dr-shell-foo")
        )
        with zipfile.ZipFile(io.BytesIO(artifact.payload)) as zf:
            databricks_yml = zf.read("databricks.yml").decode("utf-8")
        assert "dr-shell-foo" in databricks_yml

    @pytest.mark.asyncio
    async def test_databricks_yml_includes_all_obo_scopes(self) -> None:
        """Bundle template must declare every OBO scope the framework's
        Databricks-bound tools can require.

        Regression test for the shell-app OBO bug: previously the template
        listed only ``vectorsearch.vector-search-endpoints`` so vector-index
        queries (which call ``query_index`` against the index path, not the
        endpoint) failed with permission errors that the SDK surfaced as
        an empty result. Keep this list in sync with the main DRE bundle
        at databricks-deep-research-app/databricks.yml:124-130 and with
        the inline-deploy constant ``_APP_USER_API_SCOPES`` in
        deep_research/services/deployment/shell_app_apps_api.py.
        """
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        artifact = await translator.translate(agent, revision, _valid_config())
        with zipfile.ZipFile(io.BytesIO(artifact.payload)) as zf:
            databricks_yml = zf.read("databricks.yml").decode("utf-8")

        for scope in (
            "sql",
            "serving.serving-endpoints",
            "vectorsearch.vector-search-endpoints",
            "vectorsearch.vector-search-indexes",
            "dashboards.genie",
        ):
            assert scope in databricks_yml, (
                f"scope {scope!r} missing from rendered databricks.yml; "
                f"OBO will silently fall back to app SP for that surface"
            )

    @pytest.mark.asyncio
    async def test_app_py_builds_tool_factory_context(self) -> None:
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        artifact = await translator.translate(agent, revision, _valid_config())
        with zipfile.ZipFile(io.BytesIO(artifact.payload)) as zf:
            app_py = zf.read("app.py").decode("utf-8")

        assert "ToolFactoryContext.from_defaults" in app_py
        assert "_wire_text_table_context(ctx)" in app_py
        assert "wire_statement_execution_text_table_context" in app_py
        assert "class _StatementExecutionTableSQL" not in app_py
        assert "TableBindingRegistry" not in app_py
        assert "SchemaCache" not in app_py
        assert "STORAGE_WAREHOUSE_ID" in app_py
        assert "FrameworkLLMClient.from_databricks()" in app_py
        # OBO threading: the shell app must build a per-request runner whose
        # WorkspaceClient carries the caller's x-forwarded-access-token. A
        # singleton _runner shared across users blocks UC-gated tools
        # (vector_search, Genie, ACL'd endpoints) from ever reaching the
        # caller's permissions — see plans/proud-scribbling-alpaca.md.
        assert "_build_per_request_runner" in app_py
        assert "x-forwarded-access-token" in app_py
        assert 'auth_type="pat"' in app_py
        assert "obo_token_present" in app_py
        # Fail-closed gate (codex HIGH): missing OBO under Databricks Apps
        # runtime with a Databricks-bound workflow must yield a structured
        # SSE error event with code 'missing_obo_token' instead of falling
        # back to the app SP.
        assert "missing_obo_token" in app_py
        assert "_IS_DATABRICKS_APPS_RUNTIME" in app_py
        assert "_WORKFLOW_REQUIRES_DATABRICKS" in app_py
        # strict_tool_resolution (codex HIGH): factory misconfig (missing
        # index_name, dead workspace_client) must raise WorkflowError
        # instead of being swallowed. Phase-3-era ``strict_tool_resolution``
        # was hard-coded; the kwarg is now feature-detected via
        # ``inspect.signature`` so the template is forward/back-compatible
        # with framework versions that lack the kwarg. Both the gating
        # constant and the dict assignment must be present.
        assert "_SUPPORTS_STRICT_TOOL_RESOLUTION" in app_py
        assert 'stream_kwargs["strict_tool_resolution"] = True' in app_py
        # Tool-failure logging (codex MEDIUM): ToolResult(success=False)
        # surfaces in logs to disambiguate auth failures from genuinely
        # empty queries.
        assert "SHELL_APP_TOOL_FAILURE" in app_py
        assert "ToolResultEvent" in app_py
        assert "logging.basicConfig" in app_py
        assert "SHELL_APP_STREAM_EVENT" in app_py
        assert "SHELL_APP_INDEX_SERVED" in app_py
        assert "MLFLOW_ENABLED" in app_py
        assert "SHELL_APP_MLFLOW_TRACING_DISABLED" in app_py
        assert "ping=_SSE_HEARTBEAT_SECONDS" in app_py
        assert "SHELL_APP_CHAT_CLIENT_DISCONNECTED" in app_py
        assert "planner_guidance_present" in app_py
        assert '_SHELL_APP_TEMPLATE_VERSION = "2026-05-28.1"' in app_py
        assert "Cache-Control" in app_py
        assert "X-Shell-App-Template-Version" in app_py

    @pytest.mark.asyncio
    async def test_static_chat_ui_parses_crlf_sse_frames(self) -> None:
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        artifact = await translator.translate(agent, revision, _valid_config())
        with zipfile.ZipFile(io.BytesIO(artifact.payload)) as zf:
            html = zf.read("static/index.html").decode("utf-8")

        assert "replace(/\\r\\n/g, '\\n').replace(/\\r/g, '\\n')" in html
        assert "buffer.split(/\\n\\n+/)" in html
        assert "const dataText = dataLines.join('\\n')" in html
        assert "drainFrames(true)" in html
        assert "X-Shell-App-Template-Version" in html
        assert "2026-05-28.1" in html
        assert "[shell-app] stream frame" in html
        assert html.count("let buffer = '';") == 1
        assert html.index("let buffer = '';") < html.index("function drainFrames")
        assert "function renderMarkdown" in html
        assert "function appendActivity" in html
        assert "function renderFinalAnswer" in html
        assert "activityPanelEl.open = false" in html
        assert "markdown-body" in html
        assert "latestFinalReport" in html
        assert "renderFinalAnswer(payload.output)" in html
        assert "appendActivity(eventName, payload)" in html

    @pytest.mark.asyncio
    async def test_web_search_bundle_binds_brave_secret(self) -> None:
        translator = ShellAppExporter()
        agent, revision = _agent_revision(web_search=True)
        artifact = await translator.translate(agent, revision, _valid_config())
        with zipfile.ZipFile(io.BytesIO(artifact.payload)) as zf:
            databricks_yml = zf.read("databricks.yml").decode("utf-8")
            app_yaml = zf.read("app.yaml").decode("utf-8")

        assert "BRAVE_API_KEY" in databricks_yml
        assert "value_from: 'brave-api-key'" in databricks_yml
        assert "secret:" in databricks_yml
        assert "scope: 'deep-research-secrets'" in databricks_yml
        assert "key: 'BRAVE_API_KEY'" in databricks_yml
        assert "BRAVE_API_KEY" in app_yaml
        assert "valueFrom: 'brave-api-key'" in app_yaml
        assert artifact.metadata["requires_web_search"] == "true"

    @pytest.mark.asyncio
    async def test_non_web_search_bundle_does_not_bind_brave_secret(self) -> None:
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        artifact = await translator.translate(agent, revision, _valid_config())
        with zipfile.ZipFile(io.BytesIO(artifact.payload)) as zf:
            databricks_yml = zf.read("databricks.yml").decode("utf-8")
            app_yaml = zf.read("app.yaml").decode("utf-8")

        assert "BRAVE_API_KEY" not in databricks_yml
        assert "BRAVE_API_KEY" not in app_yaml
        assert artifact.metadata["requires_web_search"] == "false"

    @pytest.mark.asyncio
    async def test_table_tool_bundle_binds_storage_warehouse(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("TABLE_TOOLS_WAREHOUSE_ID", "d837825f69a03500")
        translator = ShellAppExporter()
        agent, revision = _agent_revision(table_tools=True)
        artifact = await translator.translate(agent, revision, _valid_config())
        with zipfile.ZipFile(io.BytesIO(artifact.payload)) as zf:
            databricks_yml = zf.read("databricks.yml").decode("utf-8")
            app_yaml = zf.read("app.yaml").decode("utf-8")
            agent_yaml = zf.read("agent.yaml").decode("utf-8")

        assert "STORAGE_WAREHOUSE_ID" in databricks_yml
        assert "value: 'd837825f69a03500'" in databricks_yml
        assert "sql_warehouse:" in databricks_yml
        assert "id: 'd837825f69a03500'" in databricks_yml
        assert "permission: CAN_USE" in databricks_yml
        assert "STORAGE_WAREHOUSE_ID" in app_yaml
        assert "value: 'd837825f69a03500'" in app_yaml
        assert "table_read" in agent_yaml
        assert "main.officeqa_benchmark.treasury_chunks" in agent_yaml
        assert artifact.metadata["requires_sql_warehouse"] == "true"
        assert artifact.metadata["storage_warehouse_id_configured"] == "true"

    @pytest.mark.asyncio
    async def test_workflow_name_alone_does_not_bind_brave_secret(self) -> None:
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        revision.definition = {
            "name": "web_search",
            "version": 1,
            "tools": [],
            "root": {"type": "sequence", "children": []},
        }
        artifact = await translator.translate(agent, revision, _valid_config())

        assert artifact.metadata["requires_web_search"] == "false"

    @pytest.mark.asyncio
    async def test_agent_yaml_contains_definition_payload(self) -> None:
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        # Inject a unique marker into the definition we can grep for.
        revision.definition = {
            "name": "deep-research",
            "version": 1,
            "marker_unique_str": "MARKER-SENTINEL-12345",
            "tools": [],
            "root": {"type": "sequence", "children": []},
        }
        artifact = await translator.translate(agent, revision, _valid_config())
        with zipfile.ZipFile(io.BytesIO(artifact.payload)) as zf:
            agent_yaml = zf.read("agent.yaml").decode("utf-8")
        assert "MARKER-SENTINEL-12345" in agent_yaml

    @pytest.mark.asyncio
    async def test_artifact_metadata(self) -> None:
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        artifact = await translator.translate(
            agent, revision, _valid_config(app_name="dr-shell-meta")
        )
        assert artifact.metadata["app_name"] == "dr-shell-meta"
        # framework_git_tag is no longer authoritative — the wheel is the
        # ground truth. Verify the new fields exist and reflect the actual
        # bundled wheel. Plan: imperative-wishing-lynx.md.
        wheel_filename = artifact.metadata["framework_wheel_filename"]
        assert wheel_filename.startswith("databricks_deep_research-")
        assert wheel_filename.endswith("-py3-none-any.whl")
        assert artifact.metadata["framework_wheel_version"] != "unknown"
        assert len(artifact.metadata["sha256"]) == 64

    @pytest.mark.asyncio
    async def test_entrypoint_sh_is_executable_in_zip(self) -> None:
        """T1 fix: entrypoint.sh in the generated zip must carry mode 0o755
        so ``apps.create``'s upload preserves the +x bit. The bit lives in
        the high 16 of ZipInfo.external_attr."""
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        artifact = await translator.translate(agent, revision, _valid_config())
        with zipfile.ZipFile(io.BytesIO(artifact.payload)) as zf:
            entry = zf.getinfo("entrypoint.sh")
            # high 16 bits of external_attr encode the unix mode.
            mode = entry.external_attr >> 16
            assert mode == 0o755, f"entrypoint.sh has mode {mode:o}, expected 0o755"
            # Regular files should NOT have the executable bit.
            app_py = zf.getinfo("app.py")
            assert app_py.external_attr >> 16 == 0o644

    @pytest.mark.asyncio
    async def test_translate_is_byte_deterministic(self) -> None:
        """W7: regeneration from the same inputs must produce the same bytes.

        The /export-zip route compares the regenerated SHA256 against the
        digest captured at deploy time. If translate() were non-deterministic
        (e.g. zip headers used wall-clock timestamps), every call would
        mismatch — making the integrity check useless. We pin all zip entry
        timestamps to the zip epoch (1980-01-01) so this round-trip is stable.
        """
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        cfg = _valid_config()
        first = await translator.translate(agent, revision, cfg)
        second = await translator.translate(agent, revision, cfg)
        assert isinstance(first.payload, bytes)
        assert isinstance(second.payload, bytes)
        assert first.payload == second.payload, (
            "shell-app zip regeneration must be byte-deterministic; if this "
            "fails, the W7 integrity check in /export-zip is broken."
        )
        assert first.metadata["sha256"] == second.metadata["sha256"]

    @pytest.mark.asyncio
    async def test_translate_differs_when_inputs_differ(self) -> None:
        """Sanity-check the determinism test: different inputs MUST differ."""
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        a = await translator.translate(
            agent, revision, _valid_config(app_name="dr-shell-aaa")
        )
        b = await translator.translate(
            agent, revision, _valid_config(app_name="dr-shell-bbb")
        )
        assert a.payload != b.payload
        assert a.metadata["sha256"] != b.metadata["sha256"]


class TestDeploy:
    @pytest.mark.asyncio
    async def test_deploy_records_sha_and_app_name(self) -> None:
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        artifact = await translator.translate(agent, revision, _valid_config())
        deployment = MagicMock(spec=AgentDeployment)
        result = await translator.deploy(artifact, _valid_config(), deployment)
        assert isinstance(result, DeploymentResult)
        assert result.success is True
        assert result.endpoint_name == "dr-shell-research"
        assert "shell_app_zip_sha256" in result.external_resource_ids
        assert len(result.external_resource_ids["shell_app_zip_sha256"]) == 64
        assert result.external_resource_ids["app_name"] == "dr-shell-research"

    @pytest.mark.asyncio
    async def test_deploy_fails_on_non_bytes_payload(self) -> None:
        translator = ShellAppExporter()
        deployment = MagicMock(spec=AgentDeployment)
        broken = Artifact(mode=DeploymentMode.SHELL_APP, payload={"not": "bytes"})
        result = await translator.deploy(broken, _valid_config(), deployment)
        assert result.success is False
        assert result.error_message is not None


class TestTemplateOutputS1:
    """Section S1: assert that the rendered templates no longer contain the
    MLFLOW_EXPERIMENT_ID binding or the 'experiment' resource block."""

    @pytest.mark.asyncio
    async def test_render_app_yaml_no_experiment_env(self) -> None:
        """app.yaml must NOT contain MLFLOW_EXPERIMENT_ID after S1 fix."""
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        artifact = await translator.translate(agent, revision, _valid_config())
        with zipfile.ZipFile(io.BytesIO(artifact.payload)) as zf:
            rendered_app_yaml = zf.read("app.yaml").decode("utf-8")
        assert "MLFLOW_EXPERIMENT_ID" not in rendered_app_yaml, (
            "app.yaml must not contain MLFLOW_EXPERIMENT_ID after S1 fix"
        )
        # Check non-comment lines only — comments may explain the valueFrom syntax
        non_comment_lines = [
            line for line in rendered_app_yaml.splitlines()
            if not line.lstrip().startswith("#")
        ]
        non_comment_text = "\n".join(non_comment_lines)
        assert "valueFrom" not in non_comment_text, (
            "app.yaml must not contain valueFrom in non-comment lines after S1 fix"
        )

    @pytest.mark.asyncio
    async def test_render_databricks_yml_no_experiment_resource(self) -> None:
        """databricks.yml must NOT contain the 'experiment' resource block."""
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        artifact = await translator.translate(agent, revision, _valid_config())
        with zipfile.ZipFile(io.BytesIO(artifact.payload)) as zf:
            rendered_databricks_yml = zf.read("databricks.yml").decode("utf-8")
        assert "name: 'experiment'" not in rendered_databricks_yml, (
            "databricks.yml must not contain the 'experiment' resource block after S1 fix"
        )

    @pytest.mark.asyncio
    async def test_render_app_yaml_still_has_mlflow_tracking_uri(self) -> None:
        """MLFLOW_TRACKING_URI must still be present in app.yaml."""
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        artifact = await translator.translate(agent, revision, _valid_config())
        with zipfile.ZipFile(io.BytesIO(artifact.payload)) as zf:
            rendered_app_yaml = zf.read("app.yaml").decode("utf-8")
        assert "MLFLOW_TRACKING_URI" in rendered_app_yaml

    @pytest.mark.asyncio
    async def test_render_app_yaml_disables_mlflow_by_default(self) -> None:
        """Shell apps must not enable MLflow tracing unless explicitly opted in.

        Regression target: trace artifact upload retries can otherwise starve
        long SSE responses behind the Databricks Apps proxy.
        """
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        artifact = await translator.translate(agent, revision, _valid_config())
        with zipfile.ZipFile(io.BytesIO(artifact.payload)) as zf:
            rendered_app_yaml = zf.read("app.yaml").decode("utf-8")
        assert "MLFLOW_ENABLED" in rendered_app_yaml
        assert "value: 'false'" in rendered_app_yaml
        assert "SHELL_APP_SSE_HEARTBEAT_SECONDS" in rendered_app_yaml

    @pytest.mark.asyncio
    async def test_render_app_yaml_declares_shared_experiment(self) -> None:
        """Every deployed shell-app must trace into the shared experiment.

        Tracing-unification plan A.1: single canonical experiment so designer-
        chat / main-chat / shell-app traces all live in one queryable surface.
        """
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        artifact = await translator.translate(agent, revision, _valid_config())
        with zipfile.ZipFile(io.BytesIO(artifact.payload)) as zf:
            rendered_app_yaml = zf.read("app.yaml").decode("utf-8")
        assert "MLFLOW_EXPERIMENT_NAME" in rendered_app_yaml
        assert "/Shared/deep-research-agent-experiments" in rendered_app_yaml

    @pytest.mark.asyncio
    async def test_render_app_yaml_carries_dr_provenance_env_vars(self) -> None:
        """The 4 DR_* env vars must be Jinja-interpolated from the exporter
        context so the deployed shell-app's traces self-identify by app
        name, agent_v2_id, agent name, and revision id.
        """
        translator = ShellAppExporter()
        agent, revision = _agent_revision()
        # _agent_revision uses MagicMock for .name; pin to a real string so
        # the template render (and our in-text assertion) sees concrete text.
        agent.name = "MyDeepResearchAgent"
        artifact = await translator.translate(agent, revision, _valid_config())
        with zipfile.ZipFile(io.BytesIO(artifact.payload)) as zf:
            rendered_app_yaml = zf.read("app.yaml").decode("utf-8")
        # Env var names present.
        assert "DR_APP_NAME" in rendered_app_yaml
        assert "DR_AGENT_V2_ID" in rendered_app_yaml
        assert "DR_AGENT_NAME" in rendered_app_yaml
        assert "DR_REVISION_ID" in rendered_app_yaml
        # Values interpolated (not left as raw Jinja).
        assert "{{" not in rendered_app_yaml, (
            "app.yaml still contains unrendered Jinja placeholders"
        )
        assert "dr-shell-research" in rendered_app_yaml  # app_name from _valid_config
        assert str(agent.id) in rendered_app_yaml  # DR_AGENT_V2_ID
        assert "MyDeepResearchAgent" in rendered_app_yaml  # DR_AGENT_NAME
        assert str(revision.rev_id) in rendered_app_yaml  # DR_REVISION_ID


class TestDeactivate:
    @pytest.mark.asyncio
    async def test_deactivate_no_external_resources_is_early_return(self) -> None:
        """When external_resource_ids is missing app_name, deactivate returns
        early without calling any SDK methods (the deployment was never live).

        The full live-path coverage lives in test_shell_app_deploy_inline.py.
        """
        translator = ShellAppExporter()
        deployment = MagicMock(spec=AgentDeployment)
        deployment.external_resource_ids = None
        result = await translator.deactivate(deployment)
        assert result is None
        # Idempotency: second call against the same empty-state row is also noop
        result2 = await translator.deactivate(deployment)
        assert result2 is None
