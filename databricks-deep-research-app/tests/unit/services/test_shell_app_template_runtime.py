"""Runtime tests for the rendered shell-app template's /api/chat handler.

These tests load the rendered ``app.py`` from a Shell-App artifact, patch
out external dependencies (LLM client, MLflow tracing, WorkspaceClient
default auth-detect), and exercise the FastAPI app in-process to prove
the OBO fail-closed gate behaves correctly.

Regression target: the original shell-app template silently fell back to
the app service principal when ``x-forwarded-access-token`` was absent,
which caused UC-gated vector_search / Genie / serving-endpoint queries
to return empty without any visible error. The fail-closed gate added in
template version 2026-05-25.1 is what these tests guard.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from deep_research.services.deployment import ShellAppExporter


def _agent_revision_with_tools(tool_kinds: list[str]) -> tuple[MagicMock, MagicMock]:
    """Build a fake (agent, revision) whose workflow declares the given tool kinds.

    The synthetic workflow body is intentionally empty — the test stubs out
    ``load_workflow_from_dict`` so it doesn't have to be schema-valid. The
    OBO fail-closed gate inspects ``_DEFINITION_DICT["tools"]`` (the raw
    declared tool kinds), not the loaded ``WorkflowDefinition``.
    """
    agent = MagicMock(id=uuid4(), name="Deep Research Agent")
    tools: list[dict[str, Any]] = []
    for kind in tool_kinds:
        config: dict[str, Any] = {}
        if kind == "vector_search":
            config["index_name"] = "main.test.idx"
        tools.append({"name": kind, "kind": kind, "config": config})
    revision = MagicMock(
        rev_id=uuid4(),
        definition={
            "id": "shell-app-test",
            "name": "deep-research",
            "version": 1,
            "tools": tools,
            "root": {"type": "sequence", "children": []},
        },
    )
    return agent, revision


def _config() -> dict[str, object]:
    return {
        "mode": "shell_app",
        "app_name": "dr-shell-research",
        "framework_git_tag": "v0.3.0",
        "target": "dev",
    }


@pytest.fixture
def cleanup_app_module() -> Iterator[None]:
    """Remove the test-loaded ``app`` module so each test loads fresh."""
    yield
    sys.modules.pop("app", None)


async def _render_and_unpack(
    tool_kinds: list[str], dest: Path
) -> Path:
    """Render the shell-app artifact and extract it to ``dest``. Returns the
    extracted directory (which contains ``app.py`` + ``agent.yaml``)."""
    exporter = ShellAppExporter()
    agent, revision = _agent_revision_with_tools(tool_kinds)
    artifact = await exporter.translate(agent, revision, _config())
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(artifact.payload)) as zf:
        zf.extractall(dest)
    return dest


def _load_app_module(app_dir: Path) -> ModuleType:
    """Import the rendered ``app.py`` as the ``app`` module, with module-
    level side effects (auth detect, MLflow init) mocked out so the test
    runs in-process without external network calls.

    Mocks:
    - ``databricks.sdk.WorkspaceClient()`` → fake whose ``.config.host`` is
      a stable string. Per-request OBO construction (``WorkspaceClient(
      host=..., token=..., auth_type='pat')``) is also stubbed so the
      handler can build a runner without real auth.
    - ``databricks_deep_research.FrameworkLLMClient.from_databricks`` →
      MagicMock; no real LLM connection.
    - ``databricks_deep_research.tracing.setup_mlflow_tracing`` → no-op.
    - ``mlflow.start_span`` → returns a no-op context manager.
    - ``databricks_deep_research.WorkflowRunner`` → MagicMock so the test
      can assert whether the runner was constructed and called (used to
      prove the fail-closed gate short-circuits before runner build).
    """
    sys_path_entry = str(app_dir)
    sys.path.insert(0, sys_path_entry)

    fake_ws = MagicMock()
    fake_ws.config.host = "https://test.cloud.databricks.com"
    fake_runner = MagicMock()
    fake_runner.stream = MagicMock()  # not async-iterable; will fail if called
    fake_runner_class = MagicMock(return_value=fake_runner)
    no_op_span = MagicMock()
    no_op_span.__enter__ = MagicMock(return_value=no_op_span)
    no_op_span.__exit__ = MagicMock(return_value=False)

    # Stub the framework's workflow loader so the test doesn't have to
    # build a fully-valid WorkflowDefinition (with every required field
    # the schema enforces). The OBO fail-closed gate operates on
    # ``_DEFINITION_DICT`` (the raw dict), not on the loaded definition.
    stub_definition = MagicMock(id="shell-app-test", output_keys=[])

    with (
        patch("databricks.sdk.WorkspaceClient", return_value=fake_ws),
        patch(
            "databricks_deep_research.FrameworkLLMClient.from_databricks",
            return_value=MagicMock(),
        ),
        patch(
            "databricks_deep_research.tracing.setup_mlflow_tracing",
            return_value=None,
        ),
        patch("mlflow.start_span", return_value=no_op_span),
        patch(
            "databricks_deep_research.WorkflowRunner",
            new=fake_runner_class,
        ),
        patch(
            "databricks_deep_research.workflow.loader.load_workflow_from_dict",
            return_value=stub_definition,
        ),
    ):
        spec = importlib.util.spec_from_file_location(
            "app", str(app_dir / "app.py")
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["app"] = module
        spec.loader.exec_module(module)

    sys.path.remove(sys_path_entry)
    # Attach the WorkflowRunner mock so tests can introspect call count.
    module._test_runner_class = fake_runner_class  # type: ignore[attr-defined]
    return module


@pytest.mark.asyncio
async def test_fail_closed_when_obo_missing_in_apps_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_app_module: None,
) -> None:
    """Under the Databricks Apps runtime, a request to a Databricks-bound
    workflow without ``x-forwarded-access-token`` must yield an SSE error
    event with code ``missing_obo_token`` BEFORE building the runner."""
    app_dir = await _render_and_unpack(["vector_search"], tmp_path / "app1")
    monkeypatch.setenv("DATABRICKS_APP_NAME", "dr-shell-test")
    app_module = _load_app_module(app_dir)

    assert app_module._IS_DATABRICKS_APPS_RUNTIME is True
    assert app_module._WORKFLOW_REQUIRES_DATABRICKS is True

    client = TestClient(app_module.app)
    response = client.post("/api/chat", json={"query": "irrelevant"})
    assert response.status_code == 200

    # Parse the first SSE frame from the body. SSE frames are
    # ``event: <name>\ndata: <json>\n\n``.
    body = response.text
    assert "event: error" in body, f"expected error event, got: {body[:500]}"
    # Find the JSON payload after "data: " on the same frame.
    data_line = next(
        (line for line in body.splitlines() if line.startswith("data: ")),
        None,
    )
    assert data_line is not None, f"no data line in SSE body: {body[:500]}"
    payload = json.loads(data_line[len("data: "):])
    assert payload["code"] == "missing_obo_token"

    # Critical: the runner must NOT have been constructed.
    assert (
        app_module._test_runner_class.call_count == 0
    ), "runner was built despite missing OBO — fail-closed gate did not fire"


@pytest.mark.asyncio
async def test_local_dev_does_not_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_app_module: None,
) -> None:
    """Without ``DATABRICKS_APP_NAME`` (local-dev path), the fail-closed gate
    must NOT fire — notebook-style dev should still work even without OBO."""
    app_dir = await _render_and_unpack(["vector_search"], tmp_path / "app2")
    monkeypatch.delenv("DATABRICKS_APP_NAME", raising=False)
    app_module = _load_app_module(app_dir)

    assert app_module._IS_DATABRICKS_APPS_RUNTIME is False
    # The gate condition is (apps_runtime AND requires_obo AND not user_token).
    # Even with requires_obo=True, apps_runtime=False short-circuits.
    assert app_module._WORKFLOW_REQUIRES_DATABRICKS is True

    client = TestClient(app_module.app)
    # Trigger the handler; we don't care about the response body — only
    # whether the per-request runner was constructed.
    client.post("/api/chat", json={"query": "irrelevant"})

    # Runner MUST be constructed because the fail-closed gate is bypassed.
    assert app_module._test_runner_class.call_count == 1, (
        "runner should have been built in local-dev mode — gate fired "
        "incorrectly when DATABRICKS_APP_NAME was unset"
    )


@pytest.mark.asyncio
async def test_purely_web_workflow_bypasses_obo_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_app_module: None,
) -> None:
    """A workflow with no Databricks-bound tools (e.g. just web_search) must
    not be gated by OBO even under the Databricks Apps runtime — there is
    no UC-gated resource to protect."""
    app_dir = await _render_and_unpack(["web_search"], tmp_path / "app3")
    monkeypatch.setenv("DATABRICKS_APP_NAME", "dr-shell-test")
    app_module = _load_app_module(app_dir)

    assert app_module._IS_DATABRICKS_APPS_RUNTIME is True
    assert app_module._WORKFLOW_REQUIRES_DATABRICKS is False

    client = TestClient(app_module.app)
    # Trigger the handler; we don't care about the response body — only
    # whether the per-request runner was constructed.
    client.post("/api/chat", json={"query": "irrelevant"})

    # Runner MUST be constructed — web-only workflows don't need OBO.
    assert app_module._test_runner_class.call_count == 1, (
        "runner should have been built for a purely-web workflow; the "
        "fail-closed gate fired incorrectly"
    )


def test_workflow_requires_databricks_helper_unit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unit-level check of ``_workflow_requires_databricks`` against
    every Databricks-bound tool kind from ``ToolKind``."""
    import asyncio

    app_dir = asyncio.run(
        _render_and_unpack(["vector_search"], tmp_path / "appunit")
    )
    monkeypatch.setenv("DATABRICKS_APP_NAME", "dr-shell-test")
    app_module = _load_app_module(app_dir)
    helper = app_module._workflow_requires_databricks

    assert helper({"tools": []}) is False
    assert helper({"tools": None}) is False
    assert helper({}) is False
    assert helper({"tools": [{"kind": "web_search"}]}) is False
    assert helper({"tools": [{"kind": "vector_search"}]}) is True
    assert helper({"tools": [{"kind": "genie"}]}) is True
    assert helper({"tools": [{"kind": "knowledge_assistant"}]}) is True
    assert helper({"tools": [{"kind": "delta_read"}]}) is True
    assert helper({"tools": [{"kind": "table_read"}]}) is True
    # Mixed declarations
    assert (
        helper({"tools": [{"kind": "web_search"}, {"kind": "genie"}]})
        is True
    )
    # Malformed entries are tolerated (no crash)
    assert helper({"tools": [{"name": "foo"}]}) is False
    assert helper({"tools": ["not_a_dict"]}) is False

    sys.modules.pop("app", None)
