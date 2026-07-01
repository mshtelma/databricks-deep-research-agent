"""Runtime tests for the rendered shell-app template's /api/chat handler.

These tests load the rendered ``app.py`` from a Shell-App artifact, patch
out external dependencies (LLM client, MLflow tracing, WorkspaceClient
default auth-detect), and exercise the FastAPI app in-process to prove
the OBO fail-closed gate behaves correctly.

Regression target: the original shell-app template silently fell back to
the app service principal when ``x-forwarded-access-token`` was absent,
which caused UC-gated vector_search / Genie / serving-endpoint queries
to return empty without any visible error. The fail-closed gate added in
template version 2026-05-28.1 is what these tests guard.
"""
from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
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
    - ``databricks_deep_research.WorkflowRunner`` → MagicMock (the template
      still imports it for an ``inspect.signature`` feature-probe).
    - ``databricks_deep_research.build_databricks_workflow_runner`` →
      MagicMock so the test can assert whether — and with what kwargs — a
      per-request runner was built (used to prove the fail-closed gate
      short-circuits before runner build). Exposed as ``_test_runner_builder``.
    """
    sys_path_entry = str(app_dir)
    sys.path.insert(0, sys_path_entry)

    fake_ws = MagicMock()
    fake_ws.config.host = "https://test.cloud.databricks.com"
    fake_runner = MagicMock()
    fake_runner.stream = MagicMock()  # not async-iterable; will fail if called
    fake_runner_class = MagicMock(return_value=fake_runner)
    # The template builds the per-request runner via the framework's
    # ``build_databricks_workflow_runner`` (it no longer constructs
    # ``WorkflowRunner`` directly). Patch that builder so tests can observe
    # whether — and with what kwargs — a runner was built.
    fake_runner_builder = MagicMock(return_value=fake_runner)
    no_op_span = MagicMock()
    no_op_span.__enter__ = MagicMock(return_value=no_op_span)
    no_op_span.__exit__ = MagicMock(return_value=False)

    # Stub the framework's workflow loader so the test doesn't have to build a
    # fully-valid WorkflowDefinition. The OBO fail-closed gate reads the LOADED
    # definition's ``.tools`` (``workflow_requires_databricks(_DEFINITION)``),
    # so the stub mirrors the real loader by exposing ``.tools`` carrying each
    # declared ``.kind`` taken from the embedded ``agent.yaml`` dict.
    def _fake_load_workflow_from_dict(definition: dict[str, Any]) -> MagicMock:
        declared = definition.get("tools") or []
        loaded = MagicMock(id=definition.get("id", "shell-app-test"), output_keys=[])
        loaded.tools = [
            SimpleNamespace(kind=tool.get("kind"), name=tool.get("name"))
            for tool in declared
            if isinstance(tool, dict)
        ]
        return loaded

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
            "databricks_deep_research.build_databricks_workflow_runner",
            new=fake_runner_builder,
        ),
        patch(
            "databricks_deep_research.workflow.loader.load_workflow_from_dict",
            side_effect=_fake_load_workflow_from_dict,
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
    # Attach the mocks so tests can introspect construction. The builder is
    # the seam the template uses now; the runner class is kept for back-compat.
    module._test_runner_class = fake_runner_class  # type: ignore[attr-defined]
    module._test_runner_builder = fake_runner_builder  # type: ignore[attr-defined]
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

    # Critical: the runner must NOT have been built.
    assert (
        app_module._test_runner_builder.call_count == 0
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

    # Runner MUST be built because the fail-closed gate is bypassed.
    assert app_module._test_runner_builder.call_count == 1, (
        "runner should have been built in local-dev mode — gate fired "
        "incorrectly when DATABRICKS_APP_NAME was unset"
    )


@pytest.mark.asyncio
async def test_table_tool_context_is_wired_in_rendered_shell_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_app_module: None,
) -> None:
    """Rendered shell apps must thread the resolved SQL warehouse into the
    framework runner builder, which wires the table_* factory dependencies.

    Regression target: shell apps used ``ToolFactoryContext.from_defaults``
    directly, which left table_registry/schema_cache/sql_executor unset and
    caused strict runtime resolution to report table tools as missing. The
    template now delegates that wiring to the framework's
    ``build_databricks_workflow_runner`` (one shared implementation). The
    actual ctx population (sql_executor/schema_cache/table_registry) is
    asserted at the framework level in
    ``databricks-deep-research/tests/unit/tools/test_databricks_runner.py``
    and ``.../builtins/text_table/test_runtime_wiring.py``; here we assert the
    app-side contribution — the resolved ``STORAGE_WAREHOUSE_ID``, OBO token,
    and SP/LLM clients are threaded into that builder.
    """
    app_dir = await _render_and_unpack(["table_read"], tmp_path / "app-table")
    monkeypatch.delenv("DATABRICKS_APP_NAME", raising=False)
    monkeypatch.setenv("STORAGE_WAREHOUSE_ID", "d837825f69a03500")
    app_module = _load_app_module(app_dir)

    app_module._build_per_request_runner("obo-token")

    assert app_module._test_runner_builder.call_count == 1
    _, kwargs = app_module._test_runner_builder.call_args
    assert kwargs["warehouse_id"] == "d837825f69a03500"
    assert kwargs["user_token"] == "obo-token"
    assert kwargs["sp_workspace_client"] is app_module._SP_WORKSPACE_CLIENT
    assert kwargs["llm_client"] is app_module._llm_client


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

    # Runner MUST be built — web-only workflows don't need OBO.
    assert app_module._test_runner_builder.call_count == 1, (
        "runner should have been built for a purely-web workflow; the "
        "fail-closed gate fired incorrectly"
    )


# NOTE: the per-kind unit check for ``workflow_requires_databricks`` lives at
# the framework level (the single source of truth) in
# ``databricks-deep-research/tests/unit/tools/test_databricks_runner.py``
# (``test_workflow_requires_databricks_*``). The rendered template imports that
# helper directly; its integrated behavior is exercised by the
# ``_WORKFLOW_REQUIRES_DATABRICKS`` assertions in the fail-closed / local-dev /
# purely-web tests above.


def _parallel_lanes_revision_binding_undeclared_web() -> tuple[MagicMock, MagicMock]:
    """A parallel_lanes agent whose lanes bind ``web_research`` while the
    workflow-level ``tools`` list is EMPTY — the exact production AST that
    failed in the shell app with "missing declared tools: ['web_research']".
    """
    agent = MagicMock(id=uuid4(), name="Correlated Stocks Researcher")
    revision = MagicMock(
        rev_id=uuid4(),
        definition={
            "id": "shell-app-lanes",
            "name": "correlated-stocks",
            "version": 1,
            "tools": [],  # the bug: nothing declared, yet lanes bind web_research
            "root": {
                "id": "root",
                "type": "sequence",
                "label": "Root",
                "children": [
                    {
                        "id": "coordinator",
                        "type": "agent",
                        "label": "Coordinator",
                        "config": {"subtype": "coordinator", "output_key": "plan"},
                    },
                    {
                        "id": "research",
                        "type": "parallel",
                        "label": "Research",
                        "children": [
                            {
                                "id": "lane_1-researcher",
                                "type": "agent",
                                "label": "Lane 1",
                                "config": {
                                    "subtype": "researcher",
                                    "output_key": "findings_1",
                                    "tools": ["web_research"],
                                },
                            },
                            {
                                "id": "lane_2-researcher",
                                "type": "agent",
                                "label": "Lane 2",
                                "config": {
                                    "subtype": "researcher",
                                    "output_key": "findings_2",
                                    "tools": ["web_research"],
                                },
                            },
                        ],
                    },
                    {
                        "id": "synth",
                        "type": "agent",
                        "label": "Synthesizer",
                        "config": {"subtype": "synthesizer", "output_key": "output"},
                    },
                ],
            },
        },
    )
    return agent, revision


@pytest.mark.asyncio
async def test_exported_parallel_lanes_agent_yaml_heals_undeclared_web_research(
    tmp_path: Path,
) -> None:
    """Regression: a parallel_lanes agent whose lanes bind an undeclared
    ``web_research`` exports an agent.yaml that the REAL framework loader heals
    on load (no WorkflowError), and the shell app's app.yaml pins the databricks
    web-search endpoint so the inheriting tool gets a backend."""
    import yaml
    from databricks_deep_research.workflow.loader import load_workflow_from_dict

    exporter = ShellAppExporter()
    agent, revision = _parallel_lanes_revision_binding_undeclared_web()
    artifact = await exporter.translate(agent, revision, _config())
    dest = tmp_path / "lanes-app"
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(artifact.payload)) as zf:
        zf.extractall(dest)

    agent_yaml = yaml.safe_load((dest / "agent.yaml").read_text("utf-8"))
    # The baked AST is the broken one (verbatim export) ...
    assert agent_yaml.get("tools") in (None, [])
    # ... but the real loader heals it: web_research is declared, so execution
    # no longer raises "missing declared tools".
    definition = load_workflow_from_dict(agent_yaml)
    declared = {t.name for t in definition.tools}
    assert "web_research" in declared

    # Gap B: the shell app pins the databricks web-search endpoint (config-driven)
    # so an inheriting web tool has a backend without a Brave key.
    app_yaml_text = (dest / "app.yaml").read_text("utf-8")
    assert "DATABRICKS_WEB_SEARCH_ENDPOINT" in app_yaml_text


# ---------------------------------------------------------------------------
# Async-dispatch run registry + resume (LAYER 2 — survives the Databricks Apps
# gateway's absolute connection cap that severs a streamed response mid-run).
# ---------------------------------------------------------------------------


async def _load_registry_module(
    tool_kinds: list[str], dest: Path, monkeypatch: pytest.MonkeyPatch
) -> ModuleType:
    """Render + import a shell-app ``app.py`` (with module side effects mocked)
    so tests can drive its inline ``_RunRegistry``."""
    monkeypatch.delenv("DATABRICKS_APP_NAME", raising=False)
    app_dir = await _render_and_unpack(tool_kinds, dest)
    return _load_app_module(app_dir)


def _sse_events(body: str) -> list[dict[str, str]]:
    """Parse an SSE response body into ``{event, id?, data}`` frames.

    Pure keepalive / comment frames (no ``data:`` line) are skipped, mirroring
    the shell-app frontend's ``handleFrame``.
    """
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    frames: list[dict[str, str]] = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        frame: dict[str, str] = {}
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                frame["event"] = line[len("event:") :].strip()
            elif line.startswith("id:"):
                frame["id"] = line[len("id:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].lstrip())
        if not data_lines:
            continue
        frame["data"] = "\n".join(data_lines)
        frames.append(frame)
    return frames


@pytest.mark.asyncio
async def test_run_registry_drain_resume_and_no_dupes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup_app_module: None
) -> None:
    """A drained run buffers every frame with sequential ``id``s; a resume from
    ``start_seq=k`` replays only ``events[k:]`` (no dupes, no loss, terminal
    frame included), and a resume past the end yields nothing."""
    app_module = await _load_registry_module(
        ["web_search"], tmp_path / "reg1", monkeypatch
    )
    reg = app_module._RunRegistry(ttl_seconds=100, idle_grace_seconds=100)

    async def _producer():
        yield {"event": "node_started", "data": json.dumps({"node_id": "n"})}
        yield {"event": "node_completed", "data": json.dumps({"node_id": "n"})}
        yield {"event": "complete", "data": json.dumps({"output": "FINAL"})}

    run = reg.start(owner="u", run_id="r1", producer=_producer())
    first = [f async for f in reg.stream(run, start_seq=0)]
    assert [f["event"] for f in first] == [
        "node_started",
        "node_completed",
        "complete",
    ]
    assert [f["id"] for f in first] == ["0", "1", "2"]
    assert run.status == "done"

    resumed = [f async for f in reg.stream(run, start_seq=1)]
    assert [f["event"] for f in resumed] == ["node_completed", "complete"]
    assert [f["id"] for f in resumed] == ["1", "2"]

    # since past the buffer end → nothing replayed, returns cleanly (terminal).
    tail = [f async for f in reg.stream(run, start_seq=99)]
    assert tail == []


@pytest.mark.asyncio
async def test_run_registry_reader_detach_does_not_cancel_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup_app_module: None
) -> None:
    """A reader disconnect (the gateway cut) must NOT cancel the producer; a
    later reader resumes and receives the frames produced while disconnected."""
    app_module = await _load_registry_module(
        ["web_search"], tmp_path / "reg2", monkeypatch
    )
    reg = app_module._RunRegistry(ttl_seconds=100, idle_grace_seconds=100)
    gate = asyncio.Event()

    async def _producer():
        yield {"event": "a", "data": "{}"}
        await gate.wait()
        yield {"event": "b", "data": "{}"}
        yield {"event": "complete", "data": "{}"}

    run = reg.start(owner="u", run_id="r2", producer=_producer())
    agen = reg.stream(run, start_seq=0)
    first = await agen.__anext__()
    assert first["event"] == "a"
    await agen.aclose()  # reader detaches mid-run (simulated gateway cut)
    assert run.status == "running"  # producer survives the detach

    gate.set()
    resumed = [f async for f in reg.stream(run, start_seq=1)]
    assert [f["event"] for f in resumed] == ["b", "complete"]
    assert run.status == "done"


@pytest.mark.asyncio
async def test_run_registry_producer_error_and_owner_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup_app_module: None
) -> None:
    """A producer exception is surfaced as a sanitized ``error`` terminal frame
    (status ``error``); ``get`` is owner-scoped; the per-owner cap counts
    in-flight runs."""
    app_module = await _load_registry_module(
        ["web_search"], tmp_path / "reg3", monkeypatch
    )

    # Producer error → sanitized terminal.
    reg = app_module._RunRegistry(ttl_seconds=100, idle_grace_seconds=100)

    async def _boom():
        yield {"event": "a", "data": "{}"}
        raise ValueError("boom")

    run = reg.start(owner="u", run_id="rb", producer=_boom())
    frames = [f async for f in reg.stream(run, start_seq=0)]
    assert frames[0]["event"] == "a"
    assert frames[-1]["event"] == "error"
    assert json.loads(frames[-1]["data"])["error_kind"] == "runtime_error"
    assert run.status == "error"

    # Owner-scoped get: foreign / unknown ids return None.
    async def _finite():
        yield {"event": "complete", "data": "{}"}

    owned = reg.start(owner="alice", run_id="r5", producer=_finite())
    await owned._task
    assert reg.get("r5", owner="bob") is None
    assert reg.get("ghost", owner="alice") is None
    assert reg.get("r5", owner="alice") is owned

    # Per-owner active count (in-flight runs only).
    cap_reg = app_module._RunRegistry(
        per_user_max=2, ttl_seconds=100, idle_grace_seconds=100
    )
    gate = asyncio.Event()

    async def _gated():
        yield {"event": "a", "data": "{}"}
        await gate.wait()

    gated_runs = [
        cap_reg.start(owner="u", run_id=f"c{i}", producer=_gated()) for i in range(2)
    ]
    assert cap_reg.active_count_for_owner("u") == 2
    assert cap_reg.active_count_for_owner("other") == 0
    assert cap_reg.active_count_for_owner("u") >= cap_reg.per_user_max
    gate.set()
    for r in gated_runs:
        if r._task is not None:
            await r._task


@pytest.mark.asyncio
async def test_run_registry_sweep_evicts_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup_app_module: None
) -> None:
    """``sweep`` evicts terminal, reader-less runs past the TTL."""
    app_module = await _load_registry_module(
        ["web_search"], tmp_path / "reg4", monkeypatch
    )
    reg = app_module._RunRegistry(ttl_seconds=0, idle_grace_seconds=100)

    async def _finite():
        yield {"event": "complete", "data": "{}"}

    run = reg.start(owner="u", run_id="rs", producer=_finite())
    _ = [f async for f in reg.stream(run, start_seq=0)]
    assert run.is_terminal
    await asyncio.sleep(0.01)  # let last_access age past the zero TTL
    assert reg.sweep() == 1
    assert reg.get("rs", owner="u") is None


@pytest.mark.asyncio
async def test_resume_endpoint_replays_buffered_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup_app_module: None
) -> None:
    """End-to-end: POST buffers the run; GET …/events?since=N replays the slice
    (including the terminal ``complete``); an unknown id → ``expired`` frame."""
    app_module = await _load_registry_module(
        ["web_search"], tmp_path / "reg5", monkeypatch
    )

    class _FakeEvent:
        def __init__(self, event_type: str) -> None:
            self.event_type = event_type
            self.node_id = "n"

        def model_dump_json(self) -> str:
            return json.dumps({"node_id": "n"})

    async def _fake_stream(**_kwargs: Any):
        for et in ("workflow_started", "node_started", "node_completed"):
            yield _FakeEvent(et)

    fake_runner = MagicMock()
    fake_runner.stream = _fake_stream
    fake_runner.last_result = SimpleNamespace(output="FINAL", sources=[])
    monkeypatch.setattr(
        app_module, "_build_per_request_runner", lambda token: fake_runner
    )

    # A single persistent portal/event-loop so the run's asyncio.Condition
    # created during POST is reusable by the resume GET.
    with TestClient(app_module.app) as client:
        resp = client.post("/api/chat", json={"query": "q"})
        assert resp.status_code == 200
        run_id = resp.headers["x-shell-app-request-id"]
        post_events = _sse_events(resp.text)
        assert [e["event"] for e in post_events] == [
            "workflow_started",
            "node_started",
            "node_completed",
            "complete",
        ]
        assert [e["id"] for e in post_events] == ["0", "1", "2", "3"]

        resume = client.get(f"/api/chat/{run_id}/events?since=2")
        assert resume.status_code == 200
        resume_events = _sse_events(resume.text)
        assert [e["event"] for e in resume_events] == ["node_completed", "complete"]
        assert [e["id"] for e in resume_events] == ["2", "3"]
        assert json.loads(resume_events[-1]["data"])["output"] == "FINAL"

        miss = client.get("/api/chat/does-not-exist/events?since=0")
        assert miss.status_code == 200
        miss_events = _sse_events(miss.text)
        assert miss_events[0]["event"] == "error"
        assert json.loads(miss_events[0]["data"])["error_kind"] == "expired"
