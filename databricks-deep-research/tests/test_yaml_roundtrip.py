"""Tests for YAML loading, saving, and roundtrip of WorkflowDefinition."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from databricks_deep_research.errors import WorkflowValidationError
from databricks_deep_research.workflow.definition import WorkflowDefinition
from databricks_deep_research.workflow.loader import (
    load_workflow,
    load_workflow_from_dict,
    load_workflow_from_string,
    save_workflow,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

MINIMAL_YAML = """\
id: test-wf
name: Test Workflow
root:
  id: root
  type: sequence
  label: Root
  children:
    - id: step1
      type: agent
      label: Researcher
      config:
        subtype: researcher
"""

REALISTIC_YAML = """\
id: deep-research
name: Deep Research Pipeline
description: Multi-step research with reflection
version: 2
required_inputs: [query, context]
output_keys: [report, citations]
token_budget: 500000
timeout_seconds: 900
root:
  id: pipeline
  type: sequence
  label: Main Pipeline
  children:
    - id: parallel-research
      type: parallel
      label: Parallel Research
      children:
        - id: web-search
          type: agent
          label: Web Researcher
          config:
            subtype: researcher
            output_key: web_results
        - id: doc-search
          type: agent
          label: Document Researcher
          config:
            subtype: researcher
            output_key: doc_results
    - id: reflect-loop
      type: loop
      label: Reflection Loop
      config:
        until:
          type: state
          key: output
          operator: exists
        max_iterations: 3
      children:
        - id: reflector
          type: agent
          label: Reflector
          config:
            subtype: reflector
      error_handling:
        on_error: retry
        max_retries: 2
        retry_delay_seconds: 0.5
    - id: synthesizer
      type: agent
      label: Synthesizer
      config:
        subtype: synthesizer
"""


def _write_yaml(tmp_path: Path, content: str, name: str = "workflow.yaml") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# 1. load_workflow from a YAML file
# ---------------------------------------------------------------------------


def test_load_workflow_from_file(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, MINIMAL_YAML)
    wf = load_workflow(path)
    assert wf.id == "test-wf"
    assert wf.name == "Test Workflow"
    assert wf.root.id == "root"
    assert len(wf.root.children) == 1
    assert wf.root.children[0].config["subtype"] == "researcher"


def test_load_workflow_file_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_workflow("/nonexistent/path.yaml")


# ---------------------------------------------------------------------------
# 2. save_workflow writes valid YAML
# ---------------------------------------------------------------------------


def test_save_workflow_writes_valid_yaml(tmp_path: Path) -> None:
    wf = load_workflow_from_string(MINIMAL_YAML)
    out = tmp_path / "out.yaml"
    save_workflow(wf, out)

    assert out.exists()
    raw = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert raw["id"] == "test-wf"
    assert raw["root"]["children"][0]["type"] == "agent"


# ---------------------------------------------------------------------------
# 3. Roundtrip: save -> load -> compare
# ---------------------------------------------------------------------------


def test_roundtrip_save_load(tmp_path: Path) -> None:
    original = load_workflow_from_string(MINIMAL_YAML)
    path = tmp_path / "roundtrip.yaml"
    save_workflow(original, path)
    reloaded = load_workflow(path)

    assert reloaded.id == original.id
    assert reloaded.name == original.name
    assert reloaded.root.id == original.root.id
    assert len(reloaded.root.children) == len(original.root.children)
    assert reloaded.model_dump() == original.model_dump()


# ---------------------------------------------------------------------------
# 3b. mcp_servers survive load + dump->load (regression: previously dropped)
# ---------------------------------------------------------------------------

MCP_YAML = """\
id: mcp-wf
name: MCP Workflow
mcp_servers:
  - name: my_remote
    url: https://mcp.example.com/sse
    transport: sse
    citeable: true
root:
  id: root
  type: sequence
  label: Root
  children:
    - id: step1
      type: agent
      label: Researcher
      config:
        subtype: researcher
"""


def test_mcp_servers_survive_load_and_roundtrip() -> None:
    # Regression: the loader constructed WorkflowDefinition WITHOUT mcp_servers,
    # so persisted/loaded workflows silently lost their MCP attachments before
    # the orchestrator could inject them. Assert they survive an initial load
    # AND a model_dump -> load_workflow_from_dict cycle (the app's persist path).
    wf = load_workflow_from_string(MCP_YAML)
    assert len(wf.mcp_servers) == 1
    assert wf.mcp_servers[0].name == "my_remote"
    assert wf.mcp_servers[0].url == "https://mcp.example.com/sse"
    assert wf.mcp_servers[0].transport == "sse"

    reloaded = load_workflow_from_dict(wf.model_dump(mode="json"))
    assert len(reloaded.mcp_servers) == 1
    assert reloaded.mcp_servers[0].name == "my_remote"
    assert reloaded.model_dump()["mcp_servers"] == wf.model_dump()["mcp_servers"]


# ---------------------------------------------------------------------------
# 4. load_workflow_from_string
# ---------------------------------------------------------------------------


def test_load_workflow_from_string_basic() -> None:
    wf = load_workflow_from_string(MINIMAL_YAML)
    assert wf.id == "test-wf"
    assert wf.root.type.value == "sequence"


# ---------------------------------------------------------------------------
# 5. WorkflowDefinition.from_yaml / to_yaml convenience methods
# ---------------------------------------------------------------------------


def test_from_yaml_classmethod(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, MINIMAL_YAML)
    wf = WorkflowDefinition.from_yaml(path)
    assert wf.id == "test-wf"


def test_to_yaml_method(tmp_path: Path) -> None:
    wf = load_workflow_from_string(MINIMAL_YAML)
    out = tmp_path / "conv.yaml"
    wf.to_yaml(out)
    assert out.exists()
    reloaded = WorkflowDefinition.from_yaml(out)
    assert reloaded.model_dump() == wf.model_dump()


# ---------------------------------------------------------------------------
# 6. Invalid YAML raises appropriate errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("loader", "payload", "match"),
    [
        ("path", "- just\n- a\n- list\n", "must be a mapping"),
        ("string", "id: x\nname: X\n", "missing required field"),
        ("string", "id: x\nname: X\nroot:\n  id: r\n  type: agent\n", "missing required field"),
        ("string", "id: x\nname: X\nroot:\n  id: r\n  type: bogus\n  label: R\n", "Unknown node type"),
        (
            "string",
            (
                "id: x\nname: X\nroot:\n"
                "  id: r\n  type: agent\n  label: R\n"
                "  config:\n"
                "    subtype: synthesizer\n"
                "    grounding_mode: none\n"
                "    output_schema:\n"
                "      synthesis_mode: reclaim\n"
            ),
            "conflicting grounding controls",
        ),
        ("string", "just a string", "must be a mapping"),
    ],
)
def test_invalid_yaml_cases(
    tmp_path: Path,
    loader: str,
    payload: str,
    match: str,
) -> None:
    with pytest.raises(WorkflowValidationError, match=match):
        if loader == "path":
            load_workflow(_write_yaml(tmp_path, payload))
        else:
            load_workflow_from_string(payload)


# ---------------------------------------------------------------------------
# 7. Realistic workflow with nested nodes survives roundtrip
# ---------------------------------------------------------------------------


def test_realistic_nested_roundtrip(tmp_path: Path) -> None:
    original = load_workflow_from_string(REALISTIC_YAML)

    assert original.id == "deep-research"
    assert original.version == 2
    assert original.token_budget == 500000
    assert original.timeout_seconds == 900
    assert original.required_inputs == ["query", "context"]
    assert original.output_keys == ["report", "citations"]

    # Verify nested structure
    root = original.root
    assert root.type.value == "sequence"
    assert len(root.children) == 3

    parallel = root.children[0]
    assert parallel.type.value == "parallel"
    assert len(parallel.children) == 2

    loop = root.children[1]
    assert loop.type.value == "loop"
    assert loop.config["max_iterations"] == 3
    assert loop.error_handling is not None
    assert loop.error_handling.on_error == "retry"
    assert loop.error_handling.max_retries == 2

    # Roundtrip via file
    path = tmp_path / "realistic.yaml"
    save_workflow(original, path)
    reloaded = load_workflow(path)
    assert reloaded.model_dump() == original.model_dump()

    # Roundtrip via string
    raw_text = path.read_text(encoding="utf-8")
    from_str = load_workflow_from_string(raw_text)
    assert from_str.model_dump() == original.model_dump()
