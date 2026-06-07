"""Integration tests for POST /api/v1/agent-designer/import-yaml.

Gated by RUN_INTEGRATION_TESTS=1 — no real DB or LLM is required; the
endpoint is fully stateless so we use _noauth_client (mock DB + anonymous
user) for all tests.

Run all:
    RUN_INTEGRATION_TESTS=1 uv run pytest tests/integration/test_yaml_import.py -v

Run without env var (expect clean module-level skip):
    uv run pytest tests/integration/test_yaml_import.py -q
"""
from __future__ import annotations

import contextlib
import os
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock

import pytest
import yaml

# Must be set before importing `app` so that Settings() validation does not
# require LAKEBASE_*/DATABASE_URL.
os.environ.setdefault("STORAGE_SERVICE_IMPL", "sqlalchemy_legacy")

# ---------------------------------------------------------------------------
# Module-level skip guard — matches Phase 1 convention
# ---------------------------------------------------------------------------

_RUN_TESTS = os.environ.get("RUN_INTEGRATION_TESTS") == "1"

if not _RUN_TESTS:
    pytest.skip("Requires RUN_INTEGRATION_TESTS=1", allow_module_level=True)

# ---------------------------------------------------------------------------
# Deferred imports (only reached when RUN_INTEGRATION_TESTS=1)
# ---------------------------------------------------------------------------

from fastapi.testclient import TestClient  # noqa: E402

from deep_research.agent_designer.registry import REGISTRY_VERSION  # noqa: E402
from deep_research.core.auth import UserIdentity  # noqa: E402
from deep_research.db.session import get_db  # noqa: E402
from deep_research.main import app  # noqa: E402
from deep_research.middleware.auth import get_current_user_identity  # noqa: E402
from deep_research.storage.observability import RecordingSink, use_sink  # noqa: E402

# ---------------------------------------------------------------------------
# Valid workflow fixture (mirrors test_yaml_export.py fixture)
# ---------------------------------------------------------------------------

_VALID_DEFINITION: dict[str, Any] = {
    "id": "yaml-import-test-wf",
    "name": "YAML Import Test Workflow",
    "version": 1,
    "root": {
        "id": "root-seq",
        "type": "sequence",
        "label": "main",
        "config": {},
        "children": [
            {
                "id": "agent-node",
                "type": "agent",
                "label": "researcher",
                "config": {"subtype": "researcher"},
                "children": [],
            },
        ],
    },
    "tools": [],
    "pools": [],
    "sources": [],
    "models": {},
    "required_inputs": ["query"],
    "output_keys": ["output"],
    "token_budget": 0,
    "timeout_seconds": 1800,
}


def _make_valid_yaml(definition: dict[str, Any] | None = None) -> bytes:
    """Wrap a definition dict in registry_version and serialise to YAML bytes."""
    defn = definition if definition is not None else _VALID_DEFINITION
    doc: dict[str, Any] = {"registry_version": REGISTRY_VERSION, **defn}
    return yaml.safe_dump(doc, sort_keys=True, allow_unicode=True, indent=2).encode()


# ---------------------------------------------------------------------------
# Test-client helper (no DB required — endpoint is stateless)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _noauth_client() -> Generator[TestClient, None, None]:
    """TestClient with mocked DB and anonymous user identity."""

    async def _override_db() -> Any:
        yield AsyncMock()

    async def _override_user() -> UserIdentity:
        return UserIdentity.anonymous()

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_user_identity] = _override_user
    try:
        # raise_server_exceptions=False so HTTPException produces a proper HTTP
        # response (with the app's error envelope) rather than propagating the
        # Python exception to the test.
        client = TestClient(app, raise_server_exceptions=False)
        yield client
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_current_user_identity, None)


_URL = "/api/v1/agent-designer/import-yaml"


# ---------------------------------------------------------------------------
# 1. Happy path — sequence root with agent child
# ---------------------------------------------------------------------------

def test_happy_path_sequence_root() -> None:
    """Valid YAML with sequence root returns 200 and a populated definition."""
    with _noauth_client() as client:
        resp = client.post(_URL, content=_make_valid_yaml())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "definition" in body
    assert "workflow_summary" in body
    summary = body["workflow_summary"]
    assert summary["node_count"] >= 1


# ---------------------------------------------------------------------------
# 2. Happy path — parallel root
# ---------------------------------------------------------------------------

def test_happy_path_parallel_root() -> None:
    """Valid YAML with parallel root is accepted."""
    defn: dict[str, Any] = {
        **_VALID_DEFINITION,
        "id": "parallel-root-wf",
        "root": {
            "id": "root-par",
            "type": "parallel",
            "label": "parallel root",
            "config": {},
            "children": [
                {
                    "id": "agent-a",
                    "type": "agent",
                    "label": "researcher-a",
                    "config": {"subtype": "researcher"},
                    "children": [],
                },
                {
                    "id": "agent-b",
                    "type": "agent",
                    "label": "researcher-b",
                    "config": {"subtype": "researcher"},
                    "children": [],
                },
            ],
        },
    }
    with _noauth_client() as client:
        resp = client.post(_URL, content=_make_valid_yaml(defn))
    assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# 3. Happy path — nested sequence > agent
# ---------------------------------------------------------------------------

def test_happy_path_nested_sequence() -> None:
    """A two-level sequence > agent structure parses and validates."""
    defn: dict[str, Any] = {
        **_VALID_DEFINITION,
        "id": "nested-seq-wf",
        "root": {
            "id": "outer-seq",
            "type": "sequence",
            "label": "outer",
            "config": {},
            "children": [
                {
                    "id": "inner-agent",
                    "type": "agent",
                    "label": "synthesizer",
                    "config": {"subtype": "synthesizer"},
                    "children": [],
                },
            ],
        },
    }
    with _noauth_client() as client:
        resp = client.post(_URL, content=_make_valid_yaml(defn))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["definition"]["id"] == "nested-seq-wf"


# ---------------------------------------------------------------------------
# 4. Malformed YAML → 400 schema_error
# ---------------------------------------------------------------------------

def test_malformed_yaml_returns_400() -> None:
    """A YAML syntax error (e.g. bad indentation) returns 400 schema_error."""
    bad_yaml = b"root:\n  - unclosed: [list\n  bad indent"
    with _noauth_client() as client:
        resp = client.post(_URL, content=bad_yaml)
    assert resp.status_code == 400, resp.text
    body = resp.json()
    errors = body["message"]["errors"]
    assert len(errors) >= 1
    assert errors[0]["kind"] == "schema_error"


# ---------------------------------------------------------------------------
# 5. Non-mapping top-level YAML → 400 schema_error
# ---------------------------------------------------------------------------

def test_non_mapping_toplevel_returns_400() -> None:
    """YAML that parses to a list (not a dict) returns 400 schema_error."""
    list_yaml = b"- item_a\n- item_b\n"
    with _noauth_client() as client:
        resp = client.post(_URL, content=list_yaml)
    assert resp.status_code == 400, resp.text
    errors = resp.json()["message"]["errors"]
    assert errors[0]["kind"] == "schema_error"


# ---------------------------------------------------------------------------
# 6. Oversized payload → 413 too_large
# ---------------------------------------------------------------------------

def test_oversized_payload_returns_413(monkeypatch: pytest.MonkeyPatch) -> None:
    """Payload exceeding AGENT_DESIGNER_YAML_MAX_BYTES returns 413 too_large."""
    monkeypatch.setenv("AGENT_DESIGNER_YAML_MAX_BYTES", "10")
    with _noauth_client() as client:
        resp = client.post(_URL, content=_make_valid_yaml())
    # Status may be 413 from endpoint pre-check or from parse_and_validate_yaml
    assert resp.status_code == 413, resp.text


# ---------------------------------------------------------------------------
# 7. Adversarial !!python/object gadget — must be caught, not executed
# ---------------------------------------------------------------------------

def test_adversarial_yaml_gadget_is_rejected() -> None:
    """``!!python/object/apply:os.system`` gadget must be caught and rejected.

    ``yaml.safe_load`` refuses to deserialise !!python/object constructs and
    raises a ``yaml.constructor.ConstructorError``.  The endpoint must return
    400 with error_kind ``schema_error`` (or ``unsafe``) — never 200.
    """
    adversarial = (
        b"registry_version: " + REGISTRY_VERSION.encode() + b"\n"
        b"id: evil\n"
        b"name: evil\n"
        b"root: !!python/object/apply:os.system ['echo pwned']\n"
    )
    with _noauth_client() as client:
        resp = client.post(_URL, content=adversarial)
    assert resp.status_code in (400, 413), resp.text
    errors = resp.json()["message"]["errors"]
    assert errors[0]["kind"] in ("schema_error", "unsafe")


# ---------------------------------------------------------------------------
# 8. Registry version mismatch → 400 registry_version_mismatch
# ---------------------------------------------------------------------------

def test_registry_version_mismatch_returns_400() -> None:
    """A ``registry_version`` that differs from the current one returns 400."""
    doc: dict[str, Any] = {"registry_version": "0.0.0", **_VALID_DEFINITION}
    bad_version_yaml = yaml.safe_dump(doc).encode()
    with _noauth_client() as client:
        resp = client.post(_URL, content=bad_version_yaml)
    assert resp.status_code == 400, resp.text
    body = resp.json()
    errors = body["message"]["errors"]
    assert errors[0]["kind"] == "registry_version_mismatch"
    assert REGISTRY_VERSION in errors[0]["message"]
    assert "0.0.0" in errors[0]["message"]
    # Actionable guidance: tells the importer how to proceed.
    assert "remove the registry_version line" in errors[0]["message"]


# ---------------------------------------------------------------------------
# 9. Missing registry_version → accepted as raw framework YAML (200)
# ---------------------------------------------------------------------------

def test_missing_registry_version_is_accepted() -> None:
    """YAML without a ``registry_version`` envelope imports as raw framework YAML.

    Absent (or null) ``registry_version`` is treated as the current version so
    hand-written framework workflows and legacy pre-envelope exports import
    without edits.  A present-but-different value is still rejected (test 8).
    """
    doc: dict[str, Any] = dict(_VALID_DEFINITION)  # no registry_version key
    no_version_yaml = yaml.safe_dump(doc).encode()
    with _noauth_client() as client:
        resp = client.post(_URL, content=no_version_yaml)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["definition"]["id"] == _VALID_DEFINITION["id"]
    assert "registry_version" not in body["definition"]


# ---------------------------------------------------------------------------
# 10. AST fails load_workflow_from_dict → 400 schema_error
# ---------------------------------------------------------------------------

def test_invalid_ast_node_type_returns_400() -> None:
    """YAML that parses fine but contains an unknown node_type fails AST validation."""
    defn: dict[str, Any] = {
        **_VALID_DEFINITION,
        "id": "bad-node-wf",
        "root": {
            "id": "bad-root",
            "type": "nonexistent_node_type",
            "label": "bad",
            "config": {},
            "children": [],
        },
    }
    with _noauth_client() as client:
        resp = client.post(_URL, content=_make_valid_yaml(defn))
    assert resp.status_code == 400, resp.text
    errors = resp.json()["message"]["errors"]
    assert errors[0]["kind"] == "schema_error"


# ---------------------------------------------------------------------------
# 11. Successful import increments outcome counter exactly once
# ---------------------------------------------------------------------------

def test_success_increments_outcome_counter_once() -> None:
    """A successful import increments yaml_import_outcome{outcome=success} by 1."""
    sink = RecordingSink()
    with use_sink(sink):
        with _noauth_client() as client:
            resp = client.post(_URL, content=_make_valid_yaml())
    assert resp.status_code == 200, resp.text
    assert sink.count("agent_designer.yaml_import_outcome", outcome="success") == 1.0


# ---------------------------------------------------------------------------
# 12. Error response shape matches expected structure
# ---------------------------------------------------------------------------

def test_error_response_shape() -> None:
    """Error responses include detail.errors[].{path, kind, message}."""
    doc: dict[str, Any] = {"registry_version": "99.0.0", **_VALID_DEFINITION}
    with _noauth_client() as client:
        resp = client.post(_URL, content=yaml.safe_dump(doc).encode())
    assert resp.status_code == 400, resp.text
    detail = resp.json()["message"]
    assert "errors" in detail
    error = detail["errors"][0]
    assert "kind" in error
    assert "message" in error
    assert "path" in error  # may be None but key must be present
