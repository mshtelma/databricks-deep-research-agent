"""Unit tests for WorkflowDefinition.run_as field (ServicePrincipalRunAs + forward-compat).

Run:
    cd databricks-deep-research && uv run pytest tests/unit/agents/test_run_as_validation.py -q
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from databricks_deep_research.workflow.definition import (
    ServicePrincipalRunAs,
    WorkflowDefinition,
)

# ---------------------------------------------------------------------------
# Minimal valid workflow dict (V1 shape — no run_as field)
# ---------------------------------------------------------------------------

_V1_MINIMAL: dict = {
    "id": "wf-001",
    "name": "Test Workflow",
    "version": 1,
    "root": {
        "id": "root",
        "type": "agent",
        "label": "researcher",
        "config": {"subtype": "researcher"},
        "children": [],
    },
}

_VALID_UUID = "12345678-1234-5678-1234-567812345678"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_definition(**overrides: object) -> WorkflowDefinition:
    """Build a WorkflowDefinition from _V1_MINIMAL with optional field overrides."""
    data = {**_V1_MINIMAL, **overrides}
    return WorkflowDefinition.model_validate(data)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDefaultCaller:
    def test_default_caller(self) -> None:
        """WorkflowDefinition with no run_as field defaults to 'caller'."""
        defn = _make_definition()
        assert defn.run_as == "caller"


class TestValidSpId:
    def test_valid_sp_id(self) -> None:
        """WorkflowDefinition with a valid UUID run_as parses to ServicePrincipalRunAs."""
        defn = _make_definition(run_as={"service_principal_id": _VALID_UUID})
        assert isinstance(defn.run_as, ServicePrincipalRunAs)
        assert defn.run_as.service_principal_id == _VALID_UUID

    def test_string_caller_accepted(self) -> None:
        """Explicit run_as='caller' is accepted as the string 'caller'."""
        defn = _make_definition(run_as="caller")
        assert defn.run_as == "caller"


class TestInvalidUuidRejected:
    def test_invalid_uuid_rejected(self) -> None:
        """Malformed UUID in service_principal_id raises ValidationError."""
        with pytest.raises(ValidationError, match="UUID"):
            _make_definition(run_as={"service_principal_id": "not-a-uuid"})

    def test_empty_uuid_rejected(self) -> None:
        """Empty string in service_principal_id raises ValidationError."""
        with pytest.raises(ValidationError):
            _make_definition(run_as={"service_principal_id": ""})

    def test_bad_run_as_string_rejected(self) -> None:
        """A run_as string other than 'caller' raises ValidationError."""
        with pytest.raises(ValidationError):
            _make_definition(run_as="sp")


class TestV1ForwardCompat:
    def test_v1_forward_compat(self) -> None:
        """A V1 fixture dict without run_as field parses cleanly with run_as=='caller'."""
        # _V1_MINIMAL deliberately has no run_as key
        assert "run_as" not in _V1_MINIMAL
        defn = WorkflowDefinition.model_validate(_V1_MINIMAL)
        assert defn.run_as == "caller"

    def test_none_run_as_coerced(self) -> None:
        """Explicit None in run_as is coerced to 'caller'."""
        defn = _make_definition(run_as=None)
        assert defn.run_as == "caller"


class TestDiscriminator:
    def test_caller_string_is_str(self) -> None:
        """run_as='caller' returns the plain string 'caller', not a model instance."""
        defn = _make_definition(run_as="caller")
        assert defn.run_as == "caller"
        assert isinstance(defn.run_as, str)

    def test_sp_dict_is_model(self) -> None:
        """run_as with service_principal_id returns a ServicePrincipalRunAs instance."""
        defn = _make_definition(run_as={"service_principal_id": _VALID_UUID})
        assert isinstance(defn.run_as, ServicePrincipalRunAs)
        assert defn.run_as.service_principal_id == _VALID_UUID
