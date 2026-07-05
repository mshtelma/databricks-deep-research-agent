"""Unit tests for POST /api/v1/agent-designer/scaffold-surface.

Stateless endpoint: no DB, no LLM. Derives a declarative UI surface from
a workflow definition's ``required_inputs``.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from deep_research.core.auth import UserIdentity
from deep_research.db.session import get_db
from deep_research.main import app
from deep_research.middleware.auth import get_current_user_identity

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_user() -> UserIdentity:
    return UserIdentity(
        user_id="test-user-surface",
        email="surface@example.com",
        display_name="Surface Tester",
    )


@pytest.fixture
def client(mock_user: UserIdentity) -> TestClient:
    from unittest.mock import AsyncMock

    async def override_get_db() -> Any:
        mock_session = AsyncMock()
        yield mock_session

    async def override_get_current_user_identity() -> UserIdentity:
        return mock_user

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user_identity] = (
        override_get_current_user_identity
    )

    yield TestClient(app)

    app.dependency_overrides.clear()


def _minimal_definition(**overrides: Any) -> dict[str, Any]:
    """A valid minimal workflow definition for the scaffold endpoint."""
    definition: dict[str, Any] = {
        "id": "wf",
        "name": "Test Workflow",
        "description": "A test workflow.",
        "version": 1,
        "root": {
            "id": "root",
            "type": "agent",
            "label": "Agent",
            "config": {
                "subtype": "researcher",
                "input_keys": ["query"],
                "output_key": "out",
            },
            "children": [],
        },
        "tools": [],
        "pools": [],
        "sources": [],
        "models": {},
        "required_inputs": ["query"],
        "output_keys": ["out"],
    }
    definition.update(overrides)
    return definition


class TestScaffoldSurfaceEndpoint:
    """Tests for POST /api/v1/agent-designer/scaffold-surface."""

    def test_happy_path_returns_200_with_run_binding(self, client: TestClient) -> None:
        """Minimal definition → 200 with a surface whose bindings[0].action == 'run'."""
        response = client.post(
            "/api/v1/agent-designer/scaffold-surface",
            json={"definition": _minimal_definition()},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert "surface" in body
        surface = body["surface"]
        assert surface["version"] == 1
        assert len(surface["bindings"]) >= 1
        assert surface["bindings"][0]["action"] == "run"
        # The form must contain a root component.
        component_ids = {c["id"] for c in surface["components"]}
        assert "root" in component_ids
        assert surface["runtime_controls"]["effort"] == "show"
        assert surface["layout"]["actions"] == "host_bar"
        assert {section["role"] for section in surface["layout"]["sections"]} == {
            "inputs",
            "results",
        }
        assert "depth_select" not in component_ids
        assert "verify_checkbox" not in component_ids
        assert "options" not in surface["data_model"]

    def test_reserved_input_key_returns_422(self, client: TestClient) -> None:
        """A definition with required_inputs containing a reserved key → HTTP 422."""
        # "plan" is in RESERVED_INPUT_KEYS (selector-shadowed).
        definition = _minimal_definition(required_inputs=["plan"])
        response = client.post(
            "/api/v1/agent-designer/scaffold-surface",
            json={"definition": definition},
        )
        assert response.status_code == 422, response.text
        # The app may serialize the HTTPException detail under "detail" or "message"
        # depending on the installed exception handler middleware.
        body = response.json()
        body_text = str(body)
        assert "plan" in body_text or "reserved" in body_text or "cannot scaffold" in body_text

    def test_extra_body_field_rejected(self, client: TestClient) -> None:
        """extra='forbid' on the request model → 422 when an unknown field is sent."""
        response = client.post(
            "/api/v1/agent-designer/scaffold-surface",
            json={"definition": _minimal_definition(), "unexpected_field": True},
        )
        assert response.status_code == 422, response.text
