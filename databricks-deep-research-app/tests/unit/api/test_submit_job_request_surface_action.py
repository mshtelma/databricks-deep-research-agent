"""SubmitJobRequest.surface_action validation (structured-output binding id)."""

import pytest
from pydantic import ValidationError

from deep_research.api.v1.jobs import SubmitJobRequest

pytestmark = pytest.mark.unit

_BASE = {"chat_id": "0e0f9a7c-64e2-4b52-a1b7-0f2a4a1a2b3c", "query": "q"}


def test_surface_action_defaults_to_none() -> None:
    req = SubmitJobRequest(**_BASE)
    assert req.surface_action is None


def test_surface_action_accepts_identifier() -> None:
    req = SubmitJobRequest(**_BASE, surface_action="run_scan")
    assert req.surface_action == "run_scan"


@pytest.mark.parametrize("bad", ["9run", "a-b", "with space", ""])
def test_surface_action_rejects_non_identifier(bad: str) -> None:
    with pytest.raises(ValidationError):
        SubmitJobRequest(**_BASE, surface_action=bad)


def test_surface_action_rejects_overlong() -> None:
    with pytest.raises(ValidationError):
        SubmitJobRequest(**_BASE, surface_action="a" * 65)
