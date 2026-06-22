"""AgentV2Service.create/update fill databricks web-search defaults on save.

Guards the designer-save hardening: a web tool persisted with provider=databricks
but no model (e.g. saved via the UI inspector, which bypasses the full designer
normalizer) is made self-describing at save time, so it does not later fail at
tool construction with 'requires a serving endpoint'.

``materialize_for_save`` is identity-mocked (it is exercised by its own tests) so
these isolate the new fill step, and the helper's config read is patched for a
deterministic endpoint.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from deep_research.models.agent_v2 import AgentV2
from deep_research.schemas.agent_v2 import (
    CreateAgentV2Request,
    UpdateAgentV2Request,
)
from deep_research.services.agent_v2_service import AgentV2Service


def _fake_app_config() -> SimpleNamespace:
    from deep_research.core.app_config import DatabricksSearchConfig

    return SimpleNamespace(
        search=SimpleNamespace(
            provider="brave",
            databricks=DatabricksSearchConfig(endpoint="databricks-gpt-5"),
        )
    )


def _databricks_web_def() -> dict:
    return {
        "id": "wf",
        "name": "Account Research",
        "version": 1,
        "required_inputs": ["query"],
        "output_keys": ["report"],
        "tools": [
            {"name": "web_research", "kind": "web_research",
             "config": {"provider": "databricks"}},
        ],
        "root": {"id": "root", "type": "sequence", "label": "wf",
                 "config": {}, "children": []},
    }


@pytest.fixture(autouse=True)
def _patch_config_and_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "deep_research.agent_designer.ast_normalizer.get_app_config",
        _fake_app_config,
    )
    monkeypatch.setattr(
        "deep_research.services.agent_v2_service.CatalogService.materialize_for_save",
        lambda self, definition, previous=None: definition,
    )


@pytest.mark.asyncio
async def test_create_fills_databricks_web_tool_model(
    mock_db_session: AsyncMock,
) -> None:
    request = CreateAgentV2Request.model_construct(
        name="AccountResearch",
        description=None,
        avatar_url=None,
        visibility="private",
        definition=_databricks_web_def(),
    )
    service = AgentV2Service(mock_db_session)

    agent = await service.create("user-1", request)

    cfg = agent.definition["tools"][0]["config"]
    assert cfg["provider"] == "databricks"
    assert cfg["model"] == "databricks-gpt-5"


@pytest.mark.asyncio
async def test_update_fills_databricks_web_tool_model(
    mock_db_session: AsyncMock,
) -> None:
    agent_id = uuid4()
    existing = AgentV2(
        owner_id="user-1",
        name="AccountResearch",
        description=None,
        visibility="private",
        definition={"root": {}},
        schema_version=1,
        etag="etag-1",
    )
    existing.id = agent_id
    # get_owned() resolves via execute(...).scalar_one_or_none(); assign an
    # explicit MagicMock result so scalar_one_or_none is sync (not an auto-created
    # async child of the AsyncMock session).
    result = MagicMock()
    result.scalar_one_or_none.return_value = existing
    mock_db_session.execute.return_value = result

    request = UpdateAgentV2Request.model_construct(
        name=None,
        description=None,
        avatar_url=None,
        visibility=None,
        definition=_databricks_web_def(),
    )
    service = AgentV2Service(mock_db_session)

    updated = await service.update(
        agent_id, "user-1", request, if_match_etag="etag-1"
    )

    assert updated is not None
    cfg = updated.definition["tools"][0]["config"]
    assert cfg["provider"] == "databricks"
    assert cfg["model"] == "databricks-gpt-5"
