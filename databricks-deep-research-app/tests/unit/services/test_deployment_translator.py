"""Unit tests for the DeploymentTranslator protocol + InAppTranslator (US-104)."""
from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from deep_research.models.agent_deployment import AgentDeployment, DeploymentMode
from deep_research.services.deployment import (
    Artifact,
    DeploymentResult,
    DeploymentTranslator,
    InAppTranslator,
    ValidationResult,
)


class TestProtocolConformance:
    def test_in_app_translator_satisfies_protocol(self) -> None:
        """runtime_checkable Protocol conformance check."""
        assert isinstance(InAppTranslator(), DeploymentTranslator)

    def test_translator_has_correct_mode_classvar(self) -> None:
        assert InAppTranslator.mode == DeploymentMode.IN_APP


class TestInAppTranslatorBehaviour:
    @pytest.mark.asyncio
    async def test_validate_returns_valid(self) -> None:
        translator = InAppTranslator()
        agent = MagicMock(id=uuid4())
        revision = MagicMock(rev_id=uuid4())
        result = await translator.validate(agent, revision, {"mode": "in_app"})
        assert isinstance(result, ValidationResult)
        assert result.valid is True
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_translate_emits_artifact_with_revision_ref(self) -> None:
        translator = InAppTranslator()
        agent_id = uuid4()
        rev_id = uuid4()
        agent = MagicMock(id=agent_id)
        revision = MagicMock(rev_id=rev_id)
        artifact = await translator.translate(agent, revision, {"mode": "in_app"})
        assert isinstance(artifact, Artifact)
        assert artifact.mode == DeploymentMode.IN_APP
        assert artifact.payload == {"revision_id": str(rev_id)}
        assert artifact.metadata == {"agent_id": str(agent_id)}

    @pytest.mark.asyncio
    async def test_deploy_returns_success(self) -> None:
        translator = InAppTranslator()
        artifact = Artifact(mode=DeploymentMode.IN_APP, payload={})
        deployment = MagicMock(spec=AgentDeployment)
        result = await translator.deploy(artifact, {"mode": "in_app"}, deployment)
        assert isinstance(result, DeploymentResult)
        assert result.success is True
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_deactivate_is_noop(self) -> None:
        translator = InAppTranslator()
        deployment = MagicMock(spec=AgentDeployment)
        # Should complete without raising; idempotent for any future re-call.
        result = await translator.deactivate(deployment)
        assert result is None
        # Calling twice in a row must also be safe (idempotency contract).
        result2 = await translator.deactivate(deployment)
        assert result2 is None
