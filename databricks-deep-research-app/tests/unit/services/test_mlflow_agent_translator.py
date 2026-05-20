"""Unit tests for MlflowAgentTranslator (US-303/305).

Mocks ``_deploy_via_sdk`` and ``_deactivate_via_sdk`` via unittest.mock.patch
so tests don't require mlflow + databricks-agents at runtime. Verifies:
  - Protocol conformance
  - validate() error paths (missing UC fields, bad endpoint_name override)
  - translate() writes the workflow definition + builds the right pip pin
  - deploy() success + failure paths
  - deactivate() idempotency
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from deep_research.models.agent_deployment import AgentDeployment, DeploymentMode
from deep_research.services.deployment import DeploymentTranslator
from deep_research.services.deployment.mlflow_deploy import (
    MlflowAgentTranslator,
)
from deep_research.services.deployment.translator import (
    Artifact,
    DeploymentResult,
    ValidationResult,
)


def _valid_config(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "mode": "mlflow_agent",
        "uc_catalog": "main",
        "uc_schema": "agents",
        "uc_model_name": "deep_research",
    }
    base.update(overrides)
    return base


def _agent_revision() -> tuple[MagicMock, MagicMock]:
    return MagicMock(id=uuid4()), MagicMock(
        rev_id=uuid4(),
        definition={"name": "wf", "version": 1, "tools": [], "root": {"type": "sequence", "children": []}},
    )


class TestProtocolConformance:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(MlflowAgentTranslator(), DeploymentTranslator)

    def test_mode_classvar(self) -> None:
        assert MlflowAgentTranslator.mode == DeploymentMode.MLFLOW_AGENT


class TestValidate:
    @pytest.mark.asyncio
    async def test_valid_when_all_required_fields_present(self) -> None:
        translator = MlflowAgentTranslator()
        agent, revision = _agent_revision()
        result = await translator.validate(agent, revision, _valid_config())
        assert isinstance(result, ValidationResult)
        assert result.valid is True

    @pytest.mark.parametrize("field", ["uc_catalog", "uc_schema", "uc_model_name"])
    @pytest.mark.asyncio
    async def test_invalid_when_field_empty(self, field: str) -> None:
        translator = MlflowAgentTranslator()
        agent, revision = _agent_revision()
        result = await translator.validate(
            agent, revision, _valid_config(**{field: ""})
        )
        assert result.valid is False
        assert any(field in e.message for e in result.errors)

    @pytest.mark.asyncio
    async def test_invalid_uc_identifier(self) -> None:
        # UC catalog starting with a digit must be rejected.
        translator = MlflowAgentTranslator()
        agent, revision = _agent_revision()
        result = await translator.validate(
            agent, revision, _valid_config(uc_catalog="9badname")
        )
        assert result.valid is False
        assert any("Unity Catalog identifier" in e.message for e in result.errors)

    @pytest.mark.asyncio
    async def test_invalid_endpoint_override(self) -> None:
        translator = MlflowAgentTranslator()
        agent, revision = _agent_revision()
        result = await translator.validate(
            agent, revision, _valid_config(endpoint_name="custom-endpoint")
        )
        assert result.valid is False
        assert any("dr-agent-" in e.message for e in result.errors)

    @pytest.mark.asyncio
    async def test_valid_endpoint_override_with_prefix(self) -> None:
        translator = MlflowAgentTranslator()
        agent, revision = _agent_revision()
        result = await translator.validate(
            agent, revision, _valid_config(endpoint_name="dr-agent-research")
        )
        assert result.valid is True


class TestTranslate:
    @pytest.mark.asyncio
    async def test_writes_workflow_definition_to_temp_file(self) -> None:
        translator = MlflowAgentTranslator()
        agent, revision = _agent_revision()
        revision.definition["marker"] = "TRANSLATE_MARKER"

        artifact = await translator.translate(agent, revision, _valid_config())
        assert isinstance(artifact, Artifact)
        assert artifact.mode == DeploymentMode.MLFLOW_AGENT

        path = artifact.payload["workflow_definition_path"]
        with open(path, encoding="utf-8") as f:
            written = json.load(f)
        assert written["marker"] == "TRANSLATE_MARKER"

    @pytest.mark.asyncio
    async def test_artifact_metadata_uc_uri_and_git_tag(self) -> None:
        translator = MlflowAgentTranslator()
        agent, revision = _agent_revision()
        artifact = await translator.translate(
            agent,
            revision,
            _valid_config(framework_git_tag="v9.9.9"),
        )
        assert artifact.metadata["uc_model_uri"] == "main.agents.deep_research"
        assert artifact.metadata["framework_git_tag"] == "v9.9.9"

    @pytest.mark.asyncio
    async def test_pip_requirements_pin_supplied_git_tag(self) -> None:
        translator = MlflowAgentTranslator()
        agent, revision = _agent_revision()
        artifact = await translator.translate(
            agent, revision, _valid_config(framework_git_tag="v0.4.2")
        )
        pin = next(
            r for r in artifact.payload["pip_requirements"] if "git+https" in r
        )
        assert "v0.4.2" in pin
        assert "subdirectory=databricks-deep-research" in pin

    @pytest.mark.asyncio
    async def test_endpoint_name_override_carried_through_artifact(self) -> None:
        """W8: ``endpoint_name`` from the wizard must reach the artifact so
        ``_deploy_via_sdk`` can forward it to ``agents.deploy()``. Previously
        validated and then silently dropped.
        """
        translator = MlflowAgentTranslator()
        agent, revision = _agent_revision()
        artifact = await translator.translate(
            agent,
            revision,
            _valid_config(endpoint_name="dr-agent-research"),
        )
        assert artifact.payload["endpoint_name_override"] == "dr-agent-research"
        assert artifact.metadata["endpoint_name_override"] == "dr-agent-research"

    @pytest.mark.asyncio
    async def test_env_overrides_carried_through_artifact(self) -> None:
        """W8: ``env_overrides`` from the wizard must reach the artifact."""
        translator = MlflowAgentTranslator()
        agent, revision = _agent_revision()
        artifact = await translator.translate(
            agent,
            revision,
            _valid_config(env_overrides={"FOO": "bar", "BAZ": "qux"}),
        )
        assert artifact.payload["env_overrides"] == {"FOO": "bar", "BAZ": "qux"}

    @pytest.mark.asyncio
    async def test_no_endpoint_name_override_leaves_field_none(self) -> None:
        """When the wizard omits the field the artifact records None."""
        translator = MlflowAgentTranslator()
        agent, revision = _agent_revision()
        artifact = await translator.translate(agent, revision, _valid_config())
        assert artifact.payload["endpoint_name_override"] is None


class TestDeploy:
    @pytest.mark.asyncio
    async def test_success_path_calls_sdk_helper(self) -> None:
        translator = MlflowAgentTranslator()
        agent, revision = _agent_revision()
        artifact = await translator.translate(agent, revision, _valid_config())
        deployment = MagicMock(spec=AgentDeployment, id=uuid4())

        fake_result = DeploymentResult(
            success=True,
            endpoint_name="dr-agent-deep-research",
            model_name="main.agents.deep_research",
            external_resource_ids={
                "uc_model": "main.agents.deep_research",
                "model_version": "1",
            },
        )
        with patch(
            "deep_research.services.deployment.mlflow_deploy._deploy_via_sdk",
            return_value=fake_result,
        ) as mock_sdk:
            result = await translator.deploy(artifact, _valid_config(), deployment)

        assert result.success is True
        assert result.endpoint_name == "dr-agent-deep-research"
        assert result.external_resource_ids["uc_model"] == "main.agents.deep_research"
        mock_sdk.assert_called_once()

    @pytest.mark.asyncio
    async def test_failure_path_surfaces_error_message(self) -> None:
        translator = MlflowAgentTranslator()
        agent, revision = _agent_revision()
        artifact = await translator.translate(agent, revision, _valid_config())
        deployment = MagicMock(spec=AgentDeployment, id=uuid4())

        with patch(
            "deep_research.services.deployment.mlflow_deploy._deploy_via_sdk",
            side_effect=RuntimeError("UC schema permission denied"),
        ):
            result = await translator.deploy(artifact, _valid_config(), deployment)

        assert result.success is False
        assert result.error_message is not None
        assert "UC schema permission denied" in result.error_message

    @pytest.mark.asyncio
    async def test_deploy_via_sdk_forwards_endpoint_name_and_env_overrides(
        self,
    ) -> None:
        """W8: ``_deploy_via_sdk`` MUST forward ``endpoint_name_override`` and
        ``env_overrides`` to ``agents.deploy()`` as kwargs. The previous code
        validated these in the wizard then silently dropped them.
        """
        from deep_research.services.deployment.mlflow_deploy import (
            _deploy_via_sdk,
        )

        # Build a minimal valid payload via the public translate() so we
        # exercise the full wiring (translate → payload → deploy).
        translator = MlflowAgentTranslator()
        agent, revision = _agent_revision()
        artifact = await translator.translate(
            agent,
            revision,
            _valid_config(
                endpoint_name="dr-agent-custom",
                env_overrides={"FEATURE_X": "1"},
            ),
        )
        deployment = MagicMock(spec=AgentDeployment, id=uuid4())

        fake_log_info = MagicMock(model_uri="models:/main.agents.deep_research/1")
        fake_register_result = MagicMock(version=7)
        fake_deploy_result = MagicMock(
            endpoint_name="dr-agent-custom", app_name="dr-agent-custom"
        )

        fake_mlflow = MagicMock()
        fake_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        fake_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        fake_mlflow.pyfunc.log_model.return_value = fake_log_info
        fake_mlflow.register_model.return_value = fake_register_result

        fake_agents = MagicMock()
        fake_agents.deploy.return_value = fake_deploy_result
        fake_databricks = MagicMock()
        fake_databricks.agents = fake_agents

        with patch.dict(
            "sys.modules",
            {
                "databricks": fake_databricks,
                "databricks.agents": fake_agents,
                "mlflow": fake_mlflow,
                "mlflow.pyfunc": fake_mlflow.pyfunc,
            },
        ):
            result = _deploy_via_sdk(artifact, _valid_config(), deployment)

        # The kwargs must reach the SDK.
        fake_agents.deploy.assert_called_once()
        args, kwargs = fake_agents.deploy.call_args
        assert args == ("main.agents.deep_research", "7")
        assert kwargs["endpoint_name"] == "dr-agent-custom"
        assert kwargs["environment_vars"] == {"FEATURE_X": "1"}
        assert result.success is True
        assert (
            result.external_resource_ids["endpoint_name_override"]
            == "dr-agent-custom"
        )

    @pytest.mark.asyncio
    async def test_deploy_via_sdk_omits_kwargs_when_overrides_unset(self) -> None:
        """When the wizard leaves the optional fields blank, no kwargs are
        passed — preserves compatibility with older databricks-agents SDKs
        that don't accept ``endpoint_name`` / ``environment_vars``.
        """
        from deep_research.services.deployment.mlflow_deploy import (
            _deploy_via_sdk,
        )

        translator = MlflowAgentTranslator()
        agent, revision = _agent_revision()
        artifact = await translator.translate(agent, revision, _valid_config())
        deployment = MagicMock(spec=AgentDeployment, id=uuid4())

        fake_log_info = MagicMock(model_uri="models:/main.agents.deep_research/1")
        fake_register_result = MagicMock(version=1)
        fake_deploy_result = MagicMock(endpoint_name=None, app_name=None)

        fake_mlflow = MagicMock()
        fake_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        fake_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        fake_mlflow.pyfunc.log_model.return_value = fake_log_info
        fake_mlflow.register_model.return_value = fake_register_result

        fake_agents = MagicMock()
        fake_agents.deploy.return_value = fake_deploy_result
        fake_databricks = MagicMock()
        fake_databricks.agents = fake_agents

        with patch.dict(
            "sys.modules",
            {
                "databricks": fake_databricks,
                "databricks.agents": fake_agents,
                "mlflow": fake_mlflow,
                "mlflow.pyfunc": fake_mlflow.pyfunc,
            },
        ):
            _deploy_via_sdk(artifact, _valid_config(), deployment)

        _args, kwargs = fake_agents.deploy.call_args
        assert "endpoint_name" not in kwargs
        assert "environment_vars" not in kwargs

    @pytest.mark.asyncio
    async def test_rejects_non_dict_payload(self) -> None:
        translator = MlflowAgentTranslator()
        # Construct an Artifact whose payload is bytes (wrong shape for this mode).
        bad = Artifact(mode=DeploymentMode.MLFLOW_AGENT, payload=b"not a dict")
        deployment = MagicMock(spec=AgentDeployment)
        result = await translator.deploy(bad, _valid_config(), deployment)
        assert result.success is False
        assert "must be a dict" in (result.error_message or "")


class TestDeactivate:
    @pytest.mark.asyncio
    async def test_deactivate_calls_sdk_helper(self) -> None:
        translator = MlflowAgentTranslator()
        deployment = MagicMock(spec=AgentDeployment, id=uuid4())
        with patch(
            "deep_research.services.deployment.mlflow_deploy._deactivate_via_sdk"
        ) as mock_sdk:
            await translator.deactivate(deployment)
        mock_sdk.assert_called_once_with(deployment)

    @pytest.mark.asyncio
    async def test_deactivate_propagates_cleanup_errors(self) -> None:
        """W4: genuine upstream failures must NOT be swallowed.

        Previously (pre-W4) the translator caught all exceptions and logged
        warnings, masking real cleanup failures as success. The new
        contract is: 404/NotFound → success (idempotent); any other error
        → raise ``DeploymentCleanupError`` so the API layer can promote
        the row to ``cleanup_failed`` after the retry threshold.
        """
        from deep_research.services.deployment.translator import (
            DeploymentCleanupError,
        )

        translator = MlflowAgentTranslator()
        deployment = MagicMock(spec=AgentDeployment, id=uuid4())
        with patch(
            "deep_research.services.deployment.mlflow_deploy._deactivate_via_sdk",
            side_effect=DeploymentCleanupError(
                "boom", resource="agents.delete_deployment"
            ),
        ), pytest.raises(DeploymentCleanupError):
            await translator.deactivate(deployment)

    @pytest.mark.asyncio
    async def test_deactivate_via_sdk_treats_not_found_as_success(self) -> None:
        """W4: a 404 / NotFound from the upstream SDK must NOT raise."""
        from deep_research.services.deployment.mlflow_deploy import (
            _deactivate_via_sdk,
        )

        deployment = MagicMock(spec=AgentDeployment, id=uuid4())
        deployment.external_resource_ids = {
            "uc_model": "cat.sch.foo",
            "model_version": "1",
        }

        class _NotFoundError(Exception):
            pass

        fake_agents = MagicMock()
        fake_agents.delete_deployment.side_effect = _NotFoundError("404 not found")
        fake_mlflow_client = MagicMock()
        fake_mlflow_client.transition_model_version_stage.side_effect = (
            _NotFoundError("not found")
        )
        fake_mlflow = MagicMock()
        fake_mlflow.MlflowClient.return_value = fake_mlflow_client

        fake_databricks = MagicMock()
        fake_databricks.agents = fake_agents

        with patch.dict(
            "sys.modules",
            {
                "databricks": fake_databricks,
                "databricks.agents": fake_agents,
                "mlflow": fake_mlflow,
            },
        ):
            # Should NOT raise — 404 is idempotent success.
            _deactivate_via_sdk(deployment)

    @pytest.mark.asyncio
    async def test_deactivate_via_sdk_raises_on_real_failure(self) -> None:
        """W4: a genuine non-404 failure must raise DeploymentCleanupError."""
        from deep_research.services.deployment.mlflow_deploy import (
            _deactivate_via_sdk,
        )
        from deep_research.services.deployment.translator import (
            DeploymentCleanupError,
        )

        deployment = MagicMock(spec=AgentDeployment, id=uuid4())
        deployment.external_resource_ids = {
            "uc_model": "cat.sch.foo",
            "model_version": "1",
        }

        fake_agents = MagicMock()
        fake_agents.delete_deployment.side_effect = RuntimeError(
            "503 Service Unavailable"
        )
        fake_mlflow_client = MagicMock()
        # archive succeeds — only one failure to report.
        fake_mlflow = MagicMock()
        fake_mlflow.MlflowClient.return_value = fake_mlflow_client

        fake_databricks = MagicMock()
        fake_databricks.agents = fake_agents

        with patch.dict(
            "sys.modules",
            {
                "databricks": fake_databricks,
                "databricks.agents": fake_agents,
                "mlflow": fake_mlflow,
            },
        ), pytest.raises(DeploymentCleanupError) as exc_info:
            _deactivate_via_sdk(deployment)
        assert "agents.delete_deployment" in str(exc_info.value)
