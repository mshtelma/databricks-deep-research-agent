"""Unit tests for BatchTranslator (US-202).

Plan reference: agent-designer-deployment.md Section F (Lakeflow + SQL).

Critical contract verified:
  - ``pipelines.channel = 'preview'`` literal MUST appear in rendered SQL
    (without it ai_query fails with "function not found" -- the #1 documented
    gotcha per plan Section F.2).
  - ``responseFormat =>`` clause emitted ONLY when config.response_format is set.
  - ``failOnError => false`` always emitted (per-row failures surface in
    errorMessage column instead of aborting the pipeline).
"""
from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from deep_research.models.agent_deployment import AgentDeployment, DeploymentMode
from deep_research.services.deployment import (
    Artifact,
    BatchTranslator,
    DeploymentResult,
    DeploymentTranslator,
    ValidationResult,
)


def _valid_config(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "mode": "batch",
        "target_endpoint": "databricks-claude-sonnet-4-5",
        "input_table": "main.research.queries",
        "output_table": "main.research.results",
        "prompt_column": "query",
        "response_format": None,
    }
    base.update(overrides)
    return base


def _agent_revision() -> tuple[MagicMock, MagicMock]:
    return MagicMock(id=uuid4()), MagicMock(rev_id=uuid4())


class TestProtocolConformance:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(BatchTranslator(), DeploymentTranslator)

    def test_mode_classvar(self) -> None:
        assert BatchTranslator.mode == DeploymentMode.BATCH


class TestValidate:
    @pytest.mark.asyncio
    async def test_valid_when_all_required_fields_present(self) -> None:
        translator = BatchTranslator()
        agent, revision = _agent_revision()
        result = await translator.validate(agent, revision, _valid_config())
        assert isinstance(result, ValidationResult)
        assert result.valid is True
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_invalid_when_target_endpoint_empty(self) -> None:
        translator = BatchTranslator()
        agent, revision = _agent_revision()
        result = await translator.validate(
            agent, revision, _valid_config(target_endpoint="")
        )
        assert result.valid is False
        assert any("target_endpoint" in e.message for e in result.errors)

    @pytest.mark.asyncio
    async def test_invalid_when_input_table_missing(self) -> None:
        translator = BatchTranslator()
        agent, revision = _agent_revision()
        cfg = _valid_config()
        del cfg["input_table"]
        result = await translator.validate(agent, revision, cfg)
        assert result.valid is False
        assert any("input_table" in e.message for e in result.errors)

    @pytest.mark.asyncio
    async def test_invalid_when_prompt_column_whitespace(self) -> None:
        translator = BatchTranslator()
        agent, revision = _agent_revision()
        result = await translator.validate(
            agent, revision, _valid_config(prompt_column="   ")
        )
        assert result.valid is False
        assert any("prompt_column" in e.message for e in result.errors)


class TestTranslate:
    @pytest.mark.asyncio
    async def test_renders_preview_channel(self) -> None:
        translator = BatchTranslator()
        agent, revision = _agent_revision()
        artifact = await translator.translate(agent, revision, _valid_config())
        assert isinstance(artifact, Artifact)
        sql = artifact.payload.decode("utf-8")
        # The #1 documented gotcha: without preview, ai_query is unresolved.
        assert "pipelines.channel = 'preview'" in sql

    @pytest.mark.asyncio
    async def test_substitutes_endpoint_and_tables(self) -> None:
        translator = BatchTranslator()
        agent, revision = _agent_revision()
        artifact = await translator.translate(
            agent,
            revision,
            _valid_config(
                target_endpoint="my-custom-endpoint",
                input_table="catalog.schema.in_tbl",
                output_table="catalog.schema.out_tbl",
                prompt_column="user_query",
            ),
        )
        sql = artifact.payload.decode("utf-8")
        # Endpoint stays a string literal (single-quoted) per ai_query
        # contract; W16 backtick-quotes table + column identifiers in the
        # rendered SQL as defense in depth against identifier injection.
        assert "'my-custom-endpoint'" in sql
        assert "`catalog`.`schema`.`in_tbl`" in sql
        assert "`catalog`.`schema`.`out_tbl`" in sql
        assert "`user_query`" in sql

    @pytest.mark.asyncio
    async def test_response_format_omitted_when_none(self) -> None:
        translator = BatchTranslator()
        agent, revision = _agent_revision()
        artifact = await translator.translate(agent, revision, _valid_config())
        sql = artifact.payload.decode("utf-8")
        assert "responseFormat" not in sql

    @pytest.mark.asyncio
    async def test_response_format_emitted_when_set(self) -> None:
        translator = BatchTranslator()
        agent, revision = _agent_revision()
        artifact = await translator.translate(
            agent,
            revision,
            _valid_config(
                response_format={"type": "json_schema", "schema": {"type": "object"}}
            ),
        )
        sql = artifact.payload.decode("utf-8")
        assert "responseFormat =>" in sql
        assert '"type": "json_schema"' in sql

    @pytest.mark.asyncio
    async def test_response_format_escapes_single_quotes(self) -> None:
        """W16: response_format JSON containing single quotes must NOT break out
        of the SQL string literal — escape ``'`` to ``''`` per SQL convention.
        """
        translator = BatchTranslator()
        agent, revision = _agent_revision()
        artifact = await translator.translate(
            agent,
            revision,
            _valid_config(
                # The JSON-serialized form will include a single-quote char
                # because the key value contains one.
                response_format={"description": "user's input"}
            ),
        )
        sql = artifact.payload.decode("utf-8")
        # The single quote inside the JSON value must appear DOUBLED inside
        # the SQL string literal — verifying the escape applied.
        assert "user''s input" in sql
        # Sanity-check: there's no raw ``user's input`` (unescaped) that
        # would break out of the surrounding single-quoted SQL literal.
        assert "user's input" not in sql

    @pytest.mark.asyncio
    async def test_translate_rejects_bad_table_identifier(self) -> None:
        """W16 defense-in-depth: the translator re-validates identifiers
        even when bypassing the Pydantic schema.
        """
        from deep_research.services.deployment.batch import _quote_table

        with pytest.raises(ValueError, match="UC"):
            _quote_table("catalog.schema.bad';DROP TABLE--")

    @pytest.mark.asyncio
    async def test_translate_rejects_bad_column_identifier(self) -> None:
        from deep_research.services.deployment.batch import _quote_column

        with pytest.raises(ValueError, match="SQL identifier"):
            _quote_column("col`hack")

    @pytest.mark.asyncio
    async def test_fail_on_error_always_false(self) -> None:
        translator = BatchTranslator()
        agent, revision = _agent_revision()
        artifact = await translator.translate(agent, revision, _valid_config())
        sql = artifact.payload.decode("utf-8")
        assert "failOnError => false" in sql

    @pytest.mark.asyncio
    async def test_artifact_metadata(self) -> None:
        translator = BatchTranslator()
        agent, revision = _agent_revision()
        artifact = await translator.translate(agent, revision, _valid_config())
        assert artifact.metadata["pipeline_channel"] == "preview"
        assert artifact.metadata["agent_id"] == str(agent.id)
        assert artifact.metadata["revision_id"] == str(revision.rev_id)


class TestDeploy:
    @pytest.mark.asyncio
    async def test_deploy_records_sha256(self) -> None:
        translator = BatchTranslator()
        agent, revision = _agent_revision()
        artifact = await translator.translate(agent, revision, _valid_config())
        deployment = MagicMock(spec=AgentDeployment)
        result = await translator.deploy(artifact, _valid_config(), deployment)
        assert isinstance(result, DeploymentResult)
        assert result.success is True
        assert "sql_artifact_sha256" in result.external_resource_ids
        # SHA256 is 64 hex chars.
        assert len(result.external_resource_ids["sql_artifact_sha256"]) == 64
        assert result.external_resource_ids["pipeline_channel"] == "preview"

    @pytest.mark.asyncio
    async def test_deploy_fails_on_non_bytes_payload(self) -> None:
        translator = BatchTranslator()
        deployment = MagicMock(spec=AgentDeployment)
        broken = Artifact(mode=DeploymentMode.BATCH, payload={"not": "bytes"})
        result = await translator.deploy(broken, _valid_config(), deployment)
        assert result.success is False
        assert result.error_message is not None


class TestDeactivate:
    @pytest.mark.asyncio
    async def test_deactivate_is_noop(self) -> None:
        translator = BatchTranslator()
        deployment = MagicMock(spec=AgentDeployment)
        # Should complete without raising; idempotent.
        result = await translator.deactivate(deployment)
        assert result is None
        # Calling twice must remain safe (Phase 3 will replace this body).
        result2 = await translator.deactivate(deployment)
        assert result2 is None
