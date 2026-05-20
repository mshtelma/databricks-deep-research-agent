"""Unit tests for deployment Pydantic schemas (US-102).

Covers:
  - discriminated-union dispatch (each mode)
  - invalid mode rejected
  - extra='forbid' rejection
  - JSON round-trip
  - prefix-pattern validators on app names / endpoint names
"""
from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import TypeAdapter, ValidationError

from deep_research.models.agent_deployment import DeploymentMode
from deep_research.schemas.deployment import (
    BatchDeploymentConfig,
    CreateDeploymentRequest,
    DeploymentConfig,
    InAppDeploymentConfig,
    MlflowAgentDeploymentConfig,
    ShellAppDeploymentConfig,
)

_DC_ADAPTER: TypeAdapter[DeploymentConfig] = TypeAdapter(DeploymentConfig)


class TestDiscriminatedUnion:
    def test_in_app_config_dispatches(self) -> None:
        cfg = _DC_ADAPTER.validate_python({"mode": "in_app"})
        assert isinstance(cfg, InAppDeploymentConfig)
        assert cfg.mode == DeploymentMode.IN_APP

    def test_shell_app_config_dispatches(self) -> None:
        cfg = _DC_ADAPTER.validate_python(
            {
                "mode": "shell_app",
                "app_name": "dr-shell-research",
                "framework_git_tag": "v0.3.0",
            }
        )
        assert isinstance(cfg, ShellAppDeploymentConfig)
        assert cfg.app_name == "dr-shell-research"
        assert cfg.target == "dev"  # default

    def test_mlflow_agent_config_dispatches(self) -> None:
        cfg = _DC_ADAPTER.validate_python(
            {
                "mode": "mlflow_agent",
                "uc_catalog": "main",
                "uc_schema": "agents",
                "uc_model_name": "deep_research",
            }
        )
        assert isinstance(cfg, MlflowAgentDeploymentConfig)
        assert cfg.uc_catalog == "main"
        assert cfg.endpoint_name is None  # auto-generated when omitted

    def test_batch_config_dispatches(self) -> None:
        cfg = _DC_ADAPTER.validate_python(
            {
                "mode": "batch",
                "target_endpoint": "databricks-claude-sonnet-4-5",
                "input_table": "main.research.queries",
                "output_table": "main.research.results",
                "prompt_column": "query",
            }
        )
        assert isinstance(cfg, BatchDeploymentConfig)
        assert cfg.response_format is None  # optional


class TestInvalidInputs:
    def test_unknown_mode_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _DC_ADAPTER.validate_python({"mode": "totally_made_up"})

    def test_extra_keys_forbidden_on_in_app(self) -> None:
        with pytest.raises(ValidationError):
            InAppDeploymentConfig.model_validate(
                {"mode": "in_app", "extra_key": "boom"}
            )

    def test_extra_keys_forbidden_on_shell_app(self) -> None:
        with pytest.raises(ValidationError):
            _DC_ADAPTER.validate_python(
                {
                    "mode": "shell_app",
                    "app_name": "dr-shell-foo",
                    "framework_git_tag": "v0.3.0",
                    "rogue": True,
                }
            )

    def test_shell_app_name_must_match_prefix(self) -> None:
        with pytest.raises(ValidationError):
            _DC_ADAPTER.validate_python(
                {
                    "mode": "shell_app",
                    "app_name": "my-app",  # missing dr-shell- prefix
                    "framework_git_tag": "v0.3.0",
                }
            )

    def test_shell_app_name_must_fit_databricks_apps_limit(self) -> None:
        with pytest.raises(ValidationError):
            _DC_ADAPTER.validate_python(
                {
                    "mode": "shell_app",
                    "app_name": "dr-shell-a528c6b0-59bb-4215-87b4-9de5116276c8",
                    "framework_git_tag": "v0.3.0",
                }
            )

    def test_mlflow_endpoint_name_must_match_prefix_when_set(self) -> None:
        with pytest.raises(ValidationError):
            _DC_ADAPTER.validate_python(
                {
                    "mode": "mlflow_agent",
                    "uc_catalog": "main",
                    "uc_schema": "agents",
                    "uc_model_name": "deep_research",
                    "endpoint_name": "custom-name",  # missing dr-agent- prefix
                }
            )


class TestBatchConfigIdentifierValidation:
    """W16: Mode 4 input fields must be SQL-identifier-safe at schema time.

    Pre-W16 the schema only enforced non-empty strings; raw user input went
    straight into the SQL template render, producing malformed/injectable
    artifacts. These tests pin the new rejection set.
    """

    def _batch(self, **overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "mode": "batch",
            "target_endpoint": "databricks-claude-sonnet-4-5",
            "input_table": "main.research.queries",
            "output_table": "main.research.results",
            "prompt_column": "query",
        }
        base.update(overrides)
        return base

    def test_endpoint_with_disallowed_chars_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _DC_ADAPTER.validate_python(
                self._batch(target_endpoint="bad endpoint; DROP TABLE")
            )

    def test_input_table_must_be_three_level(self) -> None:
        with pytest.raises(ValidationError):
            _DC_ADAPTER.validate_python(
                self._batch(input_table="main.research")  # 2-level
            )

    def test_output_table_must_be_three_level(self) -> None:
        with pytest.raises(ValidationError):
            _DC_ADAPTER.validate_python(
                self._batch(output_table="main")  # 1-level
            )

    def test_table_segment_with_disallowed_chars_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _DC_ADAPTER.validate_python(
                self._batch(
                    input_table="main.research.q`uote",  # backtick injection
                )
            )

    def test_prompt_column_must_be_sql_identifier(self) -> None:
        with pytest.raises(ValidationError):
            _DC_ADAPTER.validate_python(
                self._batch(prompt_column="user query")  # space disallowed
            )

    def test_prompt_column_cannot_start_with_digit(self) -> None:
        with pytest.raises(ValidationError):
            _DC_ADAPTER.validate_python(self._batch(prompt_column="1col"))

    def test_well_formed_inputs_accepted(self) -> None:
        cfg = _DC_ADAPTER.validate_python(
            self._batch(
                target_endpoint="my-endpoint-name",
                input_table="cat.sch.tbl_in",
                output_table="cat.sch.tbl_out",
                prompt_column="user_query",
            )
        )
        assert isinstance(cfg, BatchDeploymentConfig)


class TestRoundTrip:
    def test_in_app_round_trip(self) -> None:
        original = InAppDeploymentConfig()
        revived = _DC_ADAPTER.validate_python(original.model_dump())
        assert revived == original

    def test_create_deployment_request_round_trip(self) -> None:
        agent_id = uuid4()
        rev_id = uuid4()
        req = CreateDeploymentRequest(
            agent_id=agent_id,
            revision_id=rev_id,
            config=InAppDeploymentConfig(),
        )
        as_json = req.model_dump_json()
        assert "in_app" in as_json
        revived = CreateDeploymentRequest.model_validate_json(as_json)
        assert revived.agent_id == agent_id
        assert revived.revision_id == rev_id
        assert isinstance(revived.config, InAppDeploymentConfig)
