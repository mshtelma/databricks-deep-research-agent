"""BatchTranslator: Lakeflow Declarative Pipeline + SQL via ai_query (Mode 4).

This is the Phase 2 answer to the user's "Lakeflow pipelines + SQL or Spark?"
question (plan Section F.0): **Lakeflow + SQL is the primary path; no raw
Spark/PySpark is needed.** ai_query handles the per-row agent invocation;
Lakeflow gives us incremental processing, expectations, lineage, and the new
Data Engineering IDE for free.

Phase 2 ships the SQL artifact only -- BatchTranslator.translate() returns
the rendered SQL bytes embedded in an Artifact. The actual Lakeflow pipeline
creation via the Databricks REST API (resources.pipelines.<name>) is a
Phase 3 follow-up; the API layer caller can use the artifact today by
pasting it into a notebook or pipeline manually.

Critical constraint (plan Section F.2 -- the #1 documented gotcha):
``pipelines.channel = 'preview'`` MUST appear in BOTH the SQL TBLPROPERTIES
and the DAB ``channel: PREVIEW`` field. Without preview, ai_query fails with
"function not found".

Decoupled from Mode 3 (plan Section F.1): ``target_endpoint`` accepts ANY
Databricks serving endpoint name -- a Mode 3 deployment, the 8 pre-bound
endpoints in databricks.yml, or any other endpoint with CAN_QUERY for the
pipeline owner / RUN_AS service principal.

OBO note (plan Section F.4): Lakeflow pipelines do NOT support OBO; they run
as the pipeline owner / RUN_AS SP. The owner must have CAN_QUERY on the
target endpoint at deploy time. The API-layer pre-flight check enforces this.
"""
# Method args (agent, revision, deployment) carry context required by the
# DeploymentTranslator Protocol but are not all used by every method here --
# Mode 4 reads only `config` for SQL render, and ignores the deployment row
# in deploy() (the row is recorded by the API layer after success).
# ruff: noqa: ARG002
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from deep_research.models.agent_deployment import AgentDeployment, DeploymentMode
from deep_research.services.deployment._paths import resolve_package_data_dir
from deep_research.services.deployment.translator import (
    Artifact,
    DeploymentResult,
    ValidationError,
    ValidationResult,
)

if TYPE_CHECKING:
    from deep_research.services.deployment.auth import WorkspaceClientResolver

# The SQL template ships as package data. In a wheel install it lives at
# ``site-packages/deep_research/services/deployment/templates/spark-batch/``;
# in the source tree it lives at ``<app>/templates/spark-batch/``. The
# shared resolver in _paths.py picks the right location.
_BATCH_TEMPLATE_DIR = resolve_package_data_dir(Path(__file__), "spark-batch")
_TEMPLATE_PATH = _BATCH_TEMPLATE_DIR / "batch_inference.sql"


# W16: identifier validation regexes. The Pydantic schema enforces these
# at the API layer; the translator re-validates as defense in depth so a
# direct caller (or future test fixture that bypasses the schema) cannot
# render unsafe SQL.
_UC_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
_SQL_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote_table(uc_name: str) -> str:
    """Render a 3-level UC name with each segment backtick-quoted.

    Pre-W16 the template interpolated raw user input directly into SQL,
    leaving the artifact vulnerable to identifier injection or malformed
    rendering. Backtick-quoting forces the parser to treat each segment
    as an identifier and rejects strange characters that snuck past
    validation.
    """
    segments = uc_name.split(".")
    if len(segments) != 3:
        raise ValueError(
            f"UC table name must be 3-level (catalog.schema.table); got {uc_name!r}"
        )
    for seg in segments:
        if not _UC_IDENT_RE.fullmatch(seg):
            raise ValueError(
                f"UC identifier segment {seg!r} contains disallowed characters"
            )
    return ".".join(f"`{seg}`" for seg in segments)


def _quote_column(column: str) -> str:
    """Backtick-quote a single SQL identifier (column name)."""
    if not _SQL_IDENT_RE.fullmatch(column):
        raise ValueError(
            f"SQL identifier {column!r} contains disallowed characters"
        )
    return f"`{column}`"


def _escape_sql_string_literal(value: str) -> str:
    """Escape a string for safe insertion inside SQL single-quoted literal.

    Doubles single quotes (the SQL convention) and rejects raw backslashes
    + newlines that could otherwise terminate the literal in some dialects.
    Used for the response_format JSON serialization.
    """
    if "\x00" in value:
        raise ValueError("NUL byte not permitted in SQL string literal")
    return value.replace("'", "''")


def _format_response_format_clause(response_format: dict[str, Any] | None) -> str:
    """Render the optional ``responseFormat =>`` clause for ai_query.

    Returns an empty string when response_format is None (clause omitted)
    and a leading-comma + ``responseFormat => '<json>'`` clause otherwise.

    The leading comma is required because the clause is inserted between
    the prompt column reference and ``failOnError =>`` in the rendered SQL.

    W16: the serialized JSON is SQL-escaped before splice so keys/values
    containing single quotes can't break out of the string literal.
    """
    if response_format is None:
        return ""
    serialized = json.dumps(response_format, sort_keys=True)
    return f",\n    responseFormat => '{_escape_sql_string_literal(serialized)}'"


class BatchTranslator:
    """Translator for ``DeploymentMode.BATCH`` (Lakeflow + SQL via ai_query).

    Phase 2 deliverable. Does NOT create a live pipeline yet; the API layer
    persists the rendered artifact in ``external_resource_ids`` for the user
    to inspect or for a Phase 3 deploy() to upgrade in-place.
    """

    mode: ClassVar[DeploymentMode] = DeploymentMode.BATCH

    async def validate(
        self,
        agent: Any,
        revision: Any,
        config: dict[str, Any],
    ) -> ValidationResult:
        """Reject empty required fields + invalid SQL identifiers.

        Endpoint reachability + UC permissions are intentionally checked by
        the API layer's slow-path probe, not here. W16 adds identifier-shape
        validation as defense in depth — the Pydantic schema validates at
        the API boundary, this validates at the translator boundary.
        """
        errors: list[ValidationError] = []
        for required in ("target_endpoint", "input_table", "output_table", "prompt_column"):
            value = config.get(required, "")
            if not isinstance(value, str) or not value.strip():
                errors.append(
                    ValidationError(
                        message=f"{required} is required and must be a non-empty string",
                        path=f"config.{required}",
                    )
                )
        if errors:
            return ValidationResult(valid=False, errors=errors)

        # Shape validation (matches schemas/deployment.py rules).
        endpoint = config["target_endpoint"]
        if not _UC_IDENT_RE.fullmatch(endpoint):
            errors.append(
                ValidationError(
                    message="target_endpoint contains characters outside the serving endpoint identifier rule",
                    path="config.target_endpoint",
                )
            )
        for tbl_field in ("input_table", "output_table"):
            try:
                _quote_table(config[tbl_field])
            except ValueError as exc:
                errors.append(
                    ValidationError(message=str(exc), path=f"config.{tbl_field}")
                )
        try:
            _quote_column(config["prompt_column"])
        except ValueError as exc:
            errors.append(
                ValidationError(message=str(exc), path="config.prompt_column")
            )
        return ValidationResult(valid=not errors, errors=errors)

    async def translate(
        self,
        agent: Any,
        revision: Any,
        config: dict[str, Any],
    ) -> Artifact:
        """Render the SQL template and return it as an artifact.

        The artifact's payload is the rendered SQL bytes (UTF-8); metadata
        carries the pipeline channel for downstream consumers. W16: each
        identifier is backtick-quoted in the rendered SQL and the
        response_format JSON is SQL-escaped before splice — together with
        Pydantic-level validation, the artifact is safe by construction.
        """
        template = _TEMPLATE_PATH.read_text(encoding="utf-8")
        rendered = template.format(
            endpoint_name=config["target_endpoint"],
            input_table_quoted=_quote_table(config["input_table"]),
            output_table_quoted=_quote_table(config["output_table"]),
            prompt_column_quoted=_quote_column(config["prompt_column"]),
            response_format_clause=_format_response_format_clause(
                config.get("response_format")
            ),
        )
        return Artifact(
            mode=DeploymentMode.BATCH,
            payload=rendered.encode("utf-8"),
            metadata={
                "pipeline_channel": "preview",
                "agent_id": str(agent.id),
                "revision_id": str(revision.rev_id),
            },
        )

    async def deploy(
        self,
        artifact: Artifact,
        config: dict[str, Any],
        deployment: AgentDeployment,
    ) -> DeploymentResult:
        """Phase 2 stub: record the SHA256 of the SQL artifact, no pipeline yet.

        Phase 3 will replace this body with a Databricks REST call to create
        a Lakeflow pipeline + Job. For Phase 2, the API caller can fetch the
        deployment row, decode ``external_resource_ids['sql_artifact_sha256']``,
        and reproduce the artifact via translate() for manual paste-deploy.
        """
        if not isinstance(artifact.payload, bytes):
            return DeploymentResult(
                success=False,
                error_message="BatchTranslator artifact payload must be bytes",
            )
        digest = hashlib.sha256(artifact.payload).hexdigest()
        return DeploymentResult(
            success=True,
            external_resource_ids={
                "sql_artifact_sha256": digest,
                "sql_artifact_size_bytes": str(len(artifact.payload)),
                "pipeline_channel": "preview",
            },
        )

    async def deactivate(
        self,
        deployment: AgentDeployment,
        *,
        client_resolver: WorkspaceClientResolver | None = None,
    ) -> None:
        """No-op for Phase 2 (no live pipeline created).

        Phase 3 will pause/delete the Databricks Workflow + Pipeline here.
        Idempotent by definition. ``client_resolver`` is accepted for
        Protocol conformance and reserved for the Phase 3 implementation,
        where the user's OBO client should run the pipeline teardown.
        """
        return None
