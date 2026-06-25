"""Tests for scored role inference (DISCOVERED bindings)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from databricks_deep_research.tools.builtins.text_table.error_codes import (
    ErrorCode,
    ToolErrorException,
)
from databricks_deep_research.tools.builtins.text_table.role_inference import (
    RoleCandidate,
    infer_roles,
)
from databricks_deep_research.tools.builtins.text_table.schema_cache import (
    Schema,
    SchemaColumn,
)


def _officeqa_schema() -> Schema:
    return Schema(
        fqn="cat.s.tbl",
        columns=(
            SchemaColumn("chunk_id", "STRING"),
            SchemaColumn("content", "STRING"),
            SchemaColumn("file_name", "STRING"),
            SchemaColumn("chunk_index", "BIGINT"),
            SchemaColumn("chunk_type", "STRING"),
            SchemaColumn("bulletin_date", "TIMESTAMP"),
        ),
    )


def _officeqa_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i in range(20):
        rows.append(
            {
                "chunk_id": f"chunk-{i:04d}",
                "content": "x" * 800,
                "file_name": "doc.pdf" if i < 10 else "other.pdf",
                "chunk_index": i,
                "chunk_type": "table" if i % 2 == 0 else "text",
                "bulletin_date": "2026-01-01",
            }
        )
    return rows


def test_required_roles_inferred_from_officeqa_shape() -> None:
    rm = infer_roles(_officeqa_schema(), sample_rows=_officeqa_rows())
    assert rm.id_column == "chunk_id"
    assert rm.content_column == "content"


def test_optional_roles_inferred_from_officeqa_shape() -> None:
    rm = infer_roles(_officeqa_schema(), sample_rows=_officeqa_rows())
    assert rm.partition_column == "file_name"
    assert rm.order_column == "chunk_index"
    assert rm.type_column == "chunk_type"
    assert rm.date_column == "bulletin_date"


def test_optional_role_below_threshold_is_none() -> None:
    schema = Schema(
        fqn="cat.s.tbl",
        columns=(
            SchemaColumn("chunk_id", "STRING"),
            SchemaColumn("content", "STRING"),
            SchemaColumn("misc_string", "STRING"),
        ),
    )
    rows: list[dict[str, object]] = [
        {"chunk_id": f"c-{i}", "content": "x" * 800, "misc_string": "lorem"}
        for i in range(10)
    ]
    rm = infer_roles(schema, sample_rows=rows)
    # No partition / order / type / date columns by name.
    assert rm.partition_column is None
    assert rm.order_column is None
    assert rm.type_column is None
    assert rm.date_column is None


def test_required_role_missing_raises_inference_failed_with_candidates() -> None:
    schema = Schema(
        fqn="cat.s.tbl",
        columns=(
            SchemaColumn("col1", "STRING"),
            SchemaColumn("col2", "STRING"),
        ),
    )
    rows: list[dict[str, object]] = [
        {"col1": "abc", "col2": "def"} for _ in range(5)
    ]
    with pytest.raises(ToolErrorException) as exc:
        infer_roles(schema, sample_rows=rows)
    err = exc.value.error
    assert err.error_code is ErrorCode.INFERENCE_FAILED
    assert err.details["role"] in {"content_column", "id_column"}
    candidates = err.details["candidates"]
    assert isinstance(candidates, list)
    assert len(candidates) <= 3
    assert candidates  # at least one


def test_name_pattern_tiebreak_between_competing_columns() -> None:
    schema = Schema(
        fqn="cat.s.tbl",
        columns=(
            SchemaColumn("doc_id", "STRING"),
            SchemaColumn("body", "STRING"),
            SchemaColumn("aux", "STRING"),
        ),
    )
    rows: list[dict[str, object]] = [
        {"doc_id": f"id-{i:04d}", "body": "y" * 900, "aux": "z" * 900}
        for i in range(20)
    ]
    rm = infer_roles(schema, sample_rows=rows)
    # ``body`` matches the content_column name regex; ``aux`` does not.
    assert rm.content_column == "body"
    # ``doc_id`` matches the id_column regex.
    assert rm.id_column == "doc_id"


def test_type_filter_excludes_wrong_type_columns() -> None:
    """STRING name 'chunk_index' must not be picked as order_column (INT-only)."""
    schema = Schema(
        fqn="cat.s.tbl",
        columns=(
            SchemaColumn("chunk_id", "STRING"),
            SchemaColumn("content", "STRING"),
            SchemaColumn("chunk_index", "STRING"),  # wrong type
        ),
    )
    rows: list[dict[str, object]] = [
        {"chunk_id": f"c-{i}", "content": "x" * 800, "chunk_index": str(i)}
        for i in range(10)
    ]
    rm = infer_roles(schema, sample_rows=rows)
    assert rm.order_column is None


def test_role_candidate_dataclass_is_frozen() -> None:
    c = RoleCandidate(column="x", score=0.5, rationale="r")
    with pytest.raises(FrozenInstanceError):
        c.column = "y"  # type: ignore[misc]


def test_no_sample_rows_uses_degenerate_defaults_with_named_columns() -> None:
    """With no sample data, name-match alone can carry strong matches."""
    schema = Schema(
        fqn="cat.s.tbl",
        columns=(
            SchemaColumn("doc_id", "STRING"),
            SchemaColumn("content", "STRING"),
        ),
    )
    # With no sample rows, ``content`` matches CONTENT_NAME_RE (name=1.0):
    # 0.4*0.5 + 0.3*1.0 + 0.2*1.0 + 0.1*0.5 = 0.75 ≥ 0.7. ``doc_id`` matches
    # ID_NAME_RE: 0.5*1.0 + 0.3*0.5 + 0.2*1.0 = 0.85 ≥ 0.6.
    rm = infer_roles(schema, sample_rows=None)
    assert rm.content_column == "content"
    assert rm.id_column == "doc_id"
