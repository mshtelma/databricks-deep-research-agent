"""Unit tests for boot-time validation of BOUND text-table bindings.

Boot validation runs once at app startup against the BOUND entries in
:class:`TableBindingRegistry` to confirm:

1. The configured ``fqn`` is fetchable by the schema cache.
2. Every column referenced in the binding's ``RoleMap`` exists in the
   live schema.
3. Every entry in ``numeric_columns`` exists in the live schema and has a
   numeric type.
4. Structured passage parser mappings reference supported parsers.

A successful validation produces a :class:`BindingValidationReport`.
A failure raises :class:`ToolErrorException` (default) or accumulates
into the report when ``raise_on_error=False``.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from databricks_deep_research.tools.builtins.text_table import (
    BindingInfo,
    BindingSource,
    BindingValidationReport,
    ErrorCode,
    RoleMap,
    Schema,
    SchemaColumn,
    TableBindingRegistry,
    ToolErrorException,
    validate_all_bound_bindings,
    validate_bound_binding,
)


def _docs_binding(*, name: str = "docs", fqn: str = "cat.sch.docs") -> BindingInfo:
    return BindingInfo(
        name=name,
        fqn=fqn,
        source=BindingSource.BOUND,
        description="Doc chunks.",
        roles=RoleMap(
            id_column="id",
            content_column="text",
            partition_column="doc",
            order_column="seq",
        ),
        numeric_columns=("seq",),
    )


def _docs_schema(fqn: str = "cat.sch.docs") -> Schema:
    return Schema(
        fqn=fqn,
        columns=(
            SchemaColumn(name="id", data_type="string", nullable=False),
            SchemaColumn(name="text", data_type="string"),
            SchemaColumn(name="doc", data_type="string"),
            SchemaColumn(name="seq", data_type="bigint"),
        ),
    )


class _FakeSchemaCache:
    def __init__(self, schemas: dict[str, Schema]) -> None:
        self._schemas = schemas
        self.calls: list[tuple[str, str]] = []

    def get(self, fqn: str, user_token: str) -> Schema:
        self.calls.append((fqn, user_token))
        if fqn not in self._schemas:
            raise RuntimeError(f"unknown table: {fqn}")
        return self._schemas[fqn]


# ---------------------------------------------------------------------------
# validate_bound_binding (single-binding API)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_bound_binding_succeeds_when_schema_matches() -> None:
    cache = _FakeSchemaCache({"cat.sch.docs": _docs_schema()})
    info = _docs_binding()
    # Should not raise.
    validate_bound_binding(info, schema_cache=cache, user_token="t")
    assert cache.calls == [("cat.sch.docs", "t")]


@pytest.mark.unit
def test_validate_bound_binding_rejects_non_bound_source() -> None:
    cache = _FakeSchemaCache({"cat.sch.docs": _docs_schema()})
    info = BindingInfo(
        name="docs",
        fqn="cat.sch.docs",
        source=BindingSource.DISCOVERED,
    )
    with pytest.raises(ValueError):
        validate_bound_binding(info, schema_cache=cache, user_token="t")


@pytest.mark.unit
def test_validate_bound_binding_raises_on_schema_fetch_failure() -> None:
    cache = _FakeSchemaCache({})
    info = _docs_binding()
    with pytest.raises(ToolErrorException) as exc:
        validate_bound_binding(info, schema_cache=cache, user_token="t")
    assert exc.value.error.error_code is ErrorCode.SCHEMA_FETCH_FAILED
    assert exc.value.error.binding == "docs"


@pytest.mark.unit
def test_validate_bound_binding_raises_on_missing_id_column() -> None:
    cache = _FakeSchemaCache({"cat.sch.docs": _docs_schema()})
    info = BindingInfo(
        name="docs",
        fqn="cat.sch.docs",
        source=BindingSource.BOUND,
        roles=RoleMap(id_column="missing_id", content_column="text"),
    )
    with pytest.raises(ToolErrorException) as exc:
        validate_bound_binding(info, schema_cache=cache, user_token="t")
    assert exc.value.error.error_code is ErrorCode.INVALID_COLUMN
    assert "missing_id" in exc.value.error.message


@pytest.mark.unit
def test_validate_bound_binding_raises_on_missing_content_column() -> None:
    cache = _FakeSchemaCache({"cat.sch.docs": _docs_schema()})
    info = BindingInfo(
        name="docs",
        fqn="cat.sch.docs",
        source=BindingSource.BOUND,
        roles=RoleMap(id_column="id", content_column="missing_text"),
    )
    with pytest.raises(ToolErrorException) as exc:
        validate_bound_binding(info, schema_cache=cache, user_token="t")
    assert exc.value.error.error_code is ErrorCode.INVALID_COLUMN
    assert "missing_text" in exc.value.error.message


@pytest.mark.unit
def test_validate_bound_binding_raises_on_missing_optional_role() -> None:
    cache = _FakeSchemaCache({"cat.sch.docs": _docs_schema()})
    info = BindingInfo(
        name="docs",
        fqn="cat.sch.docs",
        source=BindingSource.BOUND,
        roles=RoleMap(
            id_column="id",
            content_column="text",
            order_column="missing_seq",
        ),
    )
    with pytest.raises(ToolErrorException) as exc:
        validate_bound_binding(info, schema_cache=cache, user_token="t")
    assert exc.value.error.error_code is ErrorCode.INVALID_COLUMN
    assert "missing_seq" in exc.value.error.message


@pytest.mark.unit
def test_validate_bound_binding_raises_on_missing_numeric_column() -> None:
    cache = _FakeSchemaCache({"cat.sch.docs": _docs_schema()})
    info = BindingInfo(
        name="docs",
        fqn="cat.sch.docs",
        source=BindingSource.BOUND,
        roles=RoleMap(id_column="id", content_column="text"),
        numeric_columns=("seq", "ghost"),
    )
    with pytest.raises(ToolErrorException) as exc:
        validate_bound_binding(info, schema_cache=cache, user_token="t")
    assert exc.value.error.error_code is ErrorCode.INVALID_COLUMN
    assert "ghost" in exc.value.error.message


@pytest.mark.unit
def test_validate_bound_binding_rejects_no_roles() -> None:
    cache = _FakeSchemaCache({"cat.sch.docs": _docs_schema()})
    info = BindingInfo(
        name="docs",
        fqn="cat.sch.docs",
        source=BindingSource.BOUND,
    )
    with pytest.raises(ToolErrorException) as exc:
        validate_bound_binding(info, schema_cache=cache, user_token="t")
    assert exc.value.error.error_code is ErrorCode.INVALID_BINDING
    assert "must declare roles" in exc.value.error.message


@pytest.mark.unit
def test_validate_bound_binding_rejects_incompatible_role_type() -> None:
    cache = _FakeSchemaCache(
        {
            "cat.sch.docs": Schema(
                fqn="cat.sch.docs",
                columns=(
                    SchemaColumn(name="id", data_type="string"),
                    SchemaColumn(name="text", data_type="bigint"),
                ),
            )
        }
    )
    info = BindingInfo(
        name="docs",
        fqn="cat.sch.docs",
        source=BindingSource.BOUND,
        roles=RoleMap(id_column="id", content_column="text"),
    )
    with pytest.raises(ToolErrorException) as exc:
        validate_bound_binding(info, schema_cache=cache, user_token="t")
    assert exc.value.error.error_code is ErrorCode.INVALID_COLUMN
    assert "content" in exc.value.error.message
    assert "bigint" in exc.value.error.message


@pytest.mark.unit
def test_validate_bound_binding_rejects_non_numeric_numeric_column() -> None:
    cache = _FakeSchemaCache({"cat.sch.docs": _docs_schema()})
    info = BindingInfo(
        name="docs",
        fqn="cat.sch.docs",
        source=BindingSource.BOUND,
        roles=RoleMap(id_column="id", content_column="text"),
        numeric_columns=("doc",),
    )
    with pytest.raises(ToolErrorException) as exc:
        validate_bound_binding(info, schema_cache=cache, user_token="t")
    assert exc.value.error.error_code is ErrorCode.INVALID_COLUMN
    assert "numeric" in exc.value.error.message


@pytest.mark.unit
def test_validate_bound_binding_accepts_structured_passage_parsers() -> None:
    cache = _FakeSchemaCache(
        {
            "cat.sch.docs": Schema(
                fqn="cat.sch.docs",
                columns=(
                    SchemaColumn(name="id", data_type="string"),
                    SchemaColumn(name="text", data_type="string"),
                    SchemaColumn(name="chunk_type", data_type="string"),
                ),
            )
        }
    )
    info = BindingInfo(
        name="docs",
        fqn="cat.sch.docs",
        source=BindingSource.BOUND,
        roles=RoleMap(
            id_column="id",
            content_column="text",
            type_column="chunk_type",
        ),
        structured_passages={"table": "html", "equation": "markdown", "data": "json"},
    )
    validate_bound_binding(info, schema_cache=cache, user_token="t")


@pytest.mark.unit
def test_validate_bound_binding_rejects_invalid_structured_parser() -> None:
    cache = _FakeSchemaCache(
        {
            "cat.sch.docs": Schema(
                fqn="cat.sch.docs",
                columns=(
                    SchemaColumn(name="id", data_type="string"),
                    SchemaColumn(name="text", data_type="string"),
                    SchemaColumn(name="chunk_type", data_type="string"),
                ),
            )
        }
    )
    info = BindingInfo(
        name="docs",
        fqn="cat.sch.docs",
        source=BindingSource.BOUND,
        roles=RoleMap(
            id_column="id",
            content_column="text",
            type_column="chunk_type",
        ),
        structured_passages={"table": "xml"},
    )
    with pytest.raises(ToolErrorException) as exc:
        validate_bound_binding(info, schema_cache=cache, user_token="t")
    assert exc.value.error.error_code is ErrorCode.INVALID_BINDING
    assert "xml" in exc.value.error.message


@pytest.mark.unit
def test_validate_bound_binding_requires_type_role_for_structured_passages() -> None:
    cache = _FakeSchemaCache({"cat.sch.docs": _docs_schema()})
    info = BindingInfo(
        name="docs",
        fqn="cat.sch.docs",
        source=BindingSource.BOUND,
        roles=RoleMap(id_column="id", content_column="text"),
        structured_passages={"table": "html"},
    )
    with pytest.raises(ToolErrorException) as exc:
        validate_bound_binding(info, schema_cache=cache, user_token="t")
    assert exc.value.error.error_code is ErrorCode.INVALID_BINDING
    assert "type_column" in exc.value.error.message


# ---------------------------------------------------------------------------
# validate_all_bound_bindings (registry-level API)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_all_skips_discovered_bindings() -> None:
    registry = TableBindingRegistry()
    registry.register_bound(_docs_binding())
    registry.register_discovered(
        BindingInfo(
            name="dyn",
            fqn="cat.sch.dyn",  # NOT in cache — would fail if validated.
            source=BindingSource.DISCOVERED,
        )
    )
    cache = _FakeSchemaCache({"cat.sch.docs": _docs_schema()})
    report = validate_all_bound_bindings(
        registry, schema_cache=cache, user_token="t"
    )
    assert report.ok
    assert report.validated == ("docs",)
    assert report.errors == ()
    # The unknown DISCOVERED binding was never queried.
    assert ("cat.sch.dyn", "t") not in cache.calls


@pytest.mark.unit
def test_validate_all_returns_report_when_raise_on_error_false() -> None:
    registry = TableBindingRegistry()
    registry.register_bound(_docs_binding())
    bad = BindingInfo(
        name="bad",
        fqn="cat.sch.bad",
        source=BindingSource.BOUND,
        roles=RoleMap(id_column="missing", content_column="missing2"),
    )
    registry.register_bound(bad)
    cache = _FakeSchemaCache(
        {
            "cat.sch.docs": _docs_schema(),
            "cat.sch.bad": _docs_schema(fqn="cat.sch.bad"),
        }
    )
    report = validate_all_bound_bindings(
        registry, schema_cache=cache, user_token="t", raise_on_error=False
    )
    assert not report.ok
    assert "docs" in report.validated
    assert any(err.binding == "bad" for err in report.errors)


@pytest.mark.unit
def test_validate_all_raises_first_error_by_default() -> None:
    registry = TableBindingRegistry()
    bad = BindingInfo(
        name="bad",
        fqn="cat.sch.bad",
        source=BindingSource.BOUND,
        roles=RoleMap(id_column="missing", content_column="missing2"),
    )
    registry.register_bound(bad)
    cache = _FakeSchemaCache({"cat.sch.bad": _docs_schema(fqn="cat.sch.bad")})
    with pytest.raises(ToolErrorException):
        validate_all_bound_bindings(
            registry, schema_cache=cache, user_token="t"
        )


@pytest.mark.unit
def test_validate_all_empty_registry_is_ok() -> None:
    registry = TableBindingRegistry()
    cache = _FakeSchemaCache({})
    report = validate_all_bound_bindings(
        registry, schema_cache=cache, user_token="t"
    )
    assert report.ok
    assert report.validated == ()
    assert report.errors == ()


@pytest.mark.unit
def test_validation_report_is_immutable() -> None:
    report = BindingValidationReport(validated=("a", "b"), errors=())
    assert isinstance(report.validated, tuple)
    assert isinstance(report.errors, tuple)
    # Frozen dataclass: assignment must fail.
    with pytest.raises(FrozenInstanceError):
        report.validated = ("c",)  # type: ignore[misc]


@pytest.mark.unit
def test_validate_all_accepts_multiple_failures_in_collect_mode() -> None:
    registry = TableBindingRegistry()
    bad1 = BindingInfo(
        name="b1",
        fqn="cat.sch.b1",
        source=BindingSource.BOUND,
        roles=RoleMap(id_column="x", content_column="y"),
    )
    bad2 = BindingInfo(
        name="b2",
        fqn="cat.sch.b2",
        source=BindingSource.BOUND,
        roles=RoleMap(id_column="a", content_column="b"),
    )
    registry.register_bound(bad1)
    registry.register_bound(bad2)
    schema_b1 = Schema(
        fqn="cat.sch.b1",
        columns=(SchemaColumn(name="other", data_type="string"),),
    )
    schema_b2 = Schema(
        fqn="cat.sch.b2",
        columns=(SchemaColumn(name="other", data_type="string"),),
    )
    cache = _FakeSchemaCache({"cat.sch.b1": schema_b1, "cat.sch.b2": schema_b2})
    report = validate_all_bound_bindings(
        registry, schema_cache=cache, user_token="t", raise_on_error=False
    )
    assert not report.ok
    binding_names = {err.binding for err in report.errors}
    assert binding_names == {"b1", "b2"}
