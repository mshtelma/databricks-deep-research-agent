from __future__ import annotations

from typing import Any

from deep_research.agent_designer.assets import (
    asset_context_payload,
    assets_from_ast,
    detect_asset_contract,
    infer_assets_from_intent,
    inspect_assets,
    normalize_assets,
    recommend_tools_for_assets,
)
from deep_research.agent_designer.registry import source_kinds_payload, tool_kinds_payload

# ---------------------------------------------------------------------------
# assets_from_ast — reconstruct corpus assets from an existing workflow AST so
# an EDIT preserves the data tools instead of rebuilding web-only (Issue #2).
# ---------------------------------------------------------------------------


def _edit_ast() -> dict[str, Any]:
    return {
        "tools": [
            {"name": "vs1", "kind": "vector_search",
             "config": {"index_name": "cat.sch.idx", "columns": ["a", "b"], "query_type": "ANN"}},
            {"name": "ts1", "kind": "table_search",
             "config": {"table_name": "cat.sch.tbl", "warehouse_id": "wh-custom-123"}},
            {"name": "tr1", "kind": "table_read",
             "config": {"table_name": "cat.sch.tbl", "warehouse_id": "wh-custom-123"}},
            {"name": "gn1", "kind": "genie", "config": {"genie_space_id": "space-xyz"}},
            {"name": "web", "kind": "web_research", "config": {}},
            {"name": "wc", "kind": "web_crawl", "config": {}},
        ]
    }


def test_assets_from_ast_maps_kinds_and_excludes_web() -> None:
    assets = assets_from_ast(_edit_ast())
    by_kind = {a.kind for a in assets}
    # web_research / web_crawl are NOT assets.
    assert by_kind == {"vector_index", "delta_table", "genie_space"}
    # delta_table deduped across table_search + table_read on the same table.
    assert sum(1 for a in assets if a.kind == "delta_table") == 1


def test_assets_from_ast_preserves_nondefault_warehouse_and_columns() -> None:
    assets = assets_from_ast(_edit_ast())
    table = next(a for a in assets if a.kind == "delta_table")
    assert table.full_name == "cat.sch.tbl"
    assert table.metadata.get("warehouse_id") == "wh-custom-123"
    vec = next(a for a in assets if a.kind == "vector_index")
    assert vec.full_name == "cat.sch.idx"
    assert vec.metadata.get("columns") == ["a", "b"]
    assert vec.metadata.get("query_type") == "ANN"


def test_assets_from_ast_genie_space_id_keys() -> None:
    # Either space_id or genie_space_id resolves the genie identity (RC5).
    for key in ("space_id", "genie_space_id"):
        assets = assets_from_ast(
            {"tools": [{"name": "g", "kind": "genie", "config": {key: "sp-1"}}]}
        )
        assert [(a.kind, a.full_name) for a in assets] == [("genie_space", "sp-1")]


def test_assets_from_ast_round_trips_to_same_tools() -> None:
    # The derived assets regenerate the SAME corpus tool kinds + identities,
    # preserving the non-default warehouse through recommend_tools_for_assets.
    assets = assets_from_ast(_edit_ast())
    reco = recommend_tools_for_assets([a.model_dump(exclude_none=True) for a in assets])
    tools = reco.get("recommended_tools") or []
    kinds = {t["kind"] for t in tools}
    assert "vector_search" in kinds and "genie" in kinds
    assert any(t["kind"].startswith("table_") for t in tools)
    table_tool = next(t for t in tools if t["kind"].startswith("table_"))
    assert table_tool["config"]["warehouse_id"] == "wh-custom-123"


def test_assets_from_ast_handles_empty_and_non_dict() -> None:
    assert assets_from_ast(None) == []
    assert assets_from_ast({}) == []
    assert assets_from_ast({"tools": "nope"}) == []
    assert assets_from_ast({"tools": [{"kind": "vector_search", "config": {}}]}) == []  # no identity


def _office_like_assets() -> list[dict[str, Any]]:
    return [
        {
            "kind": "vector_index",
            "full_name": "main.officeqa_benchmark.treasury_chunks_vs_index",
            "usage": "required",
            "metadata": {
                "columns": ["chunk_id", "file_name", "content"],
                "query_type": "HYBRID",
            },
        },
        {
            "kind": "delta_table",
            "full_name": "main.officeqa_benchmark.treasury_chunks",
            "usage": "required",
            "field_roles": {
                "primary_key": "chunk_id",
                "content": "content",
                "order_by": "chunk_id",
            },
            "metadata": {
                "warehouse_id": "abc123",
                "columns": ["chunk_id", "file_name", "content", "chunk_type"],
            },
        },
        {
            "kind": "delta_table",
            "full_name": "main.officeqa_benchmark.treasury_tables",
            "usage": "required",
            "field_roles": {
                "primary_key": "chunk_id",
                "content": "content",
                "structured_json": "table_json",
                "order_by": "chunk_id",
            },
            "metadata": {
                "warehouse_id": "abc123",
                "columns": ["chunk_id", "file_name", "content", "table_json"],
            },
        },
    ]


def test_normalize_assets_accepts_context_and_deduplicates() -> None:
    raw = {
        "assets": [
            {"kind": "vector_index", "full_name": "main.cat.idx"},
            {"kind": "vector_index", "full_name": "MAIN.CAT.IDX"},
            {"kind": "delta_table", "full_name": "main.cat.rows"},
            {"kind": "delta_table"},
            "not an asset",
        ],
    }

    assets = normalize_assets(raw)

    assert [(asset.kind, asset.full_name) for asset in assets] == [
        ("vector_index", "main.cat.idx"),
        ("delta_table", "main.cat.rows"),
    ]


def test_infer_assets_from_intent_uses_prompt_resource_names(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("TABLE_TOOLS_WAREHOUSE_ID", "wh-auto")
    intent = """
    Use vector index main.officeqa_benchmark.treasury_chunks_vs_index.
    Use Delta table main.officeqa_benchmark.treasury_chunks for chunks and
    Delta table main.officeqa_benchmark.treasury_tables for structured rows.
    """

    assets = infer_assets_from_intent(intent)

    assert [(asset["kind"], asset["full_name"]) for asset in assets] == [
        ("vector_index", "main.officeqa_benchmark.treasury_chunks_vs_index"),
        ("delta_table", "main.officeqa_benchmark.treasury_chunks"),
        ("delta_table", "main.officeqa_benchmark.treasury_tables"),
    ]
    table_assets = [asset for asset in assets if asset["kind"] == "delta_table"]
    assert all(asset["metadata"]["warehouse_id"] == "wh-auto" for asset in table_assets)


def test_asset_context_and_inspection_are_compact_and_untrusted() -> None:
    payload = asset_context_payload(_office_like_assets())
    inspected = inspect_assets(payload)

    assert payload["count"] == 3
    assert inspected["count"] == 3
    assert inspected["assets"][0]["identity"].endswith("treasury_chunks_vs_index")
    assert inspected["assets"][0]["metadata_keys"] == ["columns", "query_type"]
    assert "untrusted data" in inspected["assets"][0]["note"]


def test_recommend_tools_for_assets_returns_vector_delta_and_compute_tools() -> None:
    result = recommend_tools_for_assets(
        _office_like_assets(),
        intent="answer numeric questions and calculate totals from tables",
    )

    assert result["diagnostics"] == []
    tools = result["recommended_tools"]
    kinds = [tool["kind"] for tool in tools]
    assert "vector_search" in kinds
    assert "table_search" in kinds
    assert "table_read" in kinds
    assert "table_load" in kinds
    assert "compute" in kinds
    assert "compute_namespace" in kinds

    vector_tool = next(tool for tool in tools if tool["kind"] == "vector_search")
    assert vector_tool["config"]["index_name"].endswith("treasury_chunks_vs_index")
    assert vector_tool["config"]["query_type"] == "HYBRID"

    table_tool = next(tool for tool in tools if tool["kind"] == "table_load")
    assert table_tool["config"]["warehouse_id"] == "abc123"
    assert table_tool["config"]["compute_tool_name"] == "compute"


def test_recommend_tools_for_assets_uses_generic_runtime_tool_names() -> None:
    result = recommend_tools_for_assets(
        [
            {
                "kind": "vector_index",
                "full_name": "main.secret_domain.customer_notes_vs_index",
                "usage": "required",
            },
            {
                "kind": "delta_table",
                "full_name": "main.secret_domain.customer_notes",
                "usage": "required",
                "field_roles": {
                    "primary_key": "id",
                    "content": "body",
                },
                "metadata": {
                    "warehouse_id": "abc123",
                    "columns": ["id", "body"],
                },
            },
        ],
        intent="answer questions from the selected corpus",
    )

    names = {tool["name"] for tool in result["recommended_tools"]}
    assert {"vector_search", "table_search", "table_read", "table_load"}.issubset(names)
    assert all("secret_domain" not in name and "customer_notes" not in name for name in names)


def test_recommend_tools_for_delta_table_without_warehouse_reports_diagnostic(
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("TABLE_TOOLS_WAREHOUSE_ID", raising=False)
    monkeypatch.delenv("STORAGE_WAREHOUSE_ID", raising=False)
    result = recommend_tools_for_assets(
        [
            {
                "kind": "delta_table",
                "full_name": "main.cat.rows",
                "usage": "required",
                "field_roles": {"content": "content"},
            }
        ],
        intent="find exact text",
    )

    assert result["recommended_tools"] == []
    assert result["diagnostics"][0]["severity"] == "error"
    assert "warehouse" in result["diagnostics"][0]["message"]


def test_recommend_tools_for_assets_uses_default_table_warehouse(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("STORAGE_WAREHOUSE_ID", "wh-default")
    monkeypatch.delenv("TABLE_TOOLS_WAREHOUSE_ID", raising=False)

    result = recommend_tools_for_assets(
        [{"kind": "delta_table", "full_name": "main.cat.rows"}],
        intent="answer questions from tables",
    )

    assert result["diagnostics"] == []
    table_tool = next(
        tool for tool in result["recommended_tools"] if tool["kind"] == "table_search"
    )
    assert table_tool["config"]["warehouse_id"] == "wh-default"


def test_detect_asset_contract_requires_declared_and_bound_tools() -> None:
    assets = [
        {
            "kind": "delta_table",
            "full_name": "main.cat.rows",
            "usage": "required",
            "metadata": {"warehouse_id": "abc123"},
        }
    ]
    ast = {
        "root": {
            "id": "root",
            "type": "agent",
            "config": {"tools": []},
        },
        "tools": [
            {
                "name": "rows_read",
                "kind": "table_read",
                "config": {"table_name": "main.cat.rows", "warehouse_id": "abc123"},
            }
        ],
    }

    errors = detect_asset_contract(ast, assets)

    assert len(errors) == 1
    assert "not bound" in errors[0].message

    ast["root"]["config"]["tools"] = ["rows_read"]
    assert detect_asset_contract(ast, assets) == []


def test_detect_asset_contract_fails_required_delta_tool_without_warehouse() -> None:
    assets = [
        {
            "kind": "delta_table",
            "full_name": "main.cat.rows",
            "usage": "required",
        }
    ]
    ast = {
        "root": {
            "id": "root",
            "type": "agent",
            "config": {"tools": ["rows_read"]},
        },
        "tools": [
            {
                "name": "rows_read",
                "kind": "table_read",
                "config": {"table_name": "main.cat.rows"},
            }
        ],
    }

    errors = detect_asset_contract(ast, assets)

    assert len(errors) == 1
    assert "missing config.warehouse_id" in errors[0].message


def test_detect_asset_contract_fails_required_asset_wrong_tool_kind() -> None:
    assets = [
        {
            "kind": "vector_index",
            "full_name": "main.cat.idx",
            "usage": "required",
        }
    ]
    ast = {
        "root": {
            "id": "root",
            "type": "agent",
            "config": {"tools": ["idx_custom"]},
        },
        "tools": [
            {
                "name": "idx_custom",
                "kind": "custom",
                "config": {"index_name": "main.cat.idx"},
            }
        ],
    }

    errors = detect_asset_contract(ast, assets)

    assert len(errors) == 1
    assert "incompatible tool kind" in errors[0].message
    assert "vector_search" in errors[0].message


def test_registry_exposes_generic_table_and_compute_schemas() -> None:
    source_kinds = {item["kind"] for item in source_kinds_payload()}
    tool_kinds = {item["kind"]: item for item in tool_kinds_payload()}

    assert "delta_table" in source_kinds
    assert "sql_warehouse" in source_kinds
    assert "custom" not in tool_kinds
    for kind in ("table_search", "table_read", "table_neighbors", "table_load", "table_aggregate"):
        schema = tool_kinds[kind]["config_schema"]
        assert "table_name" in schema["properties"]
        assert "warehouse_id" in schema["properties"]
        assert {"table_name", "warehouse_id"}.issubset(set(schema["required"]))

    assert "max_execution_seconds" in tool_kinds["compute"]["config_schema"]["properties"]
    assert (
        tool_kinds["compute_namespace"]["config_schema"]["properties"]["compute_tool_name"][
            "default"
        ]
        == "compute"
    )
    assert "auto_fetch_top_k" in tool_kinds["web_research"]["config_schema"]["properties"]
    assert "table_discovery" in tool_kinds
