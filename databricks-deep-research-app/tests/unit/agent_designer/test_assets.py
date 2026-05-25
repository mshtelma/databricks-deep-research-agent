from __future__ import annotations

from typing import Any

from deep_research.agent_designer.assets import (
    asset_context_payload,
    detect_asset_contract,
    inspect_assets,
    normalize_assets,
    recommend_tools_for_assets,
)
from deep_research.agent_designer.registry import source_kinds_payload, tool_kinds_payload


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
    assert "delta_read" in kinds
    assert "delta_grep" in kinds
    assert "delta_table_read" in kinds
    assert "compute" in kinds
    assert "compute_namespace" in kinds

    vector_tool = next(tool for tool in tools if tool["kind"] == "vector_search")
    assert vector_tool["config"]["index_name"].endswith("treasury_chunks_vs_index")
    assert vector_tool["config"]["query_type"] == "HYBRID"

    table_tool = next(tool for tool in tools if tool["kind"] == "delta_table_read")
    assert table_tool["config"]["warehouse_id"] == "abc123"
    assert table_tool["config"]["content_column"] == "table_json"
    assert table_tool["config"]["pk_column"] == "chunk_id"
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
    assert {"vector_search", "delta_read", "delta_grep"}.issubset(names)
    assert all("secret_domain" not in name and "customer_notes" not in name for name in names)


def test_recommend_tools_for_delta_table_without_warehouse_reports_diagnostic() -> None:
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
                "kind": "delta_read",
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
                "kind": "delta_read",
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
    assert "custom" not in tool_kinds
    for kind in ("delta_read", "delta_grep", "delta_table_read"):
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
    assert "delta_context" in tool_kinds
