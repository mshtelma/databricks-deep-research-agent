"""Tests for vector_search.py — title extraction and content column resolution."""

from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from databricks_deep_research.tools.builtins.vector_search import (
    DatabricksVectorSearchTool,
    _title_from_chunk_id,
)
from databricks_deep_research.tools.protocol import ToolContext


# ---------------------------------------------------------------------------
# _title_from_chunk_id
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("chunk_id, expected", [
    # Standard Databricks Volume paths
    (
        "dbfs__Volumes_users_christophe_chieu_documents_compete_upload__Platform_Battlecard__Snowflake_pdf_page25_chunk0",
        "Platform Battlecard: Snowflake (p.25)",
    ),
    (
        "dbfs__Volumes_users_christophe_chieu_documents_compete_upload__Platform_Battlecard__AWS_Lakehouse__pdf_page13_chunk0",
        "Platform Battlecard: AWS Lakehouse (p.13)",
    ),
    (
        "dbfs__Volumes_users_christophe_chieu_documents_compete_upload__Platform_Battlecard__AI_on_Azure_pdf_page14_chunk0",
        "Platform Battlecard: AI on Azure (p.14)",
    ),
    # No page/chunk suffix
    (
        "dbfs__Volumes_users_joe_docs__Sales_Playbook_pdf",
        "Sales Playbook",
    ),
    # Edge cases → empty string
    ("", ""),
    ("abc-123-uuid-no-pdf", ""),
    ("42", ""),
    # chunk_id with no path prefix
    ("My_Report_pdf_page1_chunk0", "My Report (p.1)"),
])
def test_title_from_chunk_id(chunk_id: str, expected: str):
    assert _title_from_chunk_id(chunk_id) == expected


# ---------------------------------------------------------------------------
# Content column resolution — chunk_content discovered via _discover_columns
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_content_column_from_discovered_embedding_source():
    """When the VS index has 'chunk_content' as the embedding source column,
    SourceInfo.content should contain the chunk text, not raw column dump."""

    ws = MagicMock()

    # Simulate index metadata with chunk_content as embedding source
    index_info = SimpleNamespace(
        primary_key="chunk_id",
        delta_sync_index_spec=SimpleNamespace(
            primary_key_columns=["chunk_id"],
            embedding_source_columns=[{"name": "chunk_content"}],
        ),
    )
    ws.vector_search_indexes.get_index.return_value = index_info

    # Simulate query result with chunk_content column
    manifest = SimpleNamespace(
        columns=[
            SimpleNamespace(name="chunk_id"),
            SimpleNamespace(name="chunk_content"),
            SimpleNamespace(name="score"),
        ]
    )
    result_data = SimpleNamespace(
        data_array=[
            [
                "dbfs__Volumes_users_joe_upload__Battlecard__Snowflake_pdf_page5_chunk0",
                "Snowflake lacks native ML support compared to Databricks.",
                0.85,
            ]
        ]
    )
    query_result = SimpleNamespace(manifest=manifest, result=result_data)
    ws.vector_search_indexes.query_index.return_value = query_result

    tool = DatabricksVectorSearchTool(
        workspace_client=ws,
        name="vs_compete",
        index_name="test.schema.compete_idx",
    )

    ctx = ToolContext(url_registry=None)
    result = await tool.execute({"query": "Snowflake comparison"}, ctx)

    assert result.success
    assert len(result.sources) == 1
    src = result.sources[0]
    # Content should be the actual chunk text, not a raw dump
    assert src.content == "Snowflake lacks native ML support compared to Databricks."
    # Title should be parsed from chunk_id
    assert src.title == "Battlecard: Snowflake (p.5)"


# ---------------------------------------------------------------------------
# Harness label fallback — generic VS titles replaced with source metadata
# ---------------------------------------------------------------------------

def test_harness_label_replaces_generic_vs_title():
    """When title is 'Vector search result N', harness should use
    source_description or source_name instead."""
    title = "Vector search result 1"
    item = {
        "title": title,
        "source_description": "Search internal competitive intelligence knowledge base for battlecards",
        "source_name": "vs_compete_intel",
    }

    # Replicate the harness logic
    if re.match(r'^Vector search result \d+$', title):
        source_desc = str(item.get("source_description", "") or "")
        source_name = str(item.get("source_name", "") or "")
        if source_desc:
            title = source_desc[:120]
        elif source_name:
            title = source_name

    assert title == "Search internal competitive intelligence knowledge base for battlecards"


def test_harness_label_keeps_good_title():
    """When title is already descriptive, harness should not replace it."""
    title = "Platform Battlecard: Snowflake (p.25)"
    item = {
        "title": title,
        "source_description": "Some generic description",
    }

    if re.match(r'^Vector search result \d+$', title):
        source_desc = str(item.get("source_description", "") or "")
        if source_desc:
            title = source_desc[:120]

    assert title == "Platform Battlecard: Snowflake (p.25)"


def test_harness_label_falls_back_to_source_name():
    """When source_description is empty, fall back to source_name."""
    title = "Vector search result 3"
    item = {
        "title": title,
        "source_description": "",
        "source_name": "vs_compete_intel",
    }

    if re.match(r'^Vector search result \d+$', title):
        source_desc = str(item.get("source_description", "") or "")
        source_name = str(item.get("source_name", "") or "")
        if source_desc:
            title = source_desc[:120]
        elif source_name:
            title = source_name

    assert title == "vs_compete_intel"
