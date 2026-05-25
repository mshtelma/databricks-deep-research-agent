"""Registry payloads exposed by the GET /api/v1/agent-designer/registry endpoint
and consumed by the chat orchestrator's list_node_types / list_tool_kinds tools.

Lives in the agent_designer package so the orchestrator (service layer) does
not need to import from the API layer.
"""
from __future__ import annotations

from typing import Any

from databricks_deep_research.agents.config import (
    AgentNodeConfig,
    ConditionalNodeConfig,
    LoopNodeConfig,
    PlanAndExecuteNodeConfig,
    SubworkflowNodeConfig,
    ToolNodeConfig,
)
from databricks_deep_research.tools.protocol import ToolKind
from databricks_deep_research.workflow.definition import NodeType
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.core.app_config import get_app_config

# Per-NodeType metadata; the editor uses this to populate the palette.
NODE_TYPE_META: dict[str, dict[str, Any]] = {
    NodeType.agent: {
        "label": "Agent",
        "icon": "robot",
        "category": "leaf",
        "is_composite": False,
        "config_model": AgentNodeConfig,
        "default_config": {"subtype": "researcher", "model_tier": "analytical"},
        "summary_template": "{{subtype}} · {{model_tier}}",
    },
    NodeType.tool: {
        "label": "Tool",
        "icon": "tool",
        "category": "leaf",
        "is_composite": False,
        "config_model": ToolNodeConfig,
        "default_config": {"ref": "web_search"},
        "summary_template": "tool: {{ref}}",
    },
    NodeType.sequence: {
        "label": "Sequence",
        "icon": "arrow-right",
        "category": "control_flow",
        "is_composite": True,
        "config_model": None,
        "children_kind": "ordered_list",
        "default_config": {},
        "summary_template": "{{children.length}} steps",
    },
    NodeType.parallel: {
        "label": "Parallel",
        "icon": "parallel",
        "category": "control_flow",
        "is_composite": True,
        "config_model": None,
        "children_kind": "ordered_list",
        "default_config": {},
        "summary_template": "{{children.length}} branches",
    },
    NodeType.loop: {
        "label": "Loop",
        "icon": "loop",
        "category": "control_flow",
        "is_composite": True,
        "config_model": LoopNodeConfig,
        "children_kind": "ordered_list",
        "default_config": {
            "until": {"kind": "key_equals", "state_key": "done", "value": True},
            "min_iterations": 1,
            "max_iterations": 10,
        },
        "summary_template": "{{min_iterations}}-{{max_iterations}} iterations",
    },
    NodeType.conditional: {
        "label": "If / Else",
        "icon": "branch",
        "category": "control_flow",
        "is_composite": True,
        "config_model": ConditionalNodeConfig,
        "children_kind": "index_paired_branches",
        "branches_pairing": {
            "spec_field": "config.conditions",
            "default_field": "config.default_branch",
            "rule": "children[i] is paired with conditions[i]; default_branch indexes the fallback child",
        },
        "default_config": {
            "conditions": [{"kind": "key_equals", "state_key": "intent", "value": "yes"}],
            "default_branch": 1,
        },
        "summary_template": "{{conditions.length}} branches + default",
    },
    NodeType.subworkflow: {
        "label": "Subworkflow",
        "icon": "subworkflow",
        "category": "leaf",
        "is_composite": False,
        "config_model": SubworkflowNodeConfig,
        "default_config": {},
        "summary_template": "workflow: {{ref}}",
    },
    NodeType.plan_and_execute: {
        "label": "Plan & Execute",
        "icon": "plan",
        "category": "control_flow",
        "is_composite": True,
        "config_model": PlanAndExecuteNodeConfig,
        "children_kind": "named_slots",
        "children_slots": [
            {
                "key": "config.body",
                "label": "Body",
                "min": 1,
                "max": 1,
                "auto_wrap_in_sequence": True,
            },
            {"key": "config.evaluator", "label": "Evaluator", "min": 0, "max": 1},
        ],
        "default_config": {
            "planner": {"subtype": "planner", "model_tier": "analytical"},
            "body": {},
            "max_iterations": 10,
        },
        "summary_template": "max={{max_iterations}}",
    },
}


AGENT_SUBTYPES: list[dict[str, Any]] = [
    {"id": "coordinator", "label": "Coordinator", "icon": "star", "default_model_tier": "complex"},
    {"id": "planner", "label": "Planner", "icon": "map", "default_model_tier": "analytical"},
    {"id": "researcher", "label": "Researcher", "icon": "search", "default_model_tier": "analytical"},
    {"id": "reflector", "label": "Reflector", "icon": "reflect", "default_model_tier": "analytical"},
    {"id": "synthesizer", "label": "Synthesizer", "icon": "merge", "default_model_tier": "complex"},
    {"id": "background", "label": "Background", "icon": "clock", "default_model_tier": "simple"},
]


SOURCE_KINDS: list[dict[str, str]] = [
    {
        "kind": "vector_index",
        "label": "Vector Index",
        "source_type": "vector_search",
        "icon": "database",
    },
    {
        "kind": "genie_space",
        "label": "Genie Space",
        "source_type": "genie",
        "icon": "sparkles",
    },
    {
        "kind": "knowledge_assistant",
        "label": "Knowledge Assistant",
        "source_type": "knowledge_assistant",
        "icon": "assistant",
    },
    {
        "kind": "serving_endpoint",
        "label": "Serving Endpoint",
        "source_type": "knowledge_assistant",
        "icon": "endpoint",
    },
    {
        "kind": "delta_table",
        "label": "Delta Table",
        "source_type": "delta_table",
        "icon": "table",
    },
]


REGISTRY_VERSION = "1.0.0"

_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}

_STRING_ARRAY_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {"type": "string"},
}

_DELTA_TABLE_FIELD: dict[str, Any] = {
    "type": "string",
    "title": "Delta Table",
    "description": "Full three-part Unity Catalog Delta table name.",
    "x-widget": "resource-select",
    "x-source-kind": "delta_table",
    "x-value-field": "full_name",
    "x-label-field": "name",
    "x-allow-manual": True,
}

_WAREHOUSE_FIELD: dict[str, Any] = {
    "type": "string",
    "title": "SQL Warehouse ID",
    "description": "Databricks SQL warehouse id used for Statement Execution.",
}

_DELTA_COMMON_PROPERTIES: dict[str, Any] = {
    "table_name": _DELTA_TABLE_FIELD,
    "warehouse_id": _WAREHOUSE_FIELD,
    "columns": {
        **_STRING_ARRAY_SCHEMA,
        "title": "Columns",
        "description": "Columns to select. Use ['*'] only when the schema is unknown.",
        "default": ["*"],
    },
    "content_column": {
        "type": "string",
        "title": "Content Column",
        "description": "Text or JSON column returned as the primary row content.",
        "default": "content",
    },
    "order_by": {
        "type": "string",
        "title": "Order By",
        "description": "Column used to order chunks or rows.",
        "default": "chunk_id",
    },
    "exclude_chunk_types": {
        **_STRING_ARRAY_SCHEMA,
        "title": "Excluded Chunk Types",
        "description": "Optional chunk_type values to exclude from reads/searches.",
    },
}

_DELTA_READ_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": dict(_DELTA_COMMON_PROPERTIES),
    "required": ["table_name", "warehouse_id"],
}

_DELTA_GREP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        **_DELTA_COMMON_PROPERTIES,
        "date_column": {
            "type": "string",
            "title": "Date Column",
            "description": "Optional YYYY-MM/DD-like publication date column for year/month filters.",
        },
    },
    "required": ["table_name", "warehouse_id"],
}

_DELTA_TABLE_READ_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        **_DELTA_COMMON_PROPERTIES,
        "pk_column": {
            "type": "string",
            "title": "Primary Key Column",
            "description": "Column used for exact row lookup.",
            "default": "chunk_id",
        },
        "store_in_compute": {
            "type": "string",
            "title": "Compute Variable",
            "description": "Optional variable name for parsed JSON/table injection into compute.",
        },
        "compute_tool_name": {
            "type": "string",
            "title": "Compute Tool Name",
            "description": "Sibling compute tool used when store_in_compute is set.",
            "default": "compute",
        },
        "structural_analysis": {
            "type": "boolean",
            "title": "Structural Analysis",
            "description": "Return table structure and force numeric extraction through compute.",
            "default": False,
        },
    },
    "required": ["table_name", "warehouse_id"],
}

_COMPUTE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "allowed_modules": {
            **_STRING_ARRAY_SCHEMA,
            "title": "Allowed Modules",
            "description": "Replace the default import allow-list with these module names.",
        },
        "extra_modules": {
            **_STRING_ARRAY_SCHEMA,
            "title": "Extra Modules",
            "description": "Additional importable modules to expose in the sandbox.",
        },
        "max_execution_seconds": {
            "type": "number",
            "title": "Max Execution Seconds",
            "minimum": 1,
            "default": 10,
        },
        "max_output_chars": {
            "type": "integer",
            "title": "Max Output Characters",
            "minimum": 100,
            "default": 10000,
        },
        "max_code_length": {
            "type": "integer",
            "title": "Max Code Length",
            "minimum": 100,
            "default": 20000,
        },
    },
}

_COMPUTE_NAMESPACE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "compute_tool_name": {
            "type": "string",
            "title": "Compute Tool Name",
            "description": "Sibling compute tool whose namespace should be inspected.",
            "default": "compute",
        }
    },
}

_TABLE_READ_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "store_in_compute": {
            "type": "string",
            "title": "Compute Variable",
            "description": "Variable name for a table loaded from the runtime table registry.",
            "default": "web_table",
        },
        "compute_tool_name": {
            "type": "string",
            "title": "Compute Tool Name",
            "description": "Sibling compute tool used when store_in_compute is set.",
            "default": "compute",
        },
    },
}

_TOOL_KIND_META: dict[str, dict[str, Any]] = {
    "web_search": {
        "layer": "A",
        "config_schema": {
            "type": "object",
            "properties": {
                "max_results": {
                    "type": "integer",
                    "title": "Max Results",
                    "description": "Default number of search results returned per call.",
                    "minimum": 1,
                    "default": 10,
                }
            },
        },
    },
    "web_crawl": {"layer": "A", "config_schema": _EMPTY_SCHEMA},
    "web_research": {
        "layer": "A",
        "config_schema": {
            "type": "object",
            "properties": {
                "auto_fetch_top_k": {
                    "type": "integer",
                    "title": "Auto-Fetch Top K",
                    "description": "Number of search results to crawl automatically.",
                    "minimum": 1,
                    "default": 5,
                },
                "total_results": {
                    "type": "integer",
                    "title": "Total Results",
                    "description": "Number of search results to retrieve before auto-crawl.",
                    "minimum": 1,
                    "default": 10,
                },
                "max_body_chars": {
                    "type": "integer",
                    "title": "Max Body Characters",
                    "description": "Maximum extracted body characters per crawled result.",
                    "minimum": 1000,
                    "default": 8000,
                },
            },
        },
    },
    "file_search": {"layer": "A", "config_schema": _EMPTY_SCHEMA},
    "vector_search": {
        "layer": "B",
        "discoverable": True,
        "discovery_path": "vector_indexes",
        "config_schema": {
            "type": "object",
            "properties": {
                "index_name": {
                    "type": "string",
                    "title": "Vector Search Index",
                    "description": "Full three-part Databricks vector search index name.",
                    "x-widget": "resource-select",
                    "x-source-kind": "vector_index",
                    "x-value-field": "full_name",
                    "x-label-field": "name",
                    "x-allow-manual": True,
                },
                "num_results": {
                    "type": "integer",
                    "title": "Default Result Count",
                    "description": "Default number of results returned when the model calls this tool.",
                    "minimum": 1,
                    "default": 10,
                },
            },
            "required": ["index_name"],
        },
    },
    "genie": {
        "layer": "B",
        "discoverable": True,
        "discovery_path": "genie_spaces",
        "config_schema": {
            "type": "object",
            "properties": {
                "space_id": {
                    "type": "string",
                    "title": "Genie Space",
                    "description": "Databricks Genie space id.",
                    "x-widget": "resource-select",
                    "x-source-kind": "genie_space",
                    "x-value-field": "full_name",
                    "x-label-field": "name",
                    "x-allow-manual": True,
                }
            },
            "required": ["space_id"],
        },
    },
    "knowledge_assistant": {
        "layer": "B",
        "discoverable": True,
        "discovery_path": "knowledge_assistants",
        "config_schema": {
            "type": "object",
            "properties": {
                "endpoint_name": {
                    "type": "string",
                    "title": "Knowledge Assistant Endpoint",
                    "description": "Databricks knowledge assistant serving endpoint name.",
                    "x-widget": "resource-select",
                    "x-source-kind": "knowledge_assistant",
                    "x-value-field": "full_name",
                    "x-label-field": "name",
                    "x-allow-manual": True,
                }
            },
            "required": ["endpoint_name"],
        },
    },
    "compute": {"layer": "C", "config_schema": _COMPUTE_SCHEMA},
    "compute_namespace": {"layer": "C", "config_schema": _COMPUTE_NAMESPACE_SCHEMA},
    "delta_read": {
        "layer": "B",
        "discoverable": True,
        "discovery_path": "delta_tables",
        "config_schema": _DELTA_READ_SCHEMA,
    },
    "delta_grep": {
        "layer": "B",
        "discoverable": True,
        "discovery_path": "delta_tables",
        "config_schema": _DELTA_GREP_SCHEMA,
    },
    "delta_context": {
        "layer": "B",
        "discoverable": True,
        "discovery_path": "delta_tables",
        "config_schema": _DELTA_READ_SCHEMA,
    },
    "delta_table_read": {
        "layer": "B",
        "discoverable": True,
        "discovery_path": "delta_tables",
        "config_schema": _DELTA_TABLE_READ_SCHEMA,
    },
    "table_read": {"layer": "B", "config_schema": _TABLE_READ_SCHEMA},
    "custom": {"layer": "D", "config_schema": _EMPTY_SCHEMA},
}


def _agent_config_schema() -> dict[str, Any]:
    """Return AgentNodeConfig JSON schema with Designer UI hints."""
    schema = AgentNodeConfig.model_json_schema()
    properties = schema.setdefault("properties", {})
    if isinstance(properties, dict):
        for key in ("system_prompt", "user_prompt_template"):
            prop = properties.get(key)
            if isinstance(prop, dict):
                prop["x-widget"] = "prompt"
                prop["format"] = "multiline"
        subtype = properties.get("subtype")
        if isinstance(subtype, dict):
            subtype["enum"] = [item["id"] for item in AGENT_SUBTYPES]
        model_tier = properties.get("model_tier")
        if isinstance(model_tier, dict):
            model_tier["enum"] = model_tiers_payload()
    return schema


def _plan_and_execute_config_schema() -> dict[str, Any]:
    """Return PlanAndExecuteNodeConfig JSON schema with nested agent schemas."""
    schema = PlanAndExecuteNodeConfig.model_json_schema()
    properties = schema.setdefault("properties", {})
    agent_schema = _agent_config_schema()
    if isinstance(properties, dict):
        properties["planner"] = {
            **agent_schema,
            "title": "Planner",
            "description": "Planner agent config used to generate executable research steps.",
        }
        properties["evaluator"] = {
            **agent_schema,
            "title": "Evaluator",
            "description": "Optional evaluator/reflector agent config used after each step.",
        }
        planner_guidance = properties.get("planner_guidance")
        if isinstance(planner_guidance, dict):
            planner_guidance["x-widget"] = "prompt"
            planner_guidance["format"] = "multiline"
    return schema


def node_types_payload() -> list[dict[str, Any]]:
    """Build the node-type entries for the registry endpoint and the
    list_node_types chat tool. config_schema is pulled live via Pydantic.
    """
    out: list[dict[str, Any]] = []
    for nt in NodeType:
        meta = NODE_TYPE_META[nt]
        config_model = meta["config_model"]
        if nt is NodeType.agent:
            config_schema = _agent_config_schema()
        elif nt is NodeType.plan_and_execute:
            config_schema = _plan_and_execute_config_schema()
        elif config_model is not None:
            config_schema = config_model.model_json_schema()
        else:
            config_schema = _EMPTY_SCHEMA
        out.append({
            "type": nt.value,
            "label": meta["label"],
            "icon": meta["icon"],
            "category": meta["category"],
            "is_composite": meta["is_composite"],
            "config_schema": config_schema,
            "default_config": meta.get("default_config", {}),
            "summary_template": meta.get("summary_template", "{{type}}"),
            **({"children_kind": meta["children_kind"]} if "children_kind" in meta else {}),
            **({"children_slots": meta["children_slots"]} if "children_slots" in meta else {}),
            **({"branches_pairing": meta["branches_pairing"]} if "branches_pairing" in meta else {}),
        })
    return out


def tool_kinds_payload() -> list[dict[str, Any]]:
    """Enumerate supported tool kinds with editor metadata.

    Tool-name harmonization from the final plan remains a separate framework
    migration; this endpoint exposes the currently executable built-in
    ToolKind values with the richer registry shape expected by the editor.
    Bare ``kind: custom`` declarations are intentionally hidden here: the
    current framework uses that kind only for compile-time instance overrides,
    not for Designer-authored deployable YAML declarations. User-visible
    custom tool defs are appended separately by ``tool_kinds_payload_with_custom``.
    """
    payload: list[dict[str, Any]] = []
    for k in ToolKind:
        if k is ToolKind.custom:
            continue
        meta = _TOOL_KIND_META.get(k.value, {})
        payload.append({
            "kind": k.value,
            "label": k.value.replace("_", " ").title(),
            "icon": "tool",
            "layer": meta.get("layer", "D"),
            "config_schema": meta.get("config_schema", _EMPTY_SCHEMA),
            "discoverable": bool(meta.get("discoverable", False)),
            "discovery_path": meta.get("discovery_path"),
        })
    return payload


def model_tiers_payload() -> list[str]:
    """Return every model role configured for this app deployment."""
    return list(get_app_config().models.keys())


def query_modes_payload() -> list[str]:
    """Return query modes accepted by the job submission API."""
    return list(get_app_config().query_modes.model_dump().keys())


def research_depths_payload() -> list[str]:
    """Return research depths available in the main chat UI and job API."""
    research_types = get_app_config().research_types
    if research_types is None:
        return ["auto"]
    return ["auto", *list(research_types.model_dump().keys())]


def source_kinds_payload() -> list[dict[str, str]]:
    """Return source kinds understood by Designer source discovery."""
    return list(SOURCE_KINDS)


async def tool_kinds_payload_with_custom(
    session: AsyncSession | None = None,
    user_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return builtin tool kinds plus user-visible custom tool defs.

    Builtin entries (Layer A–C) come from the framework ToolKind enum.
    Layer D entries are custom tool defs owned by the user OR workspace-visible.

    Args:
        session: Optional async DB session.  When ``None``, only builtin kinds
                 are returned (same as ``tool_kinds_payload()``).
        user_id: The requesting user's ID.  Required when ``session`` is set.

    Returns:
        List of tool-kind dicts for the registry payload.
    """
    # Avoid circular import: CustomToolDef lives in models which imports db.base
    from deep_research.models.agent_v2 import CustomToolDef  # noqa: PLC0415

    payload: list[dict[str, Any]] = list(tool_kinds_payload())
    if session is not None and user_id is not None:
        stmt = select(CustomToolDef).where(
            or_(
                CustomToolDef.owner_id == user_id,
                CustomToolDef.visibility == "workspace",
            )
        )
        result = await session.execute(stmt)
        for tool in result.scalars():
            payload.append({
                "kind": tool.name,
                "label": tool.name,
                "layer": "D",
                "config_schema": tool.config_schema,
                "factory_ref": tool.factory_ref,
                "icon": "tool",
            })
    return payload
