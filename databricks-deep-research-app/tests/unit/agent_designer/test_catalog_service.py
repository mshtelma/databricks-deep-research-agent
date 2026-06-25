from __future__ import annotations

from deep_research.agent_designer.catalog_service import CatalogService


def _workflow() -> dict[str, object]:
    return {
        "id": "wf",
        "name": "Workflow",
        "version": 1,
        "root": {
            "id": "researcher",
            "type": "agent",
            "label": "Researcher",
            "config": {
                "subtype": "researcher",
                "system_prompt": "{tool_catalog}",
                "user_prompt_template": "{query}",
                "tools": ["docs"],
            },
            "children": [],
        },
        "tools": [
            {
                "name": "docs",
                "kind": "vector_search",
                "config": {"index_name": "catalog.schema.index"},
            }
        ],
    }


def test_materialize_for_save_stamps_catalog_extras() -> None:
    materialized = CatalogService().materialize_for_save(_workflow())
    config = materialized["root"]["config"]  # type: ignore[index]
    extras = config["extras"]  # type: ignore[index]

    assert materialized["schema_version"] == 1
    assert extras["_framework_tool_catalog_injection_enabled"] is True
    assert extras["_framework_tool_catalog_registry_version"] == "1"
    assert extras["_framework_tool_catalog_decls_hash"]
    assert "vector_search" in extras["_framework_tool_catalog"]


def test_materialize_for_save_preserves_user_edits_without_force() -> None:
    workflow = _workflow()
    config = workflow["root"]["config"]  # type: ignore[index]
    config["extras"] = {  # type: ignore[index]
        "_framework_tool_catalog": "custom prose",
        "_framework_tool_catalog_user_edited": True,
    }

    materialized = CatalogService().materialize_for_save(workflow)
    extras = materialized["root"]["config"]["extras"]  # type: ignore[index]

    assert extras["_framework_tool_catalog"] == "custom prose"


def test_force_regen_clears_user_edit_and_rerenders() -> None:
    workflow = _workflow()
    config = workflow["root"]["config"]  # type: ignore[index]
    config["extras"] = {  # type: ignore[index]
        "_framework_tool_catalog": "custom prose",
        "_framework_tool_catalog_user_edited": True,
    }

    materialized = CatalogService().materialize_for_save(workflow, force_regen=True)
    extras = materialized["root"]["config"]["extras"]  # type: ignore[index]

    assert extras["_framework_tool_catalog_user_edited"] is False
    assert extras["_framework_tool_catalog"] != "custom prose"
