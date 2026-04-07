from __future__ import annotations

from typing import Any

from databricks_deep_research.workflow.runtime.planner_contracts import (
    NormalizedPlanContract as NormalizedPlanContract,
)
from databricks_deep_research.workflow.runtime.planner_contracts import (
    extract_raw_plan_contract as _extract_raw_plan_contract_impl,
)
from databricks_deep_research.workflow.runtime.planner_contracts import (
    finalize_plan_contract as _finalize_plan_contract_impl,
)
from databricks_deep_research.workflow.runtime.planner_contracts import (
    normalize_executable_plan_contract as _normalize_executable_plan_contract_impl,
)


def normalize_plan_contract(plan_data: Any, items_path: str) -> dict[str, Any]:
    title = ""
    thought = ""
    has_enough_context = False
    if isinstance(plan_data, dict):
        title = str(plan_data.get("title", "") or "")
        thought = str(plan_data.get("thought", "") or plan_data.get("summary", "") or "")
        has_enough_context = bool(plan_data.get("has_enough_context", False))
    elif hasattr(plan_data, "title") or hasattr(plan_data, items_path.split(".")[0]):
        title = str(getattr(plan_data, "title", "") or "")
        thought = str(getattr(plan_data, "thought", "") or getattr(plan_data, "summary", "") or "")
        has_enough_context = bool(getattr(plan_data, "has_enough_context", False))

    current = plan_data
    if isinstance(current, str):
        import json
        try:
            current = json.loads(current)
        except (json.JSONDecodeError, ValueError):
            if "```json" in current:
                try:
                    start = current.index("```json") + 7
                    end = current.index("```", start)
                    current = json.loads(current[start:end].strip())
                except (json.JSONDecodeError, ValueError, IndexError):
                    pass
    for part in items_path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            current = []
            break
        if current is None:
            current = []
            break
    items = current if isinstance(current, list) else [current] if current else []
    return {
        "items": items,
        "has_enough_context": has_enough_context,
        "title": title,
        "thought": thought,
    }


def extract_raw_plan_contract(plan_data: Any, items_path: str) -> NormalizedPlanContract:
    return _extract_raw_plan_contract_impl(plan_data, items_path, normalize_plan_contract)


def finalize_plan_contract(contract: NormalizedPlanContract, plan_data: Any) -> NormalizedPlanContract:
    return _finalize_plan_contract_impl(contract, plan_data)


def normalize_executable_plan_contract(plan_data: Any, items_path: str) -> dict[str, Any]:
    return _normalize_executable_plan_contract_impl(plan_data, items_path, normalize_plan_contract)
