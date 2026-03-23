from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NormalizedPlanContract:
    title: str
    thought: str
    has_enough_context: bool
    raw_items: list[Any]
    items: list[Any]
    repair_mode: str | None = None
    repair_reason: str | None = None


def is_executable_plan_item(item: Any) -> bool:
    if item is None:
        return False
    if isinstance(item, str):
        return bool(item.strip())
    title = str(getattr(item, "title", "") or "")
    description = str(getattr(item, "description", "") or "")
    if isinstance(item, dict):
        title = str(item.get("title", "") or "")
        description = str(item.get("description", "") or item.get("task", "") or "")
    return bool(title.strip() or description.strip())


def synthesize_plan_item(_plan_data: Any, title: str, thought: str) -> dict[str, Any] | None:
    has_substantive_context = bool((title or "").strip() and (thought or "").strip())
    if not has_substantive_context:
        return None
    return {"id": "step-1", "title": (title or "").strip(), "description": (thought or "").strip()}


def extract_raw_plan_contract(plan_data: Any, items_path: str, normalize_plan_contract: Any) -> NormalizedPlanContract:
    contract = normalize_plan_contract(plan_data, items_path)
    raw_items = list(contract["items"])
    return NormalizedPlanContract(
        title=str(contract["title"]),
        thought=str(contract["thought"]),
        has_enough_context=bool(contract["has_enough_context"]),
        raw_items=raw_items,
        items=raw_items,
    )


def finalize_plan_contract(contract: NormalizedPlanContract, plan_data: Any) -> NormalizedPlanContract:
    items = [item for item in contract.raw_items if is_executable_plan_item(item)]
    if items:
        return NormalizedPlanContract(
            title=contract.title,
            thought=contract.thought,
            has_enough_context=contract.has_enough_context,
            raw_items=contract.raw_items,
            items=items,
        )
    synthesized = synthesize_plan_item(plan_data, contract.title, contract.thought)
    if synthesized is not None:
        return NormalizedPlanContract(
            title=contract.title,
            thought=contract.thought,
            has_enough_context=contract.has_enough_context,
            raw_items=contract.raw_items,
            items=[synthesized],
            repair_mode="synthesized_from_empty_steps",
            repair_reason="planner_returned_no_executable_steps",
        )
    return NormalizedPlanContract(
        title=contract.title,
        thought=contract.thought,
        has_enough_context=contract.has_enough_context,
        raw_items=contract.raw_items,
        items=[],
        repair_mode="none",
        repair_reason="planner_returned_no_executable_steps",
    )


def normalize_executable_plan_contract(plan_data: Any, items_path: str, normalize_plan_contract: Any) -> dict[str, Any]:
    finalized = finalize_plan_contract(extract_raw_plan_contract(plan_data, items_path, normalize_plan_contract), plan_data)
    return {
        "items": finalized.items,
        "has_enough_context": finalized.has_enough_context,
        "title": finalized.title,
        "thought": finalized.thought,
        "repair_mode": finalized.repair_mode,
        "repair_reason": finalized.repair_reason,
        "raw_item_count": len(finalized.raw_items),
        "normalized_item_count": len(finalized.items),
    }
