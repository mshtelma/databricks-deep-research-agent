"""Tests verifying that tightened node config Pydantic models generate
editor-grade JSON schemas — the foundation for the Agent Designer V1
registry-driven UI form generator (Phase 0.5 task 0, Codex finding C1).

If these tests fail because someone weakened a typed field back to dict[str, Any],
the editor's form generator will produce free-form text inputs instead of
structured forms — that breaks the registry-driven extensibility contract.
"""
from __future__ import annotations

from typing import Any

from databricks_deep_research.agents.config import (
    ConditionalNodeConfig,
    LoopNodeConfig,
    PlanAndExecuteNodeConfig,
    ToolNodeConfig,
)


def _resolve_ref(schema: dict[str, Any], ref: str) -> dict[str, Any]:
    """Resolve a $ref like '#/$defs/ToolRef' inside the schema."""
    parts = ref.lstrip("#/").split("/")
    target: dict[str, Any] = schema
    for p in parts:
        target = target[p]
    return target


def _follow_field(schema: dict[str, Any], field_name: str) -> dict[str, Any]:
    """Get the inlined or $ref-resolved subschema for a top-level property."""
    prop: dict[str, Any] = schema["properties"][field_name]
    if "$ref" in prop:
        return _resolve_ref(schema, prop["$ref"])
    return prop


def test_tool_node_ref_is_typed() -> None:
    """ToolNodeConfig.ref must be an object with required string fields type+name.

    Pydantic emits ref as a $ref pointing to the ToolRef $def, which is an
    object with additionalProperties=false, properties={type: string, name: string},
    required=[type, name].
    """
    schema = ToolNodeConfig.model_json_schema()
    ref_schema = _follow_field(schema, "ref")

    assert ref_schema.get("type") == "object", (
        f"ref must be object, got {ref_schema}"
    )

    props = ref_schema.get("properties", {})
    assert "type" in props, "ref must have 'type' property"
    assert "name" in props, "ref must have 'name' property"

    assert props["type"].get("type") == "string", "ref.type must be string"
    assert props["name"].get("type") == "string", "ref.name must be string"

    required = ref_schema.get("required", [])
    assert "type" in required and "name" in required, (
        f"ref.type and ref.name must be required, got {required}"
    )

    # Sanity: ToolRef has extra="forbid" so additionalProperties must be False.
    assert ref_schema.get("additionalProperties") is False, (
        "ToolRef must have additionalProperties=false (extra='forbid'); "
        f"got {ref_schema.get('additionalProperties')!r}"
    )


def test_loop_until_is_discriminated_union() -> None:
    """LoopNodeConfig.until must be a discriminated union of >= 3 condition variants.

    Pydantic emits the Condition discriminated union directly on the property
    as a top-level oneOf (not via $ref), with a discriminator block pointing
    at the 'type' property of each variant.
    """
    schema = LoopNodeConfig.model_json_schema()
    until_prop = schema["properties"]["until"]

    # Pydantic emits discriminated unions as oneOf with discriminator info
    union_key = None
    for key in ("oneOf", "anyOf"):
        if key in until_prop:
            union_key = key
            break

    assert union_key is not None, (
        f"until must be a oneOf/anyOf discriminated union, got: {until_prop}"
    )

    variants = until_prop[union_key]
    assert len(variants) >= 3, (
        f"until must have >= 3 variants (StateCondition, LLMCondition, CompositeCondition), "
        f"got {len(variants)}: {variants}"
    )

    # Pydantic also emits a discriminator block with a propertyName of 'type'
    discriminator = until_prop.get("discriminator", {})
    assert discriminator.get("propertyName") == "type", (
        f"until discriminator must use propertyName='type', got {discriminator}"
    )

    # Resolve each variant $ref and verify each has a discriminator-like 'type' field
    for variant in variants:
        if "$ref" in variant:
            resolved = _resolve_ref(schema, variant["$ref"])
            props = resolved.get("properties", {})
            assert "type" in props, (
                f"Each condition variant must have a 'type' discriminator, "
                f"got props={list(props.keys())}"
            )


def test_conditional_uses_condition_branch() -> None:
    """ConditionalNodeConfig.conditions must be array of ConditionBranch objects.

    Pydantic emits conditions as type=array with items.$ref pointing to
    ConditionBranch in $defs, which has properties: condition (anyOf of the
    three condition types) and child_index (integer), both required.
    """
    schema = ConditionalNodeConfig.model_json_schema()
    conds_prop = schema["properties"]["conditions"]

    assert conds_prop.get("type") == "array", (
        f"conditions must be array, got {conds_prop}"
    )

    items = conds_prop.get("items", {})
    if "$ref" in items:
        items = _resolve_ref(schema, items["$ref"])

    item_props = items.get("properties", {})
    assert "condition" in item_props, (
        f"ConditionBranch must have 'condition' property, got {list(item_props.keys())}"
    )
    assert "child_index" in item_props, (
        f"ConditionBranch must have 'child_index' property, got {list(item_props.keys())}"
    )

    assert item_props["child_index"].get("type") == "integer", (
        "child_index must be integer"
    )

    # condition itself must be a union of condition variants (anyOf), not free-form
    condition_schema = item_props["condition"]
    assert "anyOf" in condition_schema or "$ref" in condition_schema, (
        f"ConditionBranch.condition must be a union (anyOf) or $ref, got {condition_schema}"
    )

    # ConditionBranch must require both fields
    required = items.get("required", [])
    assert "condition" in required, "ConditionBranch.condition must be required"
    assert "child_index" in required, "ConditionBranch.child_index must be required"


def test_plan_and_execute_body_is_workflow_node() -> None:
    """PlanAndExecuteNodeConfig.body must be a WorkflowNode-shaped object (or None).

    body is Optional[WorkflowNode] = None, so Pydantic emits:
      anyOf: [{$ref: #/$defs/WorkflowNode}, {type: null}]
    WorkflowNode has properties: id (string), type (NodeType enum), label (string),
    config (dict), children (array), plus optional error_handling and budget_seconds.
    """
    schema = PlanAndExecuteNodeConfig.model_json_schema()
    body_prop = schema["properties"]["body"]

    # body is Optional[WorkflowNode] = None -> anyOf [$ref, null]
    workflow_node_schema = None
    if "$ref" in body_prop:
        workflow_node_schema = _resolve_ref(schema, body_prop["$ref"])
    elif "anyOf" in body_prop:
        for variant in body_prop["anyOf"]:
            if "$ref" in variant:
                workflow_node_schema = _resolve_ref(schema, variant["$ref"])
                break
    elif "type" in body_prop:
        workflow_node_schema = body_prop

    assert workflow_node_schema is not None, (
        f"body must resolve to a WorkflowNode schema, got {body_prop}"
    )

    # WorkflowNode has id, type, label, config, children
    props = workflow_node_schema.get("properties", {})
    for required_prop in ("id", "type", "label"):
        assert required_prop in props, (
            f"WorkflowNode body must have '{required_prop}' property, "
            f"got {list(props.keys())}"
        )

    # id and label must be strings
    assert props["id"].get("type") == "string", "WorkflowNode.id must be string"
    assert props["label"].get("type") == "string", "WorkflowNode.label must be string"


def test_no_dict_str_any_leaks_in_critical_fields() -> None:
    """Sanity: none of the 4 tightened fields should leak as untyped object schema.

    A free-form dict/object has type=object AND no properties AND
    additionalProperties=True (or omitted, which defaults to permissive).
    """

    def _is_free_form_dict(field_schema: dict[str, Any]) -> bool:
        """A free-form dict has type=object AND no properties AND
        additionalProperties=True (or default)."""
        if field_schema.get("type") != "object":
            return False
        if "properties" in field_schema and field_schema["properties"]:
            return False
        # additionalProperties not False or schema => free-form
        ap = field_schema.get("additionalProperties", True)
        return ap is True or (isinstance(ap, dict) and ap == {})

    # ToolNodeConfig.ref must NOT be free-form
    tnc_schema = ToolNodeConfig.model_json_schema()
    tnc_ref = _follow_field(tnc_schema, "ref")
    assert not _is_free_form_dict(tnc_ref), (
        f"ToolNodeConfig.ref leaked as free-form dict: {tnc_ref}"
    )

    # LoopNodeConfig.until must NOT be free-form
    lnc_schema = LoopNodeConfig.model_json_schema()
    lnc_until = lnc_schema["properties"]["until"]
    # union form (oneOf/anyOf) is fine; only fail if it's a bare free-form object
    if (
        "type" in lnc_until
        and "$ref" not in lnc_until
        and "oneOf" not in lnc_until
        and "anyOf" not in lnc_until
    ):
        assert not _is_free_form_dict(lnc_until), (
            f"LoopNodeConfig.until leaked as free-form dict: {lnc_until}"
        )

    # ConditionalNodeConfig.conditions array items must NOT be free-form
    cnc_schema = ConditionalNodeConfig.model_json_schema()
    cnc_items = cnc_schema["properties"]["conditions"].get("items", {})
    if "$ref" in cnc_items:
        cnc_items = _resolve_ref(cnc_schema, cnc_items["$ref"])
    assert not _is_free_form_dict(cnc_items), (
        f"ConditionalNodeConfig.conditions[*] leaked as free-form dict: {cnc_items}"
    )

    # PlanAndExecuteNodeConfig.body (resolved WorkflowNode) must NOT be free-form
    pae_schema = PlanAndExecuteNodeConfig.model_json_schema()
    body_prop = pae_schema["properties"]["body"]
    wf_node: dict[str, Any] | None = None
    if "$ref" in body_prop:
        wf_node = _resolve_ref(pae_schema, body_prop["$ref"])
    elif "anyOf" in body_prop:
        for variant in body_prop["anyOf"]:
            if "$ref" in variant:
                wf_node = _resolve_ref(pae_schema, variant["$ref"])
                break
    if wf_node is not None:
        assert not _is_free_form_dict(wf_node), (
            f"PlanAndExecuteNodeConfig.body (WorkflowNode) leaked as free-form dict: {wf_node}"
        )
