"""Static condition contract validation for workflow definitions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from pydantic import BaseModel

from databricks_deep_research.agents.config import (
    AgentNodeConfig,
    ConditionalNodeConfig,
    LoopNodeConfig,
    PlanAndExecuteNodeConfig,
    SubworkflowNodeConfig,
    ToolNodeConfig,
)
from databricks_deep_research.agents.output_models import (
    BackgroundOutput,
    CoordinatorOutput,
    EvaluationOutput,
    PlanOutput,
    ReflectionOutput,
    ResearcherOutput,
    SynthesizerOutput,
)
from databricks_deep_research.templates.renderer import SafeTemplateRenderer
from databricks_deep_research.workflow.conditions import (
    CompositeCondition,
    LLMCondition,
    StateCondition,
)
from databricks_deep_research.workflow.definition import (
    NodeType,
    WorkflowDefinition,
    WorkflowNode,
)
from databricks_deep_research.workflow.runtime_keys import runtime_seed_keys

TypeKind = Literal[
    "unknown",
    "string",
    "number",
    "integer",
    "boolean",
    "object",
    "array",
    "enum",
    "null",
]
Availability = Literal["always", "maybe"]


@dataclass(frozen=True)
class FieldSchema:
    """Small internal schema representation used by condition validation."""

    type_kind: TypeKind = "unknown"
    required: bool = True
    enum_values: frozenset[Any] = frozenset()
    item_schema: FieldSchema | None = None
    properties: Mapping[str, FieldSchema] = field(default_factory=dict)


@dataclass(frozen=True)
class StateBinding:
    """A named state value available at a point in the workflow graph."""

    key: str
    schema: FieldSchema
    producer_node_id: str | None
    availability: Availability = "always"
    origin: str = ""


@dataclass(frozen=True)
class ValidationScope:
    """Condition validation scope with known state bindings."""

    bindings: Mapping[str, StateBinding]
    path: str

    def bind(
        self,
        key: str,
        schema: FieldSchema,
        *,
        producer_node_id: str | None,
        availability: Availability = "always",
        origin: str = "",
    ) -> ValidationScope:
        next_bindings = dict(self.bindings)
        next_bindings[key] = StateBinding(
            key=key,
            schema=schema,
            producer_node_id=producer_node_id,
            availability=availability,
            origin=origin,
        )
        return ValidationScope(bindings=next_bindings, path=self.path)


@dataclass(frozen=True)
class ResolvedPath:
    """Result of resolving a condition path against a validation scope."""

    binding: StateBinding
    schema: FieldSchema


_UNKNOWN_SCHEMA = FieldSchema(type_kind="unknown")

_BUILTIN_OUTPUT_MODELS: dict[str, type[BaseModel]] = {
    "background": BackgroundOutput,
    "coordinator": CoordinatorOutput,
    "evaluator": EvaluationOutput,
    "planner": PlanOutput,
    "reflector": ReflectionOutput,
    "researcher": ResearcherOutput,
    "synthesizer": SynthesizerOutput,
}

_STATE_OPERATORS = frozenset(
    {
        "eq",
        "neq",
        "gt",
        "lt",
        "gte",
        "lte",
        "contains",
        "in",
        "exists",
        "not_exists",
    }
)


def _initial_scope(definition: WorkflowDefinition) -> ValidationScope:
    """Seed the root validation scope with state available before any node runs.

    Two origins are always-available at the root:

    * ``required_inputs`` — caller-supplied workflow inputs.
    * runtime-mediated keys — the per-workflow ``runtime_injected_keys`` plus the
      framework-global registry (see :func:`runtime_seed_keys`). These are
      injected into state by the harness/runner rather than produced by a node
      ``output_key``; conditions may legitimately read them. Seeding them here
      keeps this strict validator aligned with the dataflow checker's Pass-A seed
      (``dataflow_contracts.dangling_reads``).

    Runtime-mediated keys carry an unknown schema, so existence/equality checks
    on them pass while nested dotted access still reports "descends into unknown".
    ``required_inputs`` win on any key collision (via ``setdefault``).
    """
    bindings: dict[str, StateBinding] = {
        key: StateBinding(
            key=key,
            schema=_UNKNOWN_SCHEMA,
            producer_node_id=None,
            availability="always",
            origin="required_inputs",
        )
        for key in definition.required_inputs
    }
    for key in runtime_seed_keys(definition.runtime_injected_keys):
        bindings.setdefault(
            key,
            StateBinding(
                key=key,
                schema=_UNKNOWN_SCHEMA,
                producer_node_id=None,
                availability="always",
                origin="runtime_injected",
            ),
        )
    return ValidationScope(bindings=bindings, path="root")


def validate_condition_contracts(definition: WorkflowDefinition) -> list[str]:
    """Return static condition contract errors for *definition*."""

    scope = _initial_scope(definition)
    errors: list[str] = []
    _validate_node(definition.root, scope, "root", errors)
    return errors


def _validate_node(
    node: WorkflowNode,
    scope: ValidationScope,
    path: str,
    errors: list[str],
) -> ValidationScope:
    if node.type is NodeType.agent:
        return _validate_agent_node(node, scope)

    if node.type is NodeType.tool:
        return _validate_tool_node(node, scope)

    if node.type is NodeType.subworkflow:
        return _validate_subworkflow_node(node, scope, errors)

    if node.type is NodeType.sequence:
        current = scope
        for index, child in enumerate(node.children):
            current = _validate_node(child, current, f"{path}.children[{index}]", errors)
        return current

    if node.type is NodeType.parallel:
        branch_scopes = [
            _validate_node(child, scope, f"{path}.children[{index}]", errors)
            for index, child in enumerate(node.children)
        ]
        return _merge_branch_outputs(scope, branch_scopes)

    if node.type is NodeType.conditional:
        return _validate_conditional_node(node, scope, path, errors)

    if node.type is NodeType.loop:
        return _validate_loop_node(node, scope, path, errors)

    if node.type is NodeType.plan_and_execute:
        return _validate_plan_and_execute_node(node, scope, path, errors)

    return scope


def _validate_agent_node(node: WorkflowNode, scope: ValidationScope) -> ValidationScope:
    config = AgentNodeConfig(**node.config)
    return scope.bind(
        config.output_key,
        _agent_output_schema(config),
        producer_node_id=node.id,
        origin=f"agent:{config.subtype}",
    )


def _validate_tool_node(node: WorkflowNode, scope: ValidationScope) -> ValidationScope:
    config = ToolNodeConfig(**node.config)
    schema = _schema_from_json_schema(config.output_schema) if config.output_schema else _UNKNOWN_SCHEMA
    return scope.bind(
        config.output_key,
        schema,
        producer_node_id=node.id,
        origin="tool",
    )


def _validate_subworkflow_node(
    node: WorkflowNode,
    scope: ValidationScope,
    errors: list[str],
) -> ValidationScope:
    config = SubworkflowNodeConfig(**node.config)
    for target, source in config.input_mapping.items():
        _validate_state_path(
            StateCondition(key=source, operator="exists"),
            scope,
            f"node '{node.id}' config.input_mapping[{target!r}]",
            errors,
        )

    next_scope = scope.bind(
        config.output_key,
        _UNKNOWN_SCHEMA,
        producer_node_id=node.id,
        origin="subworkflow",
    )
    for target in config.output_mapping.values():
        next_scope = next_scope.bind(
            target,
            _UNKNOWN_SCHEMA,
            producer_node_id=node.id,
            origin="subworkflow.output_mapping",
        )
    return next_scope


def _validate_conditional_node(
    node: WorkflowNode,
    scope: ValidationScope,
    path: str,
    errors: list[str],
) -> ValidationScope:
    config = ConditionalNodeConfig(**node.config)
    for index, branch in enumerate(config.conditions):
        _validate_condition(
            branch.condition,
            scope,
            f"{path}.config.conditions[{index}].condition",
            errors,
        )

    branch_scopes = [
        _validate_node(child, scope, f"{path}.children[{index}]", errors)
        for index, child in enumerate(node.children)
    ]
    return _merge_branch_outputs(scope, branch_scopes)


def _validate_loop_node(
    node: WorkflowNode,
    scope: ValidationScope,
    path: str,
    errors: list[str],
) -> ValidationScope:
    config = LoopNodeConfig(**node.config)
    current = scope
    for index, child in enumerate(node.children):
        current = _validate_node(child, current, f"{path}.children[{index}]", errors)
    _validate_condition(config.until, current, f"{path}.config.until", errors)

    if config.min_iterations < 1:
        return _mark_new_bindings_maybe(scope, current)
    return current


def _validate_plan_and_execute_node(
    node: WorkflowNode,
    scope: ValidationScope,
    path: str,
    errors: list[str],
) -> ValidationScope:
    config = PlanAndExecuteNodeConfig(**node.config)
    planner_config = AgentNodeConfig(**config.planner)
    planner_schema = _agent_output_schema(planner_config)
    planner_scope = scope.bind(
        planner_config.output_key,
        planner_schema,
        producer_node_id=f"{node.id}.planner",
        origin=f"plan_and_execute.planner:{planner_config.subtype}",
    )

    item_schema = _resolve_schema_path(planner_schema, config.items_path)
    if item_schema is None:
        errors.append(
            f"{path}.config.items_path: Plan item path {config.items_path!r} is not "
            f"declared in planner output schema for node '{node.id}'."
        )
        item_schema = _UNKNOWN_SCHEMA
    elif item_schema.type_kind == "array":
        if item_schema.item_schema is None:
            errors.append(
                f"{path}.config.items_path: Plan item path {config.items_path!r} "
                f"does not declare an item schema."
            )
            item_schema = _UNKNOWN_SCHEMA
        else:
            item_schema = item_schema.item_schema

    loop_scope = planner_scope.bind(
        config.item_state_key,
        item_schema,
        producer_node_id=f"{node.id}.planner_item",
        origin=f"plan item for {planner_config.output_key}.{config.items_path}",
    )

    body_result = loop_scope
    if config.body is not None:
        body_result = _validate_node(config.body, loop_scope, f"{path}.config.body", errors)

    evaluator_result = body_result
    if config.evaluator is not None:
        evaluator_config = AgentNodeConfig(**config.evaluator)
        evaluator_result = evaluator_result.bind(
            evaluator_config.output_key,
            _agent_output_schema(evaluator_config),
            producer_node_id=f"{node.id}.evaluator",
            origin=f"plan_and_execute.evaluator:{evaluator_config.subtype}",
        )

    outer = planner_scope
    for key, binding in evaluator_result.bindings.items():
        if key in loop_scope.bindings:
            continue
        outer = outer.bind(
            key,
            binding.schema,
            producer_node_id=binding.producer_node_id,
            availability=binding.availability,
            origin=binding.origin,
        )
    return outer


def _validate_condition(
    condition: StateCondition | LLMCondition | CompositeCondition,
    scope: ValidationScope,
    path: str,
    errors: list[str],
) -> None:
    if isinstance(condition, StateCondition):
        _validate_state_path(condition, scope, f"{path}.key", errors)
        return

    if isinstance(condition, LLMCondition):
        _validate_llm_condition(condition, scope, path, errors)
        return

    if isinstance(condition, CompositeCondition):
        if condition.operator not in {"all", "any", "not"}:
            errors.append(
                f"{path}.operator: Unknown composite operator {condition.operator!r}."
            )
        if condition.operator in {"all", "any"} and not condition.conditions:
            errors.append(
                f"{path}.conditions: Composite operator {condition.operator!r} "
                "requires at least one child condition."
            )
        if condition.operator == "not" and len(condition.conditions) != 1:
            errors.append(
                f"{path}.conditions: Composite operator 'not' requires exactly "
                "one child condition."
            )
        for index, child in enumerate(condition.conditions):
            _validate_condition(child, scope, f"{path}.conditions[{index}]", errors)


def _validate_llm_condition(
    condition: LLMCondition,
    scope: ValidationScope,
    path: str,
    errors: list[str],
) -> None:
    renderer = SafeTemplateRenderer()
    try:
        variables = renderer.extract_variables(condition.prompt_template)
    except Exception as exc:
        errors.append(f"{path}.prompt_template: Invalid LLM condition template: {exc}")
        return

    for variable in sorted(variables):
        _validate_state_path(
            StateCondition(key=variable, operator="exists"),
            scope,
            f"{path}.prompt_template[{variable!r}]",
            errors,
        )


def _validate_state_path(
    condition: StateCondition,
    scope: ValidationScope,
    path: str,
    errors: list[str],
) -> None:
    operator = condition.operator
    if operator not in _STATE_OPERATORS:
        errors.append(f"{path}: Unknown state condition operator {operator!r}.")
        return

    resolved = _resolve_condition_path(condition.key, scope)
    if isinstance(resolved, str):
        errors.append(f"{path}: {resolved}")
        return

    schema = resolved.schema
    binding = resolved.binding
    if binding.availability == "maybe" and operator not in {"exists", "not_exists"}:
        errors.append(
            f"{path}: State condition path {condition.key!r} is produced only by "
            "some branches. Use an existence check first or make all branches "
            "produce the same output key."
        )
        return

    if operator == "exists":
        return

    if operator == "not_exists":
        if binding.availability == "always" and schema.required:
            errors.append(
                f"{path}: State condition path {condition.key!r} is declared as "
                "required and always available, so 'not_exists' can never match."
            )
        return

    if schema.type_kind == "unknown":
        return

    if operator in {"eq", "neq"}:
        _validate_single_value(condition.key, schema, condition.value, path, errors)
        return

    if operator == "in":
        candidates = condition.value
        if not isinstance(candidates, (list, tuple, set, frozenset)):
            errors.append(
                f"{path}: State condition path {condition.key!r} uses 'in' but "
                "condition.value is not a list."
            )
            return
        for candidate in candidates:
            _validate_single_value(condition.key, schema, candidate, path, errors)
        return

    if operator == "contains":
        if schema.type_kind == "string":
            if not isinstance(condition.value, str):
                errors.append(
                    f"{path}: State condition path {condition.key!r} is a string "
                    "but 'contains' value is not a string."
                )
            return
        if schema.type_kind == "array":
            item_schema = schema.item_schema or _UNKNOWN_SCHEMA
            _validate_single_value(condition.key, item_schema, condition.value, path, errors)
            return
        errors.append(
            f"{path}: State condition path {condition.key!r} has type "
            f"{schema.type_kind!r}; 'contains' requires a string or array."
        )
        return

    if operator in {"gt", "lt", "gte", "lte"}:
        if schema.type_kind not in {"number", "integer"}:
            errors.append(
                f"{path}: State condition path {condition.key!r} has type "
                f"{schema.type_kind!r}; {operator!r} requires a numeric field."
            )
            return
        if not _is_number(condition.value):
            errors.append(
                f"{path}: State condition path {condition.key!r} uses {operator!r} "
                "but condition.value is not numeric."
            )


def _validate_single_value(
    condition_key: str,
    schema: FieldSchema,
    value: Any,
    path: str,
    errors: list[str],
) -> None:
    if schema.enum_values and value not in schema.enum_values:
        allowed = ", ".join(repr(item) for item in sorted(schema.enum_values, key=repr))
        errors.append(
            f"{path}: State condition path {condition_key!r} compares against "
            f"{value!r}, but allowed values are: {allowed}."
        )
        return

    if not _value_matches_schema(schema, value):
        errors.append(
            f"{path}: State condition path {condition_key!r} has type "
            f"{schema.type_kind!r}, incompatible with value {value!r}."
        )


def _resolve_condition_path(
    key: str,
    scope: ValidationScope,
) -> ResolvedPath | str:
    parts = [part for part in key.split(".") if part]
    if not parts:
        return "State condition key must be a non-empty dot path."

    binding = scope.bindings.get(parts[0])
    if binding is None:
        available = _format_available(scope.bindings)
        return (
            f"State condition path {key!r} is not declared in the available "
            f"workflow state. Available top-level keys: {available}."
        )

    schema = binding.schema
    for segment in parts[1:]:
        if schema.type_kind == "unknown":
            return (
                f"State condition path {key!r} descends into unknown state "
                f"binding {binding.key!r}. Declare an output_schema before "
                "using nested condition paths."
            )
        if schema.type_kind != "object":
            return (
                f"State condition path {key!r} cannot access {segment!r} "
                f"because the current schema is {schema.type_kind!r}."
            )
        child = schema.properties.get(segment)
        if child is None:
            available = _format_available(schema.properties)
            return (
                f"State condition path {key!r} is not declared in the available "
                f"state schema. {parts[0]!r} supports: {available}. Either "
                "remove the router, use parallel_lanes, or declare a typed "
                "upstream discriminator in output_schema."
            )
        schema = child

    return ResolvedPath(binding=binding, schema=schema)


def _resolve_schema_path(schema: FieldSchema, path: str) -> FieldSchema | None:
    current: FieldSchema | None = schema
    for segment in [part for part in path.split(".") if part]:
        if current is None or current.type_kind != "object":
            return None
        current = current.properties.get(segment)
    return current


def _agent_output_schema(config: AgentNodeConfig) -> FieldSchema:
    if config.output_schema:
        return _schema_from_json_schema(config.output_schema)

    model = _BUILTIN_OUTPUT_MODELS.get(config.subtype)
    if model is None:
        return _UNKNOWN_SCHEMA
    return _schema_from_json_schema(model.model_json_schema())


def _schema_from_json_schema(schema: Mapping[str, Any]) -> FieldSchema:
    return _convert_json_schema(schema, schema, required=True, seen=frozenset())


def _convert_json_schema(
    schema: Mapping[str, Any],
    root: Mapping[str, Any],
    *,
    required: bool,
    seen: frozenset[str],
) -> FieldSchema:
    ref = schema.get("$ref")
    if isinstance(ref, str):
        if ref in seen:
            return replace(_UNKNOWN_SCHEMA, required=required)
        resolved = _resolve_ref(ref, root)
        if resolved is None:
            return replace(_UNKNOWN_SCHEMA, required=required)
        return _convert_json_schema(
            resolved,
            root,
            required=required,
            seen=seen | {ref},
        )

    variants = schema.get("anyOf") or schema.get("oneOf")
    if isinstance(variants, list):
        non_null = [
            item
            for item in variants
            if isinstance(item, Mapping) and item.get("type") != "null"
        ]
        if len(non_null) == 1:
            return _convert_json_schema(
                non_null[0],
                root,
                required=required,
                seen=seen,
            )
        return replace(_UNKNOWN_SCHEMA, required=required)

    enum_values = schema.get("enum")
    if isinstance(enum_values, list):
        return FieldSchema(
            type_kind="enum",
            required=required,
            enum_values=frozenset(enum_values),
        )

    raw_type = schema.get("type")
    if isinstance(raw_type, list):
        raw_type = next((item for item in raw_type if item != "null"), None)

    if raw_type == "object" or "properties" in schema:
        required_fields = set(schema.get("required", []))
        raw_properties = schema.get("properties", {})
        properties: dict[str, FieldSchema] = {}
        if isinstance(raw_properties, Mapping):
            for name, child_schema in raw_properties.items():
                if isinstance(name, str) and isinstance(child_schema, Mapping):
                    properties[name] = _convert_json_schema(
                        child_schema,
                        root,
                        required=name in required_fields,
                        seen=seen,
                    )
        return FieldSchema(
            type_kind="object",
            required=required,
            properties=properties,
        )

    if raw_type == "array":
        item_schema = _UNKNOWN_SCHEMA
        raw_items = schema.get("items")
        if isinstance(raw_items, Mapping):
            item_schema = _convert_json_schema(
                raw_items,
                root,
                required=True,
                seen=seen,
            )
        return FieldSchema(
            type_kind="array",
            required=required,
            item_schema=item_schema,
        )

    return FieldSchema(
        type_kind=_json_type_kind(raw_type),
        required=required,
    )


def _resolve_ref(ref: str, root: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if not ref.startswith("#/"):
        return None
    current: Any = root
    for part in ref[2:].split("/"):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    if isinstance(current, Mapping):
        return current
    return None


def _json_type_kind(raw_type: Any) -> TypeKind:
    if raw_type == "string":
        return "string"
    if raw_type == "number":
        return "number"
    if raw_type == "integer":
        return "integer"
    if raw_type == "boolean":
        return "boolean"
    if raw_type == "object":
        return "object"
    if raw_type == "array":
        return "array"
    if raw_type == "null":
        return "null"
    return "unknown"


def _merge_branch_outputs(
    incoming: ValidationScope,
    branch_scopes: Sequence[ValidationScope],
) -> ValidationScope:
    if not branch_scopes:
        return incoming

    incoming_keys = set(incoming.bindings)
    branch_outputs: list[dict[str, StateBinding]] = []
    for branch_scope in branch_scopes:
        branch_outputs.append(
            {
                key: binding
                for key, binding in branch_scope.bindings.items()
                if key not in incoming_keys
            }
        )

    merged = incoming
    produced_keys = set().union(*(set(outputs) for outputs in branch_outputs))
    for key in produced_keys:
        producers = [outputs[key] for outputs in branch_outputs if key in outputs]
        first = producers[0]
        availability: Availability = (
            "always" if all(key in outputs for outputs in branch_outputs) else "maybe"
        )
        schema = first.schema
        if any(producer.schema != schema for producer in producers[1:]):
            schema = _UNKNOWN_SCHEMA
        merged = merged.bind(
            key,
            schema,
            producer_node_id=first.producer_node_id,
            availability=availability,
            origin=first.origin,
        )
    return merged


def _mark_new_bindings_maybe(
    incoming: ValidationScope,
    current: ValidationScope,
) -> ValidationScope:
    incoming_keys = set(incoming.bindings)
    next_scope = incoming
    for key, binding in current.bindings.items():
        if key in incoming_keys:
            continue
        next_scope = next_scope.bind(
            key,
            binding.schema,
            producer_node_id=binding.producer_node_id,
            availability="maybe",
            origin=binding.origin,
        )
    return next_scope


def _value_matches_schema(schema: FieldSchema, value: Any) -> bool:
    if schema.type_kind in {"unknown", "enum"}:
        return True
    if schema.type_kind == "string":
        return isinstance(value, str)
    if schema.type_kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema.type_kind == "number":
        return _is_number(value)
    if schema.type_kind == "boolean":
        return isinstance(value, bool)
    if schema.type_kind == "array":
        return isinstance(value, list)
    if schema.type_kind == "object":
        return isinstance(value, dict)
    if schema.type_kind == "null":
        return value is None
    return True


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _format_available(values: Mapping[str, Any]) -> str:
    if not values:
        return "<none>"
    return ", ".join(sorted(values))
