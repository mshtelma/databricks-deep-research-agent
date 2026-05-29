"""Build-time dataflow reachability checks for workflow definitions.

Complements ``condition_contracts.py`` (which checks condition *type* correctness)
by checking dataflow *reachability*:

* **Pass A — dangling reads:** every effective read (prompt-template variable ∪
  declared ``input_keys`` ∪ tool ``input_mapping`` ∪ condition keys) must resolve
  to a producer visible in lexical scope. A dangling *control* read (a loop
  ``until`` / conditional branch / plan_and_execute evaluator source with no
  producer) is error-severity.
* **Pass B — dead stores:** a produced value consumed by nobody (across STATE,
  POOL, and the RUNTIME-RETURN control channel) is a warning, except terminal
  workflow outputs and pool round-trips.

This is a focused, existence-only walk over a plain ``set[str]`` of visible keys.
It deliberately mirrors ``condition_contracts.py``'s per-node scoping rules with
the same verified field access, but does NOT modify or share that validator's
traversal (Pass A needs only existence, so the schema/availability lattice is
irrelevant). It ships lint-first: diagnostics are warnings unless
``DATAFLOW_CHECK_STRICT`` is set, at which point error-severity diagnostics
become validation errors.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from databricks_deep_research.agents.config import AgentNodeConfig
from databricks_deep_research.templates.renderer import SafeTemplateRenderer
from databricks_deep_research.workflow.runtime_keys import RUNTIME_INJECTED_KEYS

_renderer = SafeTemplateRenderer()

# Must match SafeTemplateRenderer's {% for x in y %} syntax: group 1 = loop var.
_FOR_LOOP_VAR = re.compile(r"\{%\s*for\s+(\w+)\s+in\s+\w+\s*%\}")


@dataclass(frozen=True)
class Diagnostic:
    """A single dataflow diagnostic. ``severity`` is intrinsic; lint/strict mode
    only governs whether error-severity diagnostics block validation."""

    message: str
    severity: Literal["error", "warning"]


@dataclass
class DataflowReport:
    """Accumulated diagnostics. ``errors`` block validation (strict mode);
    ``warnings`` are logged only (lint mode)."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _template_reads(template: str) -> set[str]:
    """STATE keys a template reads, EXCLUDING loop-local variables.

    ``extract_variables`` returns BOTH the ``{% for x in items %}`` iterable AND
    the loop var ``x`` (because ``{x}`` in the body is matched as a plain
    ``{var}``). The loop var is a local binding, not a state read, so subtract it
    or it false-flags as dangling.
    """
    if not template:
        return set()
    return _renderer.extract_variables(template) - set(_FOR_LOOP_VAR.findall(template))


def effective_reads(cfg: AgentNodeConfig, *, exclude_runtime: bool = False) -> set[str]:
    """STATE keys an agent actually consumes: declared ``input_keys`` ∪ the
    variables referenced by its system/user prompt templates (the authoritative
    read signal — ``input_keys`` are documentation-only), minus loop-local
    variables. Optionally drop runtime-injected keys.
    """
    reads = set(cfg.input_keys)
    reads |= _template_reads(cfg.system_prompt or "")
    reads |= _template_reads(cfg.user_prompt_template or "")
    if exclude_runtime:
        reads -= RUNTIME_INJECTED_KEYS
    return reads
