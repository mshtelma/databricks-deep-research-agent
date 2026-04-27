"""Custom builtin subtype — pure pass-through with no hooks.

Registers the ``"custom"`` subtype with all hooks set to ``None`` so the
harness guards at ``harness.py:319`` (``enrich_config``), ``harness.py:494``
(``execute``), and ``harness.py:776`` (``post_process``) all skip — the node
inherits the default execution path through ``execute_agent()`` with zero
hook invocation.

This is the default subtype for the Python API ``Agent(...)`` class, used
when the user does not specify a builtin subtype. It is also referenced as
the ``"custom"`` subtype in YAML workflows for nodes that need the bare
agent harness without any builtin-specific enrichment, post-processing, or
execution overrides.
"""

from __future__ import annotations

from databricks_deep_research.agents.builtins.registry import register_builtin

register_builtin(
    "custom",
    post_process=None,
    enrich_config=None,
    execute=None,
    default_system_prompt="",
    default_user_prompt="{query}",
    output_model=None,
)

__all__: list[str] = []
