"""Auto-attach the ``read_skill`` tool to agents that declare skills.

When an agent's ``config.skills`` is non-empty, the harness injects a
metadata-only skills section into the prompt and instructs the model to call
``read_skill`` to load full bodies on demand. For that to work the agent must
actually have a ``read_skill`` tool. Rather than require every author to also
list ``read_skill`` in ``config.tools``, the executor calls
:func:`maybe_attach_read_skill` after resolving declared tools: it appends a
``read_skill`` built directly from the wired store.

This is safe because the ReAct loop builds its name→tool map from the same
resolved ``tools`` list, so an appended tool is immediately callable. It is a
no-op (and never raises) when there are no skills, a ``read_skill`` is already
present, or no ``_skill_store`` is wired in the factory context.
"""

from __future__ import annotations

import logging
from typing import Any

from databricks_deep_research.tools.protocol import ResearchTool

logger = logging.getLogger(__name__)

__all__ = ["maybe_attach_read_skill", "maybe_attach_run_skill_script"]


def maybe_attach_read_skill(
    tools: list[ResearchTool],
    skills: list[str],
    factory_context: Any | None,
) -> bool:
    """Append a ``read_skill`` tool when *skills* are declared and a store is wired.

    Args:
        tools: the agent's resolved tool list (mutated in place when attaching).
        skills: ``config.skills`` — the attached skill names.
        factory_context: the resolver's :class:`ToolFactoryContext` (its
            ``extras["_skill_store"]`` supplies the store).

    Returns:
        ``True`` if a tool was appended, else ``False``.
    """
    if not skills:
        return False

    from databricks_deep_research.tools.builtins.read_skill import ReadSkillTool

    if any(isinstance(tool, ReadSkillTool) for tool in tools):
        return False

    extras = getattr(factory_context, "extras", None) or {}
    store = extras.get("_skill_store")
    if store is None:
        logger.warning(
            "READ_SKILL_AUTOATTACH_NO_STORE skills=%d — read_skill not attached "
            "(no _skill_store wired in the factory context)",
            len(skills),
        )
        return False

    tools.append(ReadSkillTool(skill_store=store))
    logger.info("READ_SKILL_AUTOATTACHED skills=%d", len(skills))
    return True


def maybe_attach_run_skill_script(
    tools: list[ResearchTool],
    skills: list[str],
    allow_skill_scripts: bool,
    factory_context: Any | None,
) -> bool:
    """Append a ``run_skill_script`` tool when script execution is enabled.

    Skill-script execution is gated by TWO switches that must BOTH be on: the
    per-agent ``allow_skill_scripts`` (passed here) and the global
    ``skills.allow_script_execution`` (surfaced by the host as
    ``factory_context.extras["_skill_scripts_enabled"]``). The tool is attached
    only when skills are declared, both switches are on, a store is wired, and no
    ``run_skill_script`` is already present. Attaching it ONLY when fully enabled
    means an agent can never reach it by listing it in ``config.tools`` (it is not
    a declarable tool kind) — closing the gate-bypass path.

    Returns ``True`` if a tool was appended, else ``False`` (and never raises).
    """
    if not skills or not allow_skill_scripts:
        return False

    extras = getattr(factory_context, "extras", None) or {}
    if not extras.get("_skill_scripts_enabled"):
        # Per-agent flag is on but the global kill-switch is off (or unset).
        return False

    from databricks_deep_research.tools.builtins.run_skill_script import (
        RunSkillScriptTool,
    )

    if any(isinstance(tool, RunSkillScriptTool) for tool in tools):
        return False

    store = extras.get("_skill_store")
    if store is None:
        logger.warning(
            "RUN_SKILL_SCRIPT_AUTOATTACH_NO_STORE skills=%d — run_skill_script not "
            "attached (no _skill_store wired in the factory context)",
            len(skills),
        )
        return False

    scanner = extras.get("_skill_scanner")
    tools.append(RunSkillScriptTool(skill_store=store, scanner=scanner, enabled=True))
    logger.info("RUN_SKILL_SCRIPT_AUTOATTACHED skills=%d", len(skills))
    return True
