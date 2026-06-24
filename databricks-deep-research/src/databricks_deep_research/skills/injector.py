"""Prompt-section injector for attached skills.

Renders a compact prompt section listing the **metadata only** (name +
description) of the skills attached to an agent.  Bodies are deliberately
excluded — the agent pulls a body on demand via the ``read_skill`` tool
(progressive disclosure), so attaching many skills costs only a few lines of
context rather than their full text.

This module is intentionally standalone and side-effect-free so it can be
unit-tested in isolation and wired into the agent loop later.
"""

from __future__ import annotations

from collections.abc import Iterable

from databricks_deep_research.skills.models import Skill, SkillMeta

__all__ = ["render_skills_section"]

_HEADER = "## Available Skills"
_PREAMBLE = (
    "The following skills provide reusable methodology. Each entry lists a "
    "name and a one-line description ONLY. To use a skill, call the "
    "`read_skill` tool with its exact name to load the full instructions "
    "before applying them. Do not assume a skill's contents from its "
    "description."
)


def _to_meta(item: SkillMeta | Skill) -> SkillMeta:
    """Normalise a :class:`Skill` or :class:`SkillMeta` to :class:`SkillMeta`."""
    if isinstance(item, Skill):
        return item.meta
    return item


def render_skills_section(skills: Iterable[SkillMeta | Skill]) -> str:
    """Render the metadata-only skills prompt section.

    Accepts either :class:`SkillMeta` or full :class:`Skill` objects; in both
    cases only ``name`` and ``description`` are emitted — never the body or
    scripts.

    Returns an empty string when *skills* is empty, so callers can append the
    result unconditionally.
    """
    metas = [_to_meta(item) for item in skills]
    if not metas:
        return ""

    lines = [_HEADER, "", _PREAMBLE, ""]
    for meta in metas:
        lines.append(f"- **{meta.name}**: {meta.description}")
    return "\n".join(lines)
