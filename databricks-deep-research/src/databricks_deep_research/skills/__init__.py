"""Skills subsystem — progressive-loaded capability packs.

A skill is governed Markdown (frontmatter + body + optional scripts) carrying
reusable methodology.  Agents see only skill METADATA in their prompt and pull
a full body on demand via the ``read_skill`` tool.

Public API::

    from databricks_deep_research.skills import (
        Skill,
        SkillMeta,
        SkillStore,
        SkillSecurityScanner,
        FilesystemSkillStore,
        parse_skill,
        render_skills_section,
    )

The framework provides backend-agnostic protocols + a read-only filesystem
store over bundled seeds; persistent backends (e.g. Lakebase) live in host
apps, keeping this package free of any database dependency.
"""

from __future__ import annotations

from databricks_deep_research.skills.injector import render_skills_section
from databricks_deep_research.skills.models import Skill, SkillMeta
from databricks_deep_research.skills.parser import (
    SkillParseError,
    parse_skill,
    split_frontmatter,
)
from databricks_deep_research.skills.store import (
    FilesystemSkillStore,
    SkillScanResult,
    SkillSecurityScanner,
    SkillStore,
    SkillStoreError,
)

__all__ = [
    "Skill",
    "SkillMeta",
    "SkillStore",
    "SkillSecurityScanner",
    "SkillScanResult",
    "SkillStoreError",
    "FilesystemSkillStore",
    "SkillParseError",
    "parse_skill",
    "split_frontmatter",
    "render_skills_section",
]
