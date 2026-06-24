"""Pydantic models for the skills subsystem.

A *skill* is a small, governed unit of know-how authored as Markdown with YAML
frontmatter.  The frontmatter carries identity (``name``, ``description``) and
optional ``scripts`` (named code snippets executed in the compute scratchpad —
never read into the LLM context).  The Markdown body is the full methodology
that an agent pulls on demand via the ``read_skill`` tool (progressive
disclosure).

Two models split metadata from the full record so the dominant access pattern
("list metadata cheaply, fetch body on demand") maps onto two storage reads:

* :class:`SkillMeta` — name + description only; what the prompt injector lists.
* :class:`Skill` — the full record (metadata + body + scripts + governance).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["Skill", "SkillMeta"]


class SkillMeta(BaseModel):
    """Lightweight skill metadata — the in-prompt listing unit.

    Only ``name`` and ``description`` are ever rendered into a prompt.  The
    body is fetched separately (via ``read_skill``) so attaching many skills
    costs only a few metadata lines, not their full bodies.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(..., min_length=1, max_length=128)
    description: str = Field(..., min_length=1, max_length=1024)


class Skill(BaseModel):
    """A full skill record: metadata + Markdown body + optional scripts.

    ``scripts`` maps a script name to its source code.  Scripts are executed in
    the compute scratchpad and are NEVER injected into the LLM context — the
    model only ever sees ``name``, ``description``, and (on demand) ``body``.

    Governance fields (``version``, ``author``, ``security_verdict``,
    ``created_at``, ``updated_at``) are optional in the framework model so the
    bundled read-only :class:`FilesystemSkillStore` can construct skills from
    seed files alone; the app's Lakebase-backed store populates them.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=128)
    description: str = Field(..., min_length=1, max_length=1024)
    body: str = Field(..., min_length=1)
    scripts: dict[str, str] = Field(default_factory=dict)

    # -- governance (optional; populated by persistent backends) -------------
    version: int = 1
    author: str | None = None
    security_verdict: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def meta(self) -> SkillMeta:
        """Return the lightweight :class:`SkillMeta` view of this skill."""
        return SkillMeta(name=self.name, description=self.description)
