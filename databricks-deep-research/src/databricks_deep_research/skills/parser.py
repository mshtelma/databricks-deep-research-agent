"""Frontmatter parser for skill Markdown files.

Skill files are Markdown with a leading YAML frontmatter block delimited by
``---`` fences::

    ---
    name: deep-research
    description: A systematic multi-angle web-research methodology.
    scripts:
      summarize: |
        print("hi")
    ---

    # Body markdown ...

Only three frontmatter keys are recognised: ``name`` (required),
``description`` (required), and ``scripts`` (optional ``dict[str, str]``).

**Security — angle-bracket guard.** ``description`` is rendered verbatim into
agent prompts (the injector lists it).  To prevent prompt-injection via
pseudo-tags (e.g. ``</system>`` or ``<important>ignore prior</important>``),
the parser **rejects** any ``description`` containing ``<`` or ``>``.  This is
fail-closed: a malformed/injected description raises rather than being
sanitised, so the caller never silently ships a tampered skill.
"""

from __future__ import annotations

import re
from typing import Any

import yaml

from databricks_deep_research.skills.models import Skill

__all__ = ["SkillParseError", "parse_skill", "split_frontmatter"]


class SkillParseError(ValueError):
    """Raised when a skill file cannot be parsed or fails validation."""


# A leading ``---`` line, the YAML block, a closing ``---`` line, then the body.
# DOTALL so the YAML/body groups span newlines; the body group is everything
# after the closing fence.
_FRONTMATTER_RE = re.compile(
    r"\A﻿?---[ \t]*\r?\n(?P<yaml>.*?)\r?\n---[ \t]*\r?\n?(?P<body>.*)\Z",
    re.DOTALL,
)

# Reject angle brackets in description (prompt-injection guard).
_ANGLE_BRACKET_RE = re.compile(r"[<>]")


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split *text* into ``(frontmatter_dict, body)``.

    Raises:
        SkillParseError: If no frontmatter fence is present, the YAML is
            invalid, or the YAML does not parse to a mapping.
    """
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        raise SkillParseError(
            "skill file missing YAML frontmatter; expected a leading '---' fence"
        )

    try:
        loaded = yaml.safe_load(match.group("yaml"))
    except yaml.YAMLError as exc:  # pragma: no cover - exercised via parse_skill
        raise SkillParseError(f"invalid YAML frontmatter: {exc}") from exc

    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise SkillParseError(
            f"frontmatter must be a mapping, got {type(loaded).__name__}"
        )

    return loaded, match.group("body").strip()


def _coerce_scripts(raw: Any) -> dict[str, str]:
    """Validate and coerce the optional ``scripts`` mapping.

    Returns an empty dict when *raw* is falsy/absent.  Raises if it is present
    but not a ``dict[str, str]``.
    """
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise SkillParseError(
            f"'scripts' must be a mapping of name->code, got {type(raw).__name__}"
        )
    scripts: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.strip():
            raise SkillParseError("'scripts' keys must be non-empty strings")
        if not isinstance(value, str):
            raise SkillParseError(
                f"'scripts.{key}' must be a string, got {type(value).__name__}"
            )
        scripts[key] = value
    return scripts


def parse_skill(text: str) -> Skill:
    """Parse a skill Markdown document into a validated :class:`Skill`.

    Validates: required ``name``/``description``, the angle-bracket guard on
    ``description``, an optional ``scripts`` mapping, and a non-empty body.

    Raises:
        SkillParseError: On any structural or validation failure.
    """
    frontmatter, body = split_frontmatter(text)

    name = frontmatter.get("name")
    if not isinstance(name, str) or not name.strip():
        raise SkillParseError("skill frontmatter requires a non-empty 'name'")

    description = frontmatter.get("description")
    if not isinstance(description, str) or not description.strip():
        raise SkillParseError("skill frontmatter requires a non-empty 'description'")
    if _ANGLE_BRACKET_RE.search(description):
        raise SkillParseError(
            "skill 'description' must not contain angle brackets ('<' or '>'); "
            "rejected as a prompt-injection safeguard"
        )

    scripts = _coerce_scripts(frontmatter.get("scripts"))

    if not body.strip():
        raise SkillParseError("skill body is empty")

    try:
        return Skill(
            name=name.strip(),
            description=description.strip(),
            body=body,
            scripts=scripts,
        )
    except ValueError as exc:  # pydantic ValidationError is a ValueError subclass
        raise SkillParseError(f"skill failed validation: {exc}") from exc
