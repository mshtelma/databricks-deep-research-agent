"""``run_skill_script`` tool — execute a skill's named script in the sandbox.

A skill may bundle named scripts (``Skill.scripts``) — small, deterministic
Python helpers that are NEVER read into the LLM context. When script execution is
enabled (a global AND a per-agent switch, both default-off), an agent may invoke
one by ``skill`` + ``script`` name with JSON ``arguments``; the code runs in the
hardened :class:`ProcessSandbox` (see :mod:`skill_script_executor`).

Security posture:

* The tool is **auto-attached only** (mirroring ``read_skill``) when its switches
  are on — it is deliberately NOT a freely-declarable tool kind, so an agent
  cannot list it in ``config.tools`` to bypass the per-agent ``allow_skill_scripts``
  gate. When ``enabled`` is False the tool refuses every call with a clear error.
* Before running, the optional :class:`SkillSecurityScanner` judges the whole
  skill (fail-closed: a scan error or non-safe verdict refuses the run). The
  sandbox's AST policy + OS isolation are the hard boundary regardless.
"""

from __future__ import annotations

import logging
from typing import Any

from databricks_deep_research.skills.store import SkillSecurityScanner, SkillStore
from databricks_deep_research.tools.builtins.skill_script_executor import ProcessSandbox
from databricks_deep_research.tools.protocol import (
    SourceKind,
    ToolContext,
    ToolDefinition,
    ToolResult,
)

logger = logging.getLogger(__name__)

__all__ = ["RunSkillScriptTool"]

_DEFAULT_DESCRIPTION = (
    "Execute a named script bundled with an attached skill. Scripts are small "
    "deterministic Python helpers run in a hardened sandbox (no network, no "
    "filesystem, no secrets). Provide the skill name, the script name, and a JSON "
    "object of arguments; the script may read those arguments and assign a "
    "'result' variable to return a value. Returns the result plus any printed "
    "output."
)


class RunSkillScriptTool:
    """Executes a skill's named script in the :class:`ProcessSandbox`."""

    def __init__(
        self,
        skill_store: SkillStore,
        *,
        sandbox: ProcessSandbox | None = None,
        scanner: SkillSecurityScanner | None = None,
        enabled: bool = True,
        name: str = "run_skill_script",
        description: str = "",
    ) -> None:
        self._store = skill_store
        self._sandbox = sandbox or ProcessSandbox()
        self._scanner = scanner
        self._enabled = enabled
        self._name = name
        self._description = description or _DEFAULT_DESCRIPTION

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=self._description,
            parameters={
                "type": "object",
                "properties": {
                    "skill": {
                        "type": "string",
                        "description": "Exact name of the attached skill.",
                    },
                    "script": {
                        "type": "string",
                        "description": "Name of the script within that skill to run.",
                    },
                    "arguments": {
                        "type": "object",
                        "description": "JSON object passed to the script as named globals.",
                    },
                },
                "required": ["skill", "script"],
                "additionalProperties": False,
            },
            source_type="builtin",
            source_kind=SourceKind.builtin,
            metadata={"budget_free": False},
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        skill = arguments.get("skill")
        script = arguments.get("script")
        if not isinstance(skill, str) or not skill.strip():
            raise ValueError("run_skill_script requires a non-empty 'skill' name")
        if not isinstance(script, str) or not script.strip():
            raise ValueError("run_skill_script requires a non-empty 'script' name")
        raw_args = arguments.get("arguments", {})
        if raw_args is None:
            raw_args = {}
        if not isinstance(raw_args, dict):
            raise ValueError("'arguments' must be a JSON object")
        return {
            "skill": skill.strip(),
            "script": script.strip(),
            "arguments": raw_args,
        }

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        del context  # required by the ResearchTool protocol; unused here
        if not self._enabled:
            return ToolResult(
                content=(
                    "Skill script execution is disabled. Ask an administrator to "
                    "enable it (skills.allow_script_execution) and set "
                    "allow_skill_scripts on this agent."
                ),
                success=False,
                error="skill_scripts_disabled",
                data={"source_kind": SourceKind.builtin},
            )

        skill_name = arguments["skill"]
        script_name = arguments["script"]
        script_args = arguments["arguments"]

        skill = await self._store.get_skill(skill_name)
        if skill is None:
            available = ", ".join(meta.name for meta in await self._store.list_skills())
            hint = f" Available skills: {available}." if available else ""
            return ToolResult(
                content=f"No skill named {skill_name!r} was found.{hint}",
                success=False,
                error="skill_not_found",
                data={"source_kind": SourceKind.builtin},
            )

        code = skill.scripts.get(script_name)
        if code is None:
            available = ", ".join(sorted(skill.scripts)) or "(none)"
            return ToolResult(
                content=(
                    f"Skill {skill_name!r} has no script named {script_name!r}. "
                    f"Available scripts: {available}."
                ),
                success=False,
                error="script_not_found",
                data={"source_kind": SourceKind.builtin},
            )

        # Fail-closed security scan (defense in depth; the sandbox is the hard
        # boundary). A scanner error or a non-safe verdict refuses the run.
        if self._scanner is not None:
            try:
                verdict = await self._scanner.scan(skill)
                safe = bool(verdict.safe)
            except Exception:  # noqa: BLE001 — scan failure must fail closed
                logger.exception("SKILL_SCRIPT_SCAN_FAILED skill=%s", skill_name)
                safe = False
            if not safe:
                return ToolResult(
                    content=(
                        f"Skill {skill_name!r} did not pass the security scan; its "
                        "scripts cannot be run."
                    ),
                    success=False,
                    error="skill_unsafe",
                    data={"source_kind": SourceKind.builtin},
                )

        outcome = await self._sandbox.run(code, script_args)
        logger.info(
            "SKILL_SCRIPT_RAN skill=%s script=%s ok=%s dur=%.2fs",
            skill_name,
            script_name,
            outcome.ok,
            outcome.duration_seconds,
        )
        if not outcome.ok:
            detail = outcome.error or "unknown error"
            body = f"Skill script {skill_name}/{script_name} failed: {detail}"
            if outcome.stdout:
                body += f"\n\nOutput before failure:\n{outcome.stdout}"
            return ToolResult(
                content=body,
                success=False,
                error=outcome.error_type or "skill_script_error",
                data={"source_kind": SourceKind.builtin},
            )

        parts: list[str] = []
        if outcome.result is not None:
            parts.append(f"Result: {outcome.result!r}")
        if outcome.stdout:
            parts.append(f"Output:\n{outcome.stdout}")
        if outcome.note:
            parts.append(f"Note: {outcome.note}")
        content = "\n\n".join(parts) if parts else "Script completed with no output."
        return ToolResult(
            content=content,
            success=True,
            data={
                "source_kind": SourceKind.builtin,
                "skill_name": skill_name,
                "script_name": script_name,
            },
        )
