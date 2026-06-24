"""Workspace-filesystem skill store (per-user, OBO).

Reads governed-skill Markdown from the calling user's workspace folders
(``.skills``, ``.assistant/skills``) plus any user-configured extra roots
(workspace paths or UC Volume paths), parsed by the framework parser.

Design:

* **Per-user / OBO** — constructed with the user's on-behalf-of
  ``WorkspaceClient`` so a user only ever sees skills under *their* paths
  (never the service principal's). Privacy by identity.
* **Read-only** — ``put_skill`` raises; authoring goes to the governed Lakebase
  store. This store only powers runtime reads (``read_skill`` + the prompt
  skills-section + discovery).
* **Listing is cheap + unscanned** (metadata only); **fetching a body is
  fail-closed scanned** (when a scanner is provided) so an unsafe user-authored
  skill cannot have its body injected via ``read_skill``. Verdicts are cached by
  content hash to avoid re-scanning unchanged skills.
* **Fail-soft I/O** — a missing root, an unreadable file, or one malformed
  ``.md`` is logged and skipped; the remaining skills still resolve. A skill
  source being momentarily unavailable must never break a research run.

Both the flat layout (``<root>/<name>.md``) and the nested layout
(``<root>/<name>/SKILL.md``) are discovered (one directory level deep).
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
from typing import Any

from databricks_deep_research.skills import (
    Skill,
    SkillMeta,
    SkillParseError,
    SkillSecurityScanner,
    SkillStoreError,
    parse_skill,
)

logger = logging.getLogger(__name__)

__all__ = ["WorkspaceFsSkillStore", "default_skill_roots"]

_MD_SUFFIX = ".md"
_VOLUME_PREFIX = "/Volumes/"


def default_skill_roots(user_name: str) -> list[str]:
    """The two conventional per-user skill folders under the workspace home."""
    base = f"/Workspace/Users/{user_name}"
    return [f"{base}/.skills", f"{base}/.assistant/skills"]


class WorkspaceFsSkillStore:
    """A read-only :class:`SkillStore` over the user's workspace-FS skill folders."""

    def __init__(
        self,
        ws_client: Any,
        *,
        user_name: str,
        extra_roots: list[str] | None = None,
        scanner: SkillSecurityScanner | None = None,
        ttl_seconds: int = 300,
    ) -> None:
        self._ws = ws_client
        self._roots = [*default_skill_roots(user_name), *(extra_roots or [])]
        self._scanner = scanner
        self._ttl = ttl_seconds
        self._cache: dict[str, Skill] | None = None
        self._cache_at: float = 0.0
        self._scan_verdicts: dict[str, bool] = {}  # content-hash -> safe

    # -- SkillStore ----------------------------------------------------------

    async def list_skills(self) -> list[SkillMeta]:
        """Metadata for all discoverable skills (cheap; bodies not scanned)."""
        skills = self._load()
        return [skills[name].meta for name in sorted(skills)]

    async def get_skill(self, name: str) -> Skill | None:
        """Return the skill body, fail-closed scanned when a scanner is set."""
        skill = self._load().get(name)
        if skill is None:
            return None
        if self._scanner is not None and not await self._is_safe(skill):
            logger.warning(
                "WORKSPACE_SKILL_BLOCKED name=%s reason=scan_unsafe", name
            )
            return None
        return skill

    async def put_skill(
        self, skill: Skill, *, scan: SkillSecurityScanner
    ) -> None:
        """Always raises — workspace skills are read-only at runtime."""
        del skill, scan  # part of the SkillStore protocol; read-only store
        raise SkillStoreError(
            "WorkspaceFsSkillStore is read-only; author skills in your workspace "
            "folder directly or use the governed LakebaseSkillStore."
        )

    # -- internals -----------------------------------------------------------

    async def _is_safe(self, skill: Skill) -> bool:
        digest = hashlib.sha256(
            (skill.body + json.dumps(skill.scripts, sort_keys=True)).encode("utf-8")
        ).hexdigest()
        cached = self._scan_verdicts.get(digest)
        if cached is not None:
            return cached
        try:
            result = await self._scanner.scan(skill)  # type: ignore[union-attr]
            safe = bool(result.safe)
        except Exception:  # noqa: BLE001 — fail-closed: any scan error is unsafe
            logger.exception("WORKSPACE_SKILL_SCAN_ERROR name=%s", skill.name)
            safe = False
        self._scan_verdicts[digest] = safe
        return safe

    def _load(self) -> dict[str, Skill]:
        if self._cache is not None and (time.time() - self._cache_at) < self._ttl:
            return self._cache
        skills: dict[str, Skill] = {}
        for root in self._roots:
            for path in self._iter_markdown(root):
                text = self._read_text(path)
                if not text:
                    continue
                try:
                    skill = parse_skill(text)
                except SkillParseError:
                    logger.warning("WORKSPACE_SKILL_PARSE_FAILED path=%s", path)
                    continue
                # First root wins on a name collision (deterministic precedence).
                skills.setdefault(skill.name, skill)
        self._cache = skills
        self._cache_at = time.time()
        logger.info(
            "WORKSPACE_SKILLS_LOADED roots=%d skills=%d", len(self._roots), len(skills)
        )
        return skills

    def _iter_markdown(self, root: str) -> list[str]:
        """Markdown paths at *root* and one directory level deep (fail-soft)."""
        out: list[str] = []
        for entry in self._list_dir(root):
            path, is_dir = self._entry(entry)
            if not path:
                continue
            if is_dir:
                for sub in self._list_dir(path):
                    sub_path, sub_is_dir = self._entry(sub)
                    if sub_path and not sub_is_dir and sub_path.endswith(_MD_SUFFIX):
                        out.append(sub_path)
            elif path.endswith(_MD_SUFFIX):
                out.append(path)
        return out

    @staticmethod
    def _is_volume(path: str) -> bool:
        return path.startswith(_VOLUME_PREFIX)

    def _list_dir(self, path: str) -> list[Any]:
        try:
            if self._is_volume(path):
                resp = self._ws.files.list_directory_contents(path)
                return list(getattr(resp, "contents", None) or resp or [])
            return list(self._ws.workspace.list(path) or [])
        except Exception:  # noqa: BLE001 — a missing/forbidden folder is normal
            logger.info("WORKSPACE_SKILL_DIR_UNAVAILABLE path=%s", path)
            return []

    @staticmethod
    def _entry(entry: Any) -> tuple[str | None, bool]:
        """Normalise a workspace ObjectInfo or volume DirectoryEntry to (path, is_dir)."""
        path = getattr(entry, "path", None) or (
            entry.get("path") if isinstance(entry, dict) else None
        )
        object_type = getattr(entry, "object_type", None)
        if object_type is not None:
            is_dir = str(getattr(object_type, "value", object_type)).upper() == "DIRECTORY"
        else:
            is_dir = bool(
                getattr(entry, "is_directory", False)
                or (entry.get("is_directory") if isinstance(entry, dict) else False)
            )
        return path, is_dir

    def _read_text(self, path: str) -> str:
        try:
            if self._is_volume(path):
                resp = self._ws.files.download(path)
                data = getattr(resp, "contents", resp)
                raw = data.read() if hasattr(data, "read") else data
                return self._decode(raw)
            download = getattr(self._ws.workspace, "download", None)
            if download is not None:
                resp = download(path)
                raw = resp.read() if hasattr(resp, "read") else resp
                if isinstance(raw, (bytes, bytearray)):
                    return raw.decode("utf-8")
                if isinstance(raw, str):
                    return raw
            # Version-robust fallback: export returns base64 ``content``.
            from databricks.sdk.service.workspace import ExportFormat

            exported = self._ws.workspace.export(path, format=ExportFormat.AUTO)
            content = getattr(exported, "content", None)
            if content:
                return base64.b64decode(content).decode("utf-8")
        except Exception:  # noqa: BLE001 — one unreadable file must not break load
            logger.warning("WORKSPACE_SKILL_READ_FAILED path=%s", path)
        return ""

    @staticmethod
    def _decode(raw: Any) -> str:
        if isinstance(raw, (bytes, bytearray)):
            return raw.decode("utf-8")
        return str(raw) if raw is not None else ""
