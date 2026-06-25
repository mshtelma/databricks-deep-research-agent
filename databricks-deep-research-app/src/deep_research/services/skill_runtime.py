"""Compose the per-request runtime :class:`SkillStore` from request context.

The runtime needs one :class:`SkillStore` spanning every configured source, in
precedence order:

1. **Workspace-FS** (the calling user's ``.skills`` / ``.assistant/skills`` +
   any configured extra roots) — per-user, OBO, fail-closed body scan.
2. **Lakebase** governed authored skills — included only when a DB session is
   available at the call site.
3. **Bundled seeds** (always available, framework-shipped).

This factory keeps the composition decision in ONE place so callers
(framework orchestrator, agent serving) just pass their request context. It
never raises: a source that cannot be built is logged and omitted.
"""

from __future__ import annotations

import logging
from typing import Any

from databricks_deep_research import FrameworkLLMClient
from databricks_deep_research.core.databricks_auth import resolve_workspace_client
from databricks_deep_research.skills import FilesystemSkillStore, SkillMeta, SkillStore

from deep_research.services.composite_skill_store import CompositeSkillStore
from deep_research.services.skill_store import (
    LakebaseSkillStore,
    LLMSkillSecurityScanner,
)
from deep_research.services.workspace_fs_skill_store import WorkspaceFsSkillStore

logger = logging.getLogger(__name__)

__all__ = ["build_runtime_skill_store", "list_runtime_skills"]


def _resolve_user_name(ws_client: Any | None) -> str:
    if ws_client is None:
        return ""
    try:
        return getattr(ws_client.current_user.me(), "user_name", "") or ""
    except Exception:  # noqa: BLE001 — identity lookup failure must not break the run
        logger.warning("SKILL_RUNTIME_USERNAME_UNRESOLVED", exc_info=True)
        return ""


def build_runtime_skill_store(
    *,
    llm_client: FrameworkLLMClient | None,
    workspace_client: Any | None,
    user_token: str | None,
    session: Any | None = None,
    extra_roots: list[str] | None = None,
    user_name: str | None = None,
) -> SkillStore:
    """Compose the runtime skill store from the request's identity + resources.

    Args:
        llm_client: framework LLM client (backs the workspace-FS body scanner).
        workspace_client: the request's service-principal / default client.
        user_token: OBO token; when present the workspace-FS source reads as the
            calling user (never the SP).
        session: optional DB session — includes the governed Lakebase source.
        extra_roots: optional additional skill-folder roots (workspace / Volume).
        user_name: optional pre-resolved user name (else derived from the client).

    Returns:
        A :class:`CompositeSkillStore` spanning the available sources (always at
        least the bundled seeds).
    """
    stores: list[SkillStore] = []

    obo = resolve_workspace_client(sp_client=workspace_client, user_token=user_token)
    name = user_name or _resolve_user_name(obo)
    if obo is not None and name:
        scanner = (
            LLMSkillSecurityScanner(llm_client) if llm_client is not None else None
        )
        stores.append(
            WorkspaceFsSkillStore(
                obo, user_name=name, extra_roots=extra_roots, scanner=scanner
            )
        )
    else:
        logger.info(
            "SKILL_RUNTIME_NO_WORKSPACE_SOURCE has_client=%s has_name=%s",
            obo is not None,
            bool(name),
        )

    if session is not None:
        stores.append(LakebaseSkillStore(session))

    stores.append(FilesystemSkillStore())  # bundled seeds — always available
    return CompositeSkillStore(stores)


async def list_runtime_skills(
    *,
    workspace_client: Any | None,
    user_token: str | None,
    session: Any | None = None,
    extra_roots: list[str] | None = None,
    user_name: str | None = None,
) -> list[SkillMeta]:
    """List the skills available to the caller (metadata only; for discovery UIs).

    Builds the same composite store as :func:`build_runtime_skill_store` (without
    a scanner — listing never reads bodies, so no scan runs) and returns its
    deduped metadata. Fail-soft: returns ``[]`` on any error so a discovery
    surface never hard-fails on skills.
    """
    try:
        store = build_runtime_skill_store(
            llm_client=None,
            workspace_client=workspace_client,
            user_token=user_token,
            session=session,
            extra_roots=extra_roots,
            user_name=user_name,
        )
        return await store.list_skills()
    except Exception:  # noqa: BLE001 — discovery is best-effort
        logger.warning("SKILL_DISCOVERY_FAILED", exc_info=True)
        return []
