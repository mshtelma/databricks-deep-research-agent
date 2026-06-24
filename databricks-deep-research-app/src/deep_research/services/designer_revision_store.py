"""Fail-closed persist gate + append-only audit log for Designer changes.

Spec §5.6 / §4.4 — "Designer self-evolution sub-patterns (patch + audit)".

This module composes the two governance halves of the feature:

* the **shared fail-closed LLM security scan** (US-110 / 2.2) — reused, not
  re-implemented — gates a Designer-authored prompt/skill BEFORE persist; and
* the **append-only revision/audit log** — one immutable :class:`DesignerRevision`
  row per authored change, recording prev/new + the scan verdict.

The single entry point :meth:`DesignerRevisionStore.record_authored_change`
runs the scan and is **fail-closed**: it denies on a scan EXCEPTION and on any
non-safe verdict; only an explicit SAFE verdict writes the audit row and
returns it. It composes WITH the existing edit-lane (it does not re-apply or
re-route edits) — callers invoke it at the designer save/deploy boundary with
the authored content the edit lane produced.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from databricks_deep_research.skills import Skill, SkillSecurityScanner
from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.models.designer_revision import DesignerRevision

logger = logging.getLogger(__name__)

__all__ = [
    "DesignerRevisionError",
    "DesignerRevisionStore",
    "authored_text_as_skill",
]


class DesignerRevisionError(RuntimeError):
    """Raised when the fail-closed scan blocks a Designer-authored change.

    Mirrors :class:`SkillStoreError`'s contract for the skill path: the message
    distinguishes a scanner EXCEPTION ("scan failed") from a non-safe VERDICT
    ("rejected") so callers can surface the right reason to the user.
    """


def authored_text_as_skill(
    *,
    name: str,
    text: str,
    description: str = "",
) -> Skill:
    """Wrap an authored prompt/text as a scannable :class:`Skill`.

    The shared :class:`SkillSecurityScanner` inspects a ``Skill`` (name +
    description + body + scripts) for prompt-injection / instruction-override /
    exfiltration. A Designer-authored prompt is exactly such a body, so we wrap
    it as a minimal Skill and reuse the SAME deny-by-default scan code path —
    no new security logic. ``body`` must be non-empty (the Skill model enforces
    ``min_length=1``); an empty authored prompt is coerced to a single space so
    the scanner still runs rather than raising a validation error before the
    gate.
    """
    return Skill(
        name=name or "designer-authored",
        description=description or name or "designer-authored change",
        body=text if text else " ",
    )


class DesignerRevisionStore:
    """Writes the append-only Designer audit log behind a fail-closed scan.

    The caller owns the transaction/commit. Each successful authored change is
    a NEW row — the table is never updated in place.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record_authored_change(
        self,
        *,
        subject_type: str,
        subject_ref: str,
        scanned: Skill,
        new_snapshot: dict[str, Any],
        created_by: str,
        scan: SkillSecurityScanner,
        prev_snapshot: dict[str, Any] | None = None,
    ) -> DesignerRevision:
        """Scan *scanned*; on a SAFE verdict, append one audit row and return it.

        Fail-closed: a scan EXCEPTION or any non-safe verdict raises
        :class:`DesignerRevisionError` and writes NOTHING — only an explicit
        SAFE verdict (``result.safe is True``) persists. The recorded
        ``security_verdict`` is the scanner's reason, or the literal token
        ``"safe"`` when the scanner gave no reason.

        Args:
            subject_type: Kind of subject (e.g. ``"prompt"``, ``"agent"``,
                ``"skill"``) — pairs with *subject_ref* to identify the target.
            subject_ref: Stable id of the subject (node id, agent id, skill
                name, ...).
            scanned: The :class:`Skill`-shaped payload the scanner judges. Use
                :func:`authored_text_as_skill` to wrap a raw prompt.
            new_snapshot: JSON snapshot of the authored content AFTER the change.
            created_by: Actor (user id / principal) attributed with the change.
            scan: The shared fail-closed :class:`SkillSecurityScanner`.
            prev_snapshot: JSON snapshot BEFORE the change (``None`` on first).

        Raises:
            DesignerRevisionError: If the scan raises or returns a non-safe
                verdict — the audit row is NOT written.
        """
        try:
            result = await scan.scan(scanned)
        except Exception as exc:  # noqa: BLE001 - fail-closed on scanner error
            logger.warning(
                "DESIGNER_REVISION_SCAN_ERROR subject=%s/%s err=%s",
                subject_type,
                subject_ref,
                exc,
            )
            raise DesignerRevisionError(
                f"security scan failed for {subject_type} {subject_ref!r}: {exc}"
            ) from exc

        if not result.safe:
            logger.warning(
                "DESIGNER_REVISION_SCAN_REJECTED subject=%s/%s reason=%s",
                subject_type,
                subject_ref,
                result.reason,
            )
            raise DesignerRevisionError(
                f"security scan rejected {subject_type} {subject_ref!r}: "
                f"{result.reason}"
            )

        verdict = result.reason or "safe"
        row = DesignerRevision(
            rev_id=uuid4(),
            subject_type=subject_type,
            subject_ref=subject_ref,
            prev_snapshot=prev_snapshot,
            new_snapshot=new_snapshot,
            security_verdict=verdict,
            created_at=datetime.now(UTC),
            created_by=created_by,
        )
        self._session.add(row)
        await self._session.flush()
        return row
