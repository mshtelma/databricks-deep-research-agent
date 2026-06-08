"""Contract tests for the three deployment-status sets (W2).

The model exposes three orthogonal sets:

- ``ACTIVE_STATUSES``: deployment is mid-lifecycle (pending/deploying) or live;
  the deletion guard counts these as "blocking parent agent delete".
- ``TERMINAL_STATUSES``: poll-stop semantics — the UI should stop refetching
  status for these rows. Includes FAILED so the UI doesn't poll forever.
- ``DELETABLE_STATUSES``: physical-row-delete semantics — force-deleting the
  parent agent may cascade-delete only these rows. EXCLUDES FAILED so the
  audit trail of failed deployments is preserved (must explicitly deactivate
  first).

These three sets are intentionally not equivalent. Tests pin the membership
so future edits cannot silently re-introduce the "failed polls forever"
regression codex flagged.
"""
from __future__ import annotations

from deep_research.models.agent_deployment import (
    ACTIVE_STATUSES,
    DELETABLE_STATUSES,
    TERMINAL_STATUSES,
    DeploymentStatus,
)


def test_active_set_membership() -> None:
    assert frozenset(
        {
            DeploymentStatus.PENDING.value,
            DeploymentStatus.DEPLOYING.value,
            DeploymentStatus.ACTIVE.value,
        }
    ) == ACTIVE_STATUSES


def test_terminal_set_includes_failed_for_poll_stop() -> None:
    # FAILED must be terminal for UI polling — otherwise failed deployments
    # poll forever (codex finding W2).
    assert DeploymentStatus.FAILED.value in TERMINAL_STATUSES
    assert DeploymentStatus.DEACTIVATED.value in TERMINAL_STATUSES
    assert DeploymentStatus.CLEANUP_FAILED.value in TERMINAL_STATUSES


def test_deletable_set_excludes_failed_for_forensics() -> None:
    # FAILED must NOT be in the deletable set — force-delete of the parent
    # agent should preserve failed rows as an audit trail. Users must
    # explicitly deactivate (which lands DEACTIVATED, deletable).
    assert DeploymentStatus.FAILED.value not in DELETABLE_STATUSES
    assert frozenset(
        {
            DeploymentStatus.DEACTIVATED.value,
            DeploymentStatus.CLEANUP_FAILED.value,
        }
    ) == DELETABLE_STATUSES


def test_sets_partition_correctly() -> None:
    # No status may be simultaneously active and terminal.
    assert ACTIVE_STATUSES.isdisjoint(TERMINAL_STATUSES)
    # Every deletable status is terminal (terminal is the broader poll-stop
    # set; deletable is the narrower physical-cleanup set).
    assert DELETABLE_STATUSES.issubset(TERMINAL_STATUSES)
    # Every defined status is accounted for in exactly one of {active,
    # terminal} (no orphans).
    all_statuses = {s.value for s in DeploymentStatus}
    assert all_statuses == ACTIVE_STATUSES | TERMINAL_STATUSES
