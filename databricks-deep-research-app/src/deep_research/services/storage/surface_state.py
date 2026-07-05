"""Pure helper for merging ``surface_state`` patches into stored metadata.

``surface_state`` lives at ``metadata["surface_state"]`` and has the shape::

    {
        "<agent_id>": {
            "data_model": {...},
            "action_runs": {
                "<action>": {
                    "session_id": str,
                    "message_id": str,
                    "status": "running"|"completed"|"failed"|"cancelled",
                    "updated_at": <iso-str>,
                }
            },
            "surface_etag": str | None,
        }
    }

Merge semantics (shared by both legacy ORM and cached impls):
* Per-agent entries are shallow-merged: incoming keys overwrite stored keys.
* ``action_runs`` is merged per-action with idempotence:
  - an incoming action entry REPLACES the stored one only if:
    - the stored entry is absent, OR
    - the incoming ``updated_at`` >= the stored ``updated_at``, OR
    - same ``session_id`` AND same ``status`` (idempotent replay).
* Multiple agents in one patch are independent — each follows the same rule.
"""

from __future__ import annotations

from typing import Any


def merge_surface_state(existing: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Return a new surface_state dict with *patch* merged into *existing*.

    Neither argument is mutated. Both must be ``dict[str, Any]`` keyed by
    agent-id strings.
    """
    result: dict[str, Any] = dict(existing)

    for agent_id, incoming_entry in patch.items():
        if not isinstance(incoming_entry, dict):
            continue

        existing_entry: dict[str, Any] = dict(result.get(agent_id) or {})

        # Shallow-merge all keys from incoming_entry, deferring action_runs.
        for k, v in incoming_entry.items():
            if k != "action_runs":
                existing_entry[k] = v

        # Merge action_runs with idempotence / newest-wins semantics.
        incoming_runs: dict[str, Any] = incoming_entry.get("action_runs") or {}
        if incoming_runs:
            merged_runs: dict[str, Any] = dict(existing_entry.get("action_runs") or {})
            for action, incoming_run in incoming_runs.items():
                if not isinstance(incoming_run, dict):
                    merged_runs[action] = incoming_run
                    continue

                stored_run: dict[str, Any] | None = merged_runs.get(action)
                if stored_run is None:
                    # No stored entry — always accept.
                    merged_runs[action] = incoming_run
                    continue

                # Same session_id AND same status → idempotent replay, accept.
                if (
                    incoming_run.get("session_id") == stored_run.get("session_id")
                    and incoming_run.get("status") == stored_run.get("status")
                ):
                    merged_runs[action] = incoming_run
                    continue

                # Newest-updated_at-wins.
                incoming_ts: str | None = incoming_run.get("updated_at")
                stored_ts: str | None = stored_run.get("updated_at")
                if incoming_ts is not None and stored_ts is not None:
                    if incoming_ts >= stored_ts:
                        merged_runs[action] = incoming_run
                elif incoming_ts is not None:
                    # Stored has no timestamp — incoming wins.
                    merged_runs[action] = incoming_run
                # else: stored has a timestamp but incoming doesn't — keep stored.

            existing_entry["action_runs"] = merged_runs

        result[agent_id] = existing_entry

    return result
