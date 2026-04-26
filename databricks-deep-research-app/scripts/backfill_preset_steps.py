"""Backfill script: migrate agent_preset_steps rows -> custom_agents.steps JSONB.

Idempotent: reads ``custom_agents.steps`` first; if a step UUID already exists
there, it is skipped. Run once after deploying the cached CustomAgentService.

Usage:
    STORAGE_BACKEND=lakebase uv run python scripts/backfill_preset_steps.py
    STORAGE_BACKEND=sql_warehouse uv run python scripts/backfill_preset_steps.py

The script reads from both the legacy ``agent_preset_steps`` ORM table and
writes the denormalised JSON array into ``custom_agents.steps`` using the
StorageStack cold-path upsert so the write lands correctly on both backends.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections import defaultdict
from datetime import UTC, datetime
from uuid import UUID

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def _run() -> None:
    from sqlalchemy import select

    from deep_research.core.config import get_settings
    from deep_research.db.session import get_session_maker
    from deep_research.models.custom_agent import AgentPresetStep, CustomAgent
    from deep_research.storage.factory import build_storage_stack

    settings = get_settings()
    session_maker = get_session_maker(settings)
    stack = await build_storage_stack(settings)
    await stack.start()

    try:
        async with session_maker() as db:
            # 1. Load all agents from legacy table
            agents_result = await db.execute(select(CustomAgent))
            agents = list(agents_result.scalars().all())
            logger.info("Found %d agents in custom_agents table", len(agents))

            # 2. Load all preset steps grouped by agent_id
            steps_result = await db.execute(
                select(AgentPresetStep).order_by(
                    AgentPresetStep.agent_id, AgentPresetStep.order
                )
            )
            all_steps = list(steps_result.scalars().all())
            logger.info("Found %d preset steps in agent_preset_steps table", len(all_steps))

            steps_by_agent: dict[UUID, list[AgentPresetStep]] = defaultdict(list)
            for step in all_steps:
                steps_by_agent[step.agent_id].append(step)

            # 3. For each agent, read existing steps JSON then merge
            upserted = 0
            skipped = 0
            for agent in agents:
                # Load current cached row (may already have some steps)
                existing_rows = await stack.backend.list_rows(
                    "custom_agents", {"id": str(agent.id)}
                )
                existing_step_ids: set[str] = set()
                existing_steps: list[dict] = []
                if existing_rows:
                    existing_steps = existing_rows[0].get("steps") or []
                    existing_step_ids = {s["id"] for s in existing_steps}

                # Build new steps list from legacy rows (skip already-present UUIDs)
                new_steps = list(existing_steps)
                added = 0
                for step in steps_by_agent.get(agent.id, []):
                    step_id_str = str(step.id)
                    if step_id_str in existing_step_ids:
                        skipped += 1
                        continue
                    new_steps.append({
                        "id": step_id_str,
                        "agent_id": str(step.agent_id),
                        "title": step.title,
                        "description": step.description,
                        "order": step.order,
                        "is_required": step.is_required,
                        "source_hints": step.source_hints,
                        "source_scope": step.source_scope,
                        "created_at": step.created_at.isoformat()
                        if step.created_at
                        else datetime.now(UTC).isoformat(),
                        "updated_at": step.updated_at.isoformat()
                        if step.updated_at
                        else datetime.now(UTC).isoformat(),
                    })
                    added += 1

                if added == 0:
                    continue

                # Sort by order
                new_steps.sort(key=lambda s: s.get("order", 0))

                # Build upsert row matching the cached service's schema
                row: dict = {
                    "id": str(agent.id),
                    "owner_id": agent.owner_id,
                    "name": agent.name,
                    "description": agent.description,
                    "avatar_url": agent.avatar_url,
                    "system_prompt_template_id": str(agent.system_prompt_template_id)
                    if agent.system_prompt_template_id
                    else None,
                    "synthesis_template_id": str(agent.synthesis_template_id)
                    if agent.synthesis_template_id
                    else None,
                    "source_scope": agent.source_scope or "all",
                    "enabled_sources": agent.enabled_sources,
                    "disabled_sources": agent.disabled_sources or [],
                    "use_planner": agent.use_planner,
                    "default_depth": agent.default_depth or "medium",
                    "default_mode": agent.default_mode or "planner",
                    "enable_clarification": agent.enable_clarification,
                    "output_format": agent.output_format or "markdown",
                    "output_schema": agent.output_schema,
                    "visibility": agent.visibility or "private",
                    "model_overrides": agent.model_overrides,
                    "domain_filter_mode": agent.domain_filter_mode,
                    "include_domains": agent.include_domains,
                    "exclude_domains": agent.exclude_domains,
                    "created_at": agent.created_at.isoformat()
                    if agent.created_at
                    else datetime.now(UTC).isoformat(),
                    "updated_at": datetime.now(UTC).isoformat(),
                    "steps": new_steps,
                }
                await stack.backend.upsert_row("custom_agents", row, pk="id")
                upserted += 1
                logger.info(
                    "Backfilled agent %s (%s): added %d steps",
                    agent.id,
                    agent.name,
                    added,
                )

        logger.info(
            "Backfill complete. Agents updated: %d, steps already present (skipped): %d",
            upserted,
            skipped,
        )
    finally:
        await stack.stop()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
