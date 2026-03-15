from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from databricks_deep_research.agents.config import PoolWriteConfig
from databricks_deep_research.agents.execution.output_normalizer import NormalizedResearchOutput, is_semantically_empty


@dataclass(frozen=True)
class PoolWriteBatch:
    pool_name: str
    extract: str
    items: list[Any]
    skip_reason: str | None = None


def build_research_pool_batch(
    normalized: NormalizedResearchOutput,
    pool_write: PoolWriteConfig,
    output_key: str,
) -> PoolWriteBatch:
    if pool_write.extract == "sources":
        items = [] if normalized.skip_source_writes else list(normalized.sources)
        return PoolWriteBatch(pool_write.pool, pool_write.extract, items, None if items else "empty_sources")
    if pool_write.extract in {output_key, "findings", "observation"}:
        text_value = normalized.observation_text or normalized.findings_text or normalized.state_text
        items = [] if normalized.skip_observation_writes or not text_value.strip() else [text_value]
        return PoolWriteBatch(pool_write.pool, pool_write.extract, items, None if items else "empty_observation_text")
    return PoolWriteBatch(pool_write.pool, pool_write.extract, [], "unsupported_extract")


def extract_pool_items(output: Any, pool_write: PoolWriteConfig, output_key: str = "") -> list[Any]:
    if pool_write.extract == output_key and isinstance(output, str) and output.strip():
        return [output]
    current = output
    for part in pool_write.extract.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return []
        if current is None:
            return []
    if isinstance(current, list):
        return [item for item in current if not is_semantically_empty(item)]
    if is_semantically_empty(current):
        return []
    return [current]
