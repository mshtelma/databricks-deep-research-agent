from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any

from pydantic import BaseModel, ConfigDict


class PoolStats(BaseModel):
    attempted: int = 0
    added: int = 0
    rejected_duplicate_key: int = 0
    rejected_duplicate_hash: int = 0
    evicted: int = 0

logger = logging.getLogger(__name__)


class PoolConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    item_type: str = "text"  # text, source, claim, evidence
    dedup_key: str | None = None  # Field for key-based dedup
    dedup_content_hash: bool = True
    max_items: int = 0  # 0 = unlimited


class PoolState:
    def __init__(self, config: PoolConfig) -> None:
        self.config = config
        self.items: list[Any] = []
        self.seen_keys: set[str] = set()
        self.seen_hashes: set[str] = set()
        self.stats = PoolStats()
        self._lock = asyncio.Lock()

    def _content_hash(self, item: Any) -> str:
        """Hash item content for dedup."""
        return hashlib.sha256(
            json.dumps(item, sort_keys=True, default=str).encode()
        ).hexdigest()

    def _extract_key(self, item: Any) -> str | None:
        """Extract dedup key from item."""
        if self.config.dedup_key is None:
            return None
        if isinstance(item, dict):
            return str(item.get(self.config.dedup_key, ""))
        return str(getattr(item, self.config.dedup_key, ""))

    def add(self, item: Any) -> bool:
        """Add item with dedup. Returns True if added, False if duplicate."""
        self.stats.attempted += 1
        # Key-based dedup
        if self.config.dedup_key:
            key = self._extract_key(item)
            if key and key in self.seen_keys:
                self.stats.rejected_duplicate_key += 1
                logger.debug(
                    "POOL_DEDUP_KEY pool=%s key=%s",
                    self.config.name, str(key)[:100],
                )
                return False
            if key:
                self.seen_keys.add(key)

        # Content hash dedup
        if self.config.dedup_content_hash:
            h = self._content_hash(item)
            if h in self.seen_hashes:
                self.stats.rejected_duplicate_hash += 1
                logger.debug(
                    "POOL_DEDUP_HASH pool=%s hash=%s",
                    self.config.name, h[:16],
                )
                return False
            self.seen_hashes.add(h)

        # Capacity check — evict oldest if at max
        if self.config.max_items > 0 and len(self.items) >= self.config.max_items:
            evicted = self.items.pop(0)
            self.stats.evicted += 1
            logger.info(
                "POOL_EVICTION pool=%s max_items=%d evicted=%s",
                self.config.name, self.config.max_items,
                str(evicted)[:100],
            )

        self.items.append(item)
        self.stats.added += 1
        logger.info(
            "POOL_ADD pool=%s size=%d item_preview=%s",
            self.config.name, len(self.items),
            str(item)[:100],
        )
        return True

    async def extend_async(self, items: list[Any]) -> int:
        """Bulk add with lock. Returns count of items actually added."""
        added = 0
        async with self._lock:
            for item in items:
                if self.add(item):
                    added += 1
        return added

    def search_scored(self, query: str, limit: int = 10) -> list[tuple[float, Any]]:
        """Keyword-based search with fallback overlap scores."""
        query_words = set(query.lower().split())
        scored: list[tuple[float, Any]] = []
        for item in self.items:
            text = (
                json.dumps(item, default=str).lower()
                if not isinstance(item, str)
                else item.lower()
            )
            item_words = set(text.split())
            overlap = len(query_words & item_words)
            if overlap > 0:
                scored.append((overlap / max(len(query_words), 1), item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:limit]

    def search(self, query: str, limit: int = 10) -> list[Any]:
        """Keyword-based search (BM25 optional, this is the fallback)."""
        results = [item for _, item in self.search_scored(query, limit=limit)]
        logger.info(
            "POOL_SEARCH pool=%s query=%s results=%d pool_size=%d",
            self.config.name, query[:100], len(results), len(self.items),
        )
        return results

    def get_recent(self, n: int = 10) -> list[Any]:
        return self.items[-n:]

    def snapshot(self) -> list[Any]:
        """Return a shallow snapshot of current pool items."""
        return list(self.items)

    def count(self) -> int:
        return len(self.items)

    def topics(self) -> list[str]:
        """Extract unique topic labels from items."""
        topics_set: set[str] = set()
        for item in self.items:
            if isinstance(item, dict) and "topic" in item:
                topics_set.add(str(item["topic"]))
            elif isinstance(item, dict) and "title" in item:
                topics_set.add(str(item["title"]))
        return sorted(topics_set)

    def get_by_index(self, index: int) -> Any | None:
        if 0 <= index < len(self.items):
            return self.items[index]
        return None
