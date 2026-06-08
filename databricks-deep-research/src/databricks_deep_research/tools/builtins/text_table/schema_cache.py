from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

DEFAULT_TTL_S: float = 600.0
DEFAULT_LRU_SIZE: int = 256


@dataclass(frozen=True)
class SchemaColumn:
    name: str
    data_type: str
    nullable: bool = True


@dataclass(frozen=True)
class Schema:
    fqn: str
    columns: tuple[SchemaColumn, ...]
    column_map: Mapping[str, SchemaColumn] = field(init=False)

    def __post_init__(self) -> None:
        # Bypass frozen=True via object.__setattr__ to install the immutable
        # column_map after dataclass __init__ runs.
        proxy = MappingProxyType({c.name: c for c in self.columns})
        object.__setattr__(self, "column_map", proxy)

    def has_column(self, name: str) -> bool:
        return name in self.column_map

    def get_column(self, name: str) -> SchemaColumn | None:
        return self.column_map.get(name)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


class SchemaCache:
    """Two-tier schema cache.

    Tier 1 is a per-step dict cleared on every begin_step()/end_step() boundary,
    so a single agent step never re-fetches a table's schema. Tier 2 is a
    process-wide LRU keyed by (fqn, sha256(token)[:16]) with a TTL, so the
    plaintext user token is never held in memory and entries expire under load.

    Not thread-safe. Designed for single-threaded asyncio contexts; concurrent
    calls from threads can corrupt the OrderedDict LRU state.
    """

    def __init__(
        self,
        fetcher: Callable[[str, str], Schema],
        *,
        ttl_s: float = DEFAULT_TTL_S,
        lru_size: int = DEFAULT_LRU_SIZE,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._fetcher = fetcher
        self._ttl_s = ttl_s
        self._lru_size = lru_size
        self._now = now
        self._step_cache: dict[tuple[str, str], Schema] = {}
        self._process_cache: OrderedDict[tuple[str, str], tuple[Schema, float]] = (
            OrderedDict()
        )

    def begin_step(self) -> None:
        self._step_cache = {}

    def end_step(self) -> None:
        self._step_cache = {}

    def refresh(self, fqn: str, user_token: str) -> None:
        key = (fqn, _hash_token(user_token))
        self._step_cache.pop(key, None)
        self._process_cache.pop(key, None)

    def get(self, fqn: str, user_token: str) -> Schema:
        key = (fqn, _hash_token(user_token))
        step_hit = self._step_cache.get(key)
        if step_hit is not None:
            return step_hit
        process_hit = self._process_cache.get(key)
        if process_hit is not None:
            schema, expires_at = process_hit
            if self._now() < expires_at:
                self._process_cache.move_to_end(key)
                self._step_cache[key] = schema
                return schema
            del self._process_cache[key]
        schema = self._fetcher(fqn, user_token)
        self._populate(key, schema)
        return schema

    def _populate(self, key: tuple[str, str], schema: Schema) -> None:
        expires_at = self._now() + self._ttl_s
        self._process_cache[key] = (schema, expires_at)
        self._process_cache.move_to_end(key)
        while len(self._process_cache) > self._lru_size:
            self._process_cache.popitem(last=False)
        self._step_cache[key] = schema
