from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

import pytest

from databricks_deep_research.tools.builtins.text_table.schema_cache import (
    Schema,
    SchemaCache,
    SchemaColumn,
)


def _schema(fqn: str = "cat.s.t") -> Schema:
    return Schema(
        fqn=fqn,
        columns=(
            SchemaColumn(name="id", data_type="STRING", nullable=False),
            SchemaColumn(name="content", data_type="STRING"),
        ),
    )


def test_schema_column_holds_fields() -> None:
    col = SchemaColumn(name="id", data_type="BIGINT", nullable=False)
    assert col.name == "id"
    assert col.data_type == "BIGINT"
    assert col.nullable is False


def test_schema_has_column_and_get_column() -> None:
    schema = _schema()
    assert schema.has_column("id") is True
    assert schema.has_column("missing") is False
    got = schema.get_column("content")
    assert got is not None
    assert got.data_type == "STRING"
    assert schema.get_column("missing") is None


def test_cache_uses_step_tier_first() -> None:
    schema = _schema()
    fetcher = MagicMock(return_value=schema)
    cache = SchemaCache(fetcher=fetcher)
    cache.begin_step()
    a = cache.get(fqn="cat.s.t", user_token="tok")
    b = cache.get(fqn="cat.s.t", user_token="tok")
    assert a is b
    assert fetcher.call_count == 1


def test_cache_falls_through_to_process_tier() -> None:
    schema = _schema()
    fetcher = MagicMock(return_value=schema)
    cache = SchemaCache(fetcher=fetcher)
    cache.begin_step()
    cache.get(fqn="cat.s.t", user_token="tok")
    cache.end_step()
    cache.begin_step()
    cache.get(fqn="cat.s.t", user_token="tok")
    assert fetcher.call_count == 1


def test_cache_invokes_fetcher_when_cold() -> None:
    schema = _schema()
    fetcher = MagicMock(return_value=schema)
    cache = SchemaCache(fetcher=fetcher)
    cache.begin_step()
    out = cache.get(fqn="cat.s.t", user_token="tok")
    assert out is schema
    assert fetcher.call_count == 1
    fetcher.assert_called_once_with("cat.s.t", "tok")


def test_begin_step_clears_step_tier() -> None:
    schema = _schema()
    fetcher = MagicMock(return_value=schema)
    cache = SchemaCache(fetcher=fetcher)
    cache.begin_step()
    cache.get(fqn="cat.s.t", user_token="tok")
    cache.begin_step()
    cache.get(fqn="cat.s.t", user_token="tok")
    assert fetcher.call_count == 1


def test_refresh_invalidates_both_tiers() -> None:
    schema = _schema()
    fetcher = MagicMock(return_value=schema)
    cache = SchemaCache(fetcher=fetcher)
    cache.begin_step()
    cache.get(fqn="cat.s.t", user_token="tok")
    cache.refresh(fqn="cat.s.t", user_token="tok")
    assert fetcher.call_count == 2
    cache.get(fqn="cat.s.t", user_token="tok")
    assert fetcher.call_count == 2


def test_process_tier_uses_hashed_token() -> None:
    schema = _schema()
    fetcher = MagicMock(return_value=schema)
    cache = SchemaCache(fetcher=fetcher)
    plaintext_token = "super-secret-token-do-not-leak"

    cache.begin_step()
    cache.get(fqn="cat.s.t", user_token=plaintext_token)
    cache.end_step()
    cache.begin_step()
    cache.get(fqn="cat.s.t", user_token=plaintext_token)

    assert fetcher.call_count == 1
    expected_hash = hashlib.sha256(plaintext_token.encode("utf-8")).hexdigest()[:16]
    process_repr = str(cache._process_cache)
    assert plaintext_token not in process_repr
    assert expected_hash in process_repr


def test_process_tier_ttl_expiry() -> None:
    schema = _schema()
    fetcher = MagicMock(return_value=schema)
    fake_time = [1000.0]

    def now() -> float:
        return fake_time[0]

    cache = SchemaCache(fetcher=fetcher, ttl_s=10.0, now=now)
    cache.begin_step()
    cache.get(fqn="cat.s.t", user_token="tok")
    cache.end_step()
    fake_time[0] += 11.0
    cache.begin_step()
    cache.get(fqn="cat.s.t", user_token="tok")
    assert fetcher.call_count == 2


def test_process_tier_lru_eviction() -> None:
    schema_a = Schema(fqn="cat.s.a", columns=(SchemaColumn(name="x", data_type="STRING"),))
    schema_b = Schema(fqn="cat.s.b", columns=(SchemaColumn(name="x", data_type="STRING"),))
    schema_c = Schema(fqn="cat.s.c", columns=(SchemaColumn(name="x", data_type="STRING"),))
    by_fqn = {"cat.s.a": schema_a, "cat.s.b": schema_b, "cat.s.c": schema_c}

    def fetcher_fn(fqn: str, user_token: str) -> Schema:
        return by_fqn[fqn]

    fetcher = MagicMock(side_effect=fetcher_fn)
    cache = SchemaCache(fetcher=fetcher, lru_size=2)

    cache.begin_step()
    cache.get(fqn="cat.s.a", user_token="tok")
    cache.end_step()
    cache.begin_step()
    cache.get(fqn="cat.s.b", user_token="tok")
    cache.end_step()
    cache.begin_step()
    cache.get(fqn="cat.s.c", user_token="tok")
    cache.end_step()

    assert fetcher.call_count == 3

    cache.begin_step()
    cache.get(fqn="cat.s.a", user_token="tok")
    assert fetcher.call_count == 4

    cache.end_step()
    cache.begin_step()
    cache.get(fqn="cat.s.c", user_token="tok")
    assert fetcher.call_count == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
