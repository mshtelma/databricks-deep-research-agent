"""Unit tests for the tolerant LLM-JSON parsing ladder.

Black-box: text in, ``(value, recovered)`` out. Each case exercises one rung
of the ladder (clean / fenced / repair / regex-block / model-coerce / garbage)
plus the suspicion guard and the ``recovered`` telemetry flag.
"""
from __future__ import annotations

import logging

import pytest
from pydantic import BaseModel

from databricks_deep_research.agents.json_parsing import parse_llm_json


class _Tiny(BaseModel):
    name: str
    count: int


# ---------------------------------------------------------------------------
# Ladder rungs
# ---------------------------------------------------------------------------


def test_clean_json_dict() -> None:
    value, recovered = parse_llm_json('{"a": 1, "b": "two"}')
    assert value == {"a": 1, "b": "two"}
    assert recovered is False


def test_json_fenced() -> None:
    text = '```json\n{"a": 1}\n```'
    value, recovered = parse_llm_json(text)
    assert value == {"a": 1}
    assert recovered is False


def test_bare_fenced_no_lang() -> None:
    text = '```\n{"a": 1}\n```'
    value, recovered = parse_llm_json(text)
    assert value == {"a": 1}
    assert recovered is False


def test_trailing_comma() -> None:
    value, recovered = parse_llm_json('{"a": 1, "b": 2,}')
    assert value == {"a": 1, "b": 2}
    assert recovered is True


def test_single_quoted_keys_and_values() -> None:
    value, recovered = parse_llm_json("{'a': 'one', 'b': 'two'}")
    assert value == {"a": "one", "b": "two"}
    assert recovered is True


def test_prose_wrapped_object() -> None:
    text = 'Here is the answer: {"a": 1, "b": 2} hope this helps'
    value, recovered = parse_llm_json(text)
    assert value == {"a": 1, "b": 2}
    assert recovered is True


def test_prose_wrapped_array() -> None:
    text = "Sure! The list is [1, 2, 3]. Done."
    value, recovered = parse_llm_json(text)
    assert value == [1, 2, 3]
    assert recovered is True


def test_pure_garbage_returns_default() -> None:
    value, _ = parse_llm_json("not json at all", default={"d": True})
    assert value == {"d": True}


def test_substantive_garbage_returns_default_recovered_true() -> None:
    # >=20 non-whitespace chars of unrecoverable text: a real payload we tried
    # and failed to parse -> default with recovered=True.
    value, recovered = parse_llm_json(
        "this is a sentence with no json whatsoever in it", default={"d": True}
    )
    assert value == {"d": True}
    assert recovered is True


def test_trivial_input_returns_default_not_recovered() -> None:
    value, recovered = parse_llm_json("", default=None)
    assert value is None
    assert recovered is False


def test_short_garbage_returns_default_not_recovered() -> None:
    # < 20 non-whitespace chars: below the substantive threshold, so the
    # exhausted ladder reports recovered=False (not a "real payload").
    value, recovered = parse_llm_json("nope", default=None)
    assert value is None
    assert recovered is False


# ---------------------------------------------------------------------------
# model= coercion
# ---------------------------------------------------------------------------


def test_model_valid_returns_instance() -> None:
    value, recovered = parse_llm_json('{"name": "x", "count": 3}', model=_Tiny)
    assert isinstance(value, _Tiny)
    assert value.name == "x"
    assert value.count == 3
    assert recovered is False


def test_model_invalid_payload_returns_default() -> None:
    # Valid JSON, but the wrong shape for _Tiny (missing/typed fields).
    value, recovered = parse_llm_json('{"name": "x"}', model=_Tiny, default=None)
    assert value is None
    assert recovered is True


def test_model_valid_via_repair() -> None:
    value, recovered = parse_llm_json("{'name': 'y', 'count': 7,}", model=_Tiny)
    assert isinstance(value, _Tiny)
    assert value.count == 7
    assert recovered is True


# ---------------------------------------------------------------------------
# Suspicion guard: substantive input that json_repair collapses to {}
# ---------------------------------------------------------------------------


def test_empty_from_substantive_input_returns_default() -> None:
    # >=500 chars of prose that json_repair would coerce to an empty/near-empty
    # container must be rejected by the suspicion guard, not returned.
    prose = (
        "The researcher considered many angles and wrote at length about the "
        "topic without ever emitting a JSON object. " * 20
    )
    assert len(prose) >= 500
    value, recovered = parse_llm_json(prose, default={"fallback": True})
    assert value == {"fallback": True}
    assert recovered is True


# ---------------------------------------------------------------------------
# recovered flag + telemetry
# ---------------------------------------------------------------------------


def test_recovered_flag_false_for_clean_true_for_repaired() -> None:
    _, clean = parse_llm_json('{"a": 1}')
    _, repaired = parse_llm_json('{"a": 1,}')
    assert clean is False
    assert repaired is True


def test_telemetry_logged_once_on_recovery(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        parse_llm_json('{"a": 1,}', site="unit_site")
    records = [r for r in caplog.records if "LLM_JSON_RECOVERED" in r.getMessage()]
    assert len(records) == 1
    assert "site=unit_site" in records[0].getMessage()
    assert "stage=repair" in records[0].getMessage()


def test_telemetry_not_logged_on_clean(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        parse_llm_json('{"a": 1}')
    records = [r for r in caplog.records if "LLM_JSON_RECOVERED" in r.getMessage()]
    assert records == []
