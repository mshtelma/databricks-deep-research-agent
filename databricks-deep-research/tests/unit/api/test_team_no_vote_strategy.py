"""Vote strategy is explicitly rejected at construction time."""

from __future__ import annotations

import pytest

from databricks_deep_research.api import Agent, Team


def test_vote_strategy_raises_value_error() -> None:
    with pytest.raises(ValueError, match="vote.*not supported"):
        Team(members=[Agent(name="m1")], leader=Agent(name="l"), strategy="vote")


def test_vote_error_message_suggests_alternative() -> None:
    with pytest.raises(ValueError, match=r"Parallel.*synthesizer"):
        Team(members=[Agent(name="m1")], strategy="vote")
