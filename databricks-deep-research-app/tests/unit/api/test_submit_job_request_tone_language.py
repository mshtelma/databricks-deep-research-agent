"""Validation tests for the per-run tone + output-language fields on
``SubmitJobRequest`` (spec §4.4).

The unit-API conftest sets ``STORAGE_BACKEND=fake`` before importing app
modules, so importing the route schema here is safe.
"""

from __future__ import annotations

from uuid import uuid4

from deep_research.api.v1.jobs import SubmitJobRequest


def test_tone_and_language_default_to_none() -> None:
    """Absent => None (byte-identical to the pre-feature request)."""
    req = SubmitJobRequest(chat_id=uuid4(), query="What is AI?")
    assert req.tone is None
    assert req.output_language is None


def test_tone_and_language_accepted_when_set() -> None:
    req = SubmitJobRequest(
        chat_id=uuid4(),
        query="What is AI?",
        tone="objective",
        output_language="Spanish",
    )
    assert req.tone == "objective"
    assert req.output_language == "Spanish"


def test_output_language_length_capped() -> None:
    """A pathologically long language string is rejected (max_length guard)."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SubmitJobRequest(
            chat_id=uuid4(),
            query="What is AI?",
            output_language="x" * 200,
        )


def test_other_fields_unaffected() -> None:
    """Adding the new optional fields does not change existing defaults."""
    req = SubmitJobRequest(chat_id=uuid4(), query="What is AI?")
    assert req.query_mode == "deep_research"
    assert req.research_depth == "auto"
    assert req.verify_sources is True
    assert req.output_type is None
