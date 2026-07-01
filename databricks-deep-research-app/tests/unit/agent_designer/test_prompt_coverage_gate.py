"""Part 2 — deterministic prompt-term coverage save gate (pure, no API import).

  * ``prompt_term_coverage_errors`` — the Designer-Goal-stripped synthesizer
    coverage check, elevated to a blocking ``kind="coverage"`` gate. Key
    regression (codex BLOCKER 2): the appended ``## Designer Goal`` block alone
    must NOT satisfy coverage.
  * MAJOR 4 — the save critic's semantic projection now includes
    ``user_prompt_template`` and ``VALIDATOR_VERSION`` is bumped.

(The route-level ``_raise_if_coverage_blocks`` gate is tested in
``tests/unit/api/test_agents_v2_critic_gate.py`` — importing the api package
requires the Settings env configured only for the api test tree.)
"""

from __future__ import annotations

from deep_research.agent_designer.semantic_validation import prompt_term_coverage_errors
from deep_research.agent_designer.workflow_validation import (
    VALIDATOR_VERSION,
    semantic_projection,
)

_TERMS = ["fundamentals", "earnings", "competitors"]


def _definition(synth_system_prompt: str) -> dict:
    return {
        "required_prompt_terms": _TERMS,
        "root": {
            "id": "root",
            "type": "sequence",
            "config": {},
            "children": [
                {
                    "id": "synth",
                    "type": "agent",
                    "label": "Synthesizer",
                    "config": {
                        "subtype": "synthesizer",
                        "system_prompt": synth_system_prompt,
                    },
                    "children": [],
                }
            ],
        },
    }


def test_designer_goal_text_alone_does_not_satisfy_coverage() -> None:
    # REGRESSION (codex BLOCKER 2): the appended ## Designer Goal block contains
    # the full intent verbatim, so a raw scan would falsely pass. The check must
    # strip it and evaluate only the synthesizer's OWN authored content.
    goal = "\n\n## Designer Goal\nCover fundamentals, earnings, and competitors."
    errors = prompt_term_coverage_errors(_definition("You synthesize a report." + goal))
    assert errors, "Designer Goal text alone must NOT satisfy coverage"
    assert all(e.kind == "coverage" for e in errors)


def test_coverage_passes_when_synthesizer_references_terms() -> None:
    sp = "Synthesize a report covering fundamentals, earnings, and competitors with metrics."
    assert prompt_term_coverage_errors(_definition(sp)) == []


def test_coverage_no_terms_never_blocks() -> None:
    bare = {
        "root": {
            "id": "root",
            "type": "sequence",
            "config": {},
            "children": [
                {
                    "id": "synth",
                    "type": "agent",
                    "config": {"subtype": "synthesizer", "system_prompt": "generic"},
                    "children": [],
                }
            ],
        }
    }
    assert prompt_term_coverage_errors(bare) == []


def test_coverage_no_synthesizer_never_blocks() -> None:
    d = _definition("ignored")
    d["root"]["children"][0]["config"]["subtype"] = "researcher"
    assert prompt_term_coverage_errors(d) == []


def test_semantic_projection_includes_user_prompt_template() -> None:
    definition = {
        "root": {
            "id": "root",
            "type": "sequence",
            "config": {},
            "children": [
                {
                    "id": "a",
                    "type": "agent",
                    "config": {
                        "subtype": "researcher",
                        "system_prompt": "sys",
                        "user_prompt_template": "UNIQUE_USERPROMPT_TOKEN",
                    },
                    "children": [],
                }
            ],
        }
    }
    proj = semantic_projection(definition, "intent", None)
    assert "UNIQUE_USERPROMPT_TOKEN" in proj


def test_validator_version_bumped() -> None:
    assert VALIDATOR_VERSION == "v2"
