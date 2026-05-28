"""Tests for SafeTemplateRenderer."""

from __future__ import annotations

import pytest

from databricks_deep_research.agents.prompts.background import BACKGROUND_USER_PROMPT
from databricks_deep_research.agents.prompts.planner import (
    PLANNER_SYSTEM_PROMPT,
    SOURCE_AWARE_PLANNER_SYSTEM_PROMPT,
)
from databricks_deep_research.templates.renderer import (
    SafeTemplateRenderer,
    TemplateSecurityError,
)


@pytest.fixture
def renderer() -> SafeTemplateRenderer:
    return SafeTemplateRenderer()


class TestVariableSubstitution:
    def test_simple_variable(self, renderer: SafeTemplateRenderer) -> None:
        assert renderer.render("Hello {name}!", {"name": "world"}) == "Hello world!"

    def test_multiple_variables(self, renderer: SafeTemplateRenderer) -> None:
        result = renderer.render("{a} and {b}", {"a": "X", "b": "Y"})
        assert result == "X and Y"

    def test_missing_variable_becomes_empty(self, renderer: SafeTemplateRenderer) -> None:
        assert renderer.render("Hi {name}!", {}) == "Hi !"

    def test_none_variable_becomes_empty(self, renderer: SafeTemplateRenderer) -> None:
        assert renderer.render("Hi {name}!", {"name": None}) == "Hi !"

    def test_non_string_value_stringified(self, renderer: SafeTemplateRenderer) -> None:
        assert renderer.render("Count: {n}", {"n": 42}) == "Count: 42"


class TestConditionalBlocks:
    def test_if_truthy(self, renderer: SafeTemplateRenderer) -> None:
        tpl = "Start{%if show%} visible{%endif%} end"
        assert renderer.render(tpl, {"show": True}) == "Start visible end"

    def test_if_falsy(self, renderer: SafeTemplateRenderer) -> None:
        tpl = "Start{%if show%} visible{%endif%} end"
        assert renderer.render(tpl, {"show": False}) == "Start end"

    def test_if_missing_variable(self, renderer: SafeTemplateRenderer) -> None:
        tpl = "A{%if x%}B{%endif%}C"
        assert renderer.render(tpl, {}) == "AC"


class TestForLoops:
    def test_simple_for(self, renderer: SafeTemplateRenderer) -> None:
        tpl = "{%for item in items%}{item} {%endfor%}"
        result = renderer.render(tpl, {"items": ["a", "b", "c"]})
        assert result == "a b c "

    def test_for_empty_list(self, renderer: SafeTemplateRenderer) -> None:
        tpl = "{%for item in items%}{item}{%endfor%}"
        assert renderer.render(tpl, {"items": []}) == ""


class TestSecurity:
    def test_rejects_attribute_traversal(self, renderer: SafeTemplateRenderer) -> None:
        with pytest.raises(TemplateSecurityError):
            renderer.render("{obj.attr}", {"obj": "x"})

    def test_rejects_dunder(self, renderer: SafeTemplateRenderer) -> None:
        with pytest.raises(TemplateSecurityError):
            renderer.render("{__class__}", {})

    def test_rejects_brackets(self, renderer: SafeTemplateRenderer) -> None:
        with pytest.raises(TemplateSecurityError):
            renderer.render("{items[0]}", {"items": [1]})


class TestExtractVariables:
    def test_extracts_all_names(self, renderer: SafeTemplateRenderer) -> None:
        tpl = "{a} {%if b%}{c}{%endif%} {%for d in e%}{d}{%endfor%}"
        names = renderer.extract_variables(tpl)
        assert names == {"a", "b", "c", "d", "e"}

    def test_ignores_literal_brace_escapes(self, renderer: SafeTemplateRenderer) -> None:
        tpl = '{{schema}}: {real_var}'
        names = renderer.extract_variables(tpl)
        assert names == {"real_var"}

    def test_ignores_literal_braces_with_json(self, renderer: SafeTemplateRenderer) -> None:
        tpl = '{real} {{\n  "key": "value"\n}} {%if flag%}yes{%endif%}'
        names = renderer.extract_variables(tpl)
        assert names == {"real", "flag"}

    def test_background_prompt_json_example_is_safe(self, renderer: SafeTemplateRenderer) -> None:
        names = renderer.extract_variables(BACKGROUND_USER_PROMPT)
        assert names == {"query", "conversation_history"}

    def test_source_aware_planner_system_prompt_is_safe(self, renderer: SafeTemplateRenderer) -> None:
        # ``{tool_catalog}`` is the per-agent catalog block injected at
        # workflow build time (Phase 1 of the tool-catalog auto-injection
        # plan). Every other JSON example, schema sample, and replan-output
        # placeholder in this prompt is brace-escaped, so the renderer
        # must see exactly one unfilled template variable.
        names = renderer.extract_variables(SOURCE_AWARE_PLANNER_SYSTEM_PROMPT)
        assert names == {"tool_catalog"}

    def test_planner_system_prompt_is_safe(self, renderer: SafeTemplateRenderer) -> None:
        # The base planner system prompt has two unfilled template
        # variables: ``{query}`` (the active research question, embedded
        # inline in the system role) and ``{tool_catalog}`` (the per-agent
        # catalog block injected at workflow build time, Phase 1 of the
        # tool-catalog auto-injection plan). Every other JSON example,
        # schema sample, and replan-output placeholder is brace-escaped,
        # so the renderer must see exactly these two variables.
        names = renderer.extract_variables(PLANNER_SYSTEM_PROMPT)
        assert names == {"query", "tool_catalog"}
