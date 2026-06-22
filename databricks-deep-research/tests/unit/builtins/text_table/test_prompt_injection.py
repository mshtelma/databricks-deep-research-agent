"""Unit tests for ``render_table_bindings_prompt``.

Renders BOUND-only binding metadata into a deterministic, token-budgeted
text block suitable for prepending to system prompts. DISCOVERED bindings
are omitted by design — they are runtime-discovered and not yet validated
for prompt inclusion.
"""

from __future__ import annotations

import pytest

from databricks_deep_research.tools.builtins.text_table import (
    BindingInfo,
    BindingSource,
    RoleMap,
    TableBindingRegistry,
    render_table_bindings_prompt,
)


def _bound(
    name: str,
    fqn: str,
    *,
    description: str | None = None,
    roles: RoleMap | None = None,
    numeric_columns: tuple[str, ...] = (),
) -> BindingInfo:
    return BindingInfo(
        name=name,
        fqn=fqn,
        source=BindingSource.BOUND,
        description=description,
        roles=roles,
        numeric_columns=numeric_columns,
    )


def _discovered(name: str, fqn: str) -> BindingInfo:
    return BindingInfo(name=name, fqn=fqn, source=BindingSource.DISCOVERED)


@pytest.mark.unit
def test_render_returns_empty_when_no_bound_bindings() -> None:
    registry = TableBindingRegistry()
    out = render_table_bindings_prompt(registry)
    assert out == ""


@pytest.mark.unit
def test_render_excludes_discovered_bindings() -> None:
    registry = TableBindingRegistry()
    registry.register_bound(_bound("docs", "cat.sch.docs"))
    registry.register_discovered(_discovered("dyn", "cat.sch.dyn"))
    out = render_table_bindings_prompt(registry)
    assert "docs" in out
    assert "cat.sch.docs" in out
    assert "dyn" not in out
    assert "cat.sch.dyn" not in out


@pytest.mark.unit
def test_render_emits_each_bound_binding() -> None:
    registry = TableBindingRegistry()
    registry.register_bound(
        _bound(
            "docs",
            "cat.sch.docs",
            description="Internal documentation chunks.",
            roles=RoleMap(
                id_column="id",
                content_column="text",
                partition_column="doc",
                order_column="seq",
            ),
            numeric_columns=("seq",),
        )
    )
    registry.register_bound(
        _bound(
            "tickets",
            "cat.sch.tickets",
            description="Support ticket history.",
            roles=RoleMap(id_column="ticket_id", content_column="body"),
        )
    )
    out = render_table_bindings_prompt(registry)
    # Both bindings present.
    assert "docs" in out
    assert "tickets" in out
    # FQNs present.
    assert "cat.sch.docs" in out
    assert "cat.sch.tickets" in out
    # Descriptions present.
    assert "Internal documentation chunks." in out
    assert "Support ticket history." in out
    # Role columns surfaced.
    assert "id" in out
    assert "text" in out
    assert "ticket_id" in out
    assert "body" in out


@pytest.mark.unit
def test_render_is_deterministic_by_binding_name() -> None:
    registry_a = TableBindingRegistry()
    registry_a.register_bound(_bound("zeta", "cat.sch.zeta"))
    registry_a.register_bound(_bound("alpha", "cat.sch.alpha"))
    registry_a.register_bound(_bound("mu", "cat.sch.mu"))

    registry_b = TableBindingRegistry()
    registry_b.register_bound(_bound("alpha", "cat.sch.alpha"))
    registry_b.register_bound(_bound("mu", "cat.sch.mu"))
    registry_b.register_bound(_bound("zeta", "cat.sch.zeta"))

    out_a = render_table_bindings_prompt(registry_a)
    out_b = render_table_bindings_prompt(registry_b)
    assert out_a == out_b
    # Confirm sort order.
    pos_alpha = out_a.find("alpha")
    pos_mu = out_a.find("mu")
    pos_zeta = out_a.find("zeta")
    assert 0 <= pos_alpha < pos_mu < pos_zeta


@pytest.mark.unit
def test_render_redacts_email_pii_in_description() -> None:
    registry = TableBindingRegistry()
    registry.register_bound(
        _bound(
            "leads",
            "cat.sch.leads",
            description="Owner: jane.doe@example.com — internal only.",
        )
    )
    out = render_table_bindings_prompt(registry)
    assert "jane.doe@example.com" not in out
    assert "[redacted-email]" in out


@pytest.mark.unit
def test_render_redacts_phone_pii_in_description() -> None:
    registry = TableBindingRegistry()
    registry.register_bound(
        _bound(
            "calls",
            "cat.sch.calls",
            description="Contact +1 (415) 555-1234 for access.",
        )
    )
    out = render_table_bindings_prompt(registry)
    # The literal phone number should not appear.
    assert "555-1234" not in out
    assert "[redacted-phone]" in out


@pytest.mark.unit
def test_render_redacts_ssn_pattern_in_description() -> None:
    registry = TableBindingRegistry()
    registry.register_bound(
        _bound(
            "hr",
            "cat.sch.hr",
            description="Records keyed by SSN 123-45-6789.",
        )
    )
    out = render_table_bindings_prompt(registry)
    assert "123-45-6789" not in out
    assert "[redacted-ssn]" in out


@pytest.mark.unit
def test_render_does_not_redact_pii_outside_description() -> None:
    """PII redaction targets free-form description text only.

    Identifiers (binding name, FQN, role columns) are structurally validated
    elsewhere and must pass through verbatim, even if they happen to look
    like a digit run.
    """
    registry = TableBindingRegistry()
    registry.register_bound(
        _bound(
            "calls_2024",
            "cat.sch.calls_2024",
            description="Plain description.",
        )
    )
    out = render_table_bindings_prompt(registry)
    assert "calls_2024" in out
    assert "cat.sch.calls_2024" in out


@pytest.mark.unit
def test_render_truncates_when_token_budget_exceeded() -> None:
    registry = TableBindingRegistry()
    # 10 bindings, each with a chunky description.
    big_desc = "Synthetic description text." * 30  # ~810 chars each
    for i in range(10):
        registry.register_bound(
            _bound(
                f"binding_{i:02d}",
                f"cat.sch.binding_{i:02d}",
                description=big_desc,
            )
        )
    # Set a token budget that cannot fit everything.
    out = render_table_bindings_prompt(registry, max_tokens=200)
    assert out  # Non-empty
    # Truncation marker must appear.
    assert "[truncated" in out
    # Estimated token count (using the same heuristic the renderer uses)
    # must respect the cap.
    assert len(out) <= 200 * 4 + 200  # cap * heuristic + truncation marker slack


@pytest.mark.unit
def test_render_no_truncation_when_under_budget() -> None:
    registry = TableBindingRegistry()
    registry.register_bound(_bound("docs", "cat.sch.docs", description="short"))
    out = render_table_bindings_prompt(registry, max_tokens=10_000)
    assert "[truncated" not in out


@pytest.mark.unit
def test_render_invalid_max_tokens_raises() -> None:
    registry = TableBindingRegistry()
    registry.register_bound(_bound("docs", "cat.sch.docs"))
    with pytest.raises(ValueError):
        render_table_bindings_prompt(registry, max_tokens=0)
    with pytest.raises(ValueError):
        render_table_bindings_prompt(registry, max_tokens=-50)


@pytest.mark.unit
def test_render_emits_numeric_columns_when_present() -> None:
    registry = TableBindingRegistry()
    registry.register_bound(
        _bound(
            "metrics",
            "cat.sch.metrics",
            roles=RoleMap(id_column="k", content_column="v"),
            numeric_columns=("amount", "quantity"),
        )
    )
    out = render_table_bindings_prompt(registry)
    assert "amount" in out
    assert "quantity" in out


@pytest.mark.unit
def test_render_handles_binding_without_roles() -> None:
    registry = TableBindingRegistry()
    registry.register_bound(_bound("raw", "cat.sch.raw", description="No roles set."))
    out = render_table_bindings_prompt(registry)
    assert "raw" in out
    assert "cat.sch.raw" in out


@pytest.mark.unit
def test_render_includes_section_header() -> None:
    registry = TableBindingRegistry()
    registry.register_bound(_bound("docs", "cat.sch.docs"))
    out = render_table_bindings_prompt(registry)
    assert out.startswith("## Available text tables")


@pytest.mark.unit
def test_render_supports_custom_token_estimator() -> None:
    registry = TableBindingRegistry()
    for i in range(5):
        registry.register_bound(
            _bound(
                f"b{i}",
                f"cat.sch.b{i}",
                description="filler " * 50,
            )
        )

    # Estimator that overcounts heavily — forces truncation early.
    def heavy(text: str) -> int:
        return len(text)  # 1 token per char

    out = render_table_bindings_prompt(
        registry, max_tokens=500, token_estimator=heavy
    )
    assert "[truncated" in out
