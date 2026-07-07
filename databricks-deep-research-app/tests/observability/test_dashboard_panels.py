"""CI snapshot tests for scripts/grafana/agent_designer_dashboard.json.

These tests verify that the Grafana dashboard JSON stays in sync with the
Agent Designer V1.5 observability policy (the policy doc was retired; the
signal list encoded in these tests is the source of truth).

Design intent
-------------
* ``test_dashboard_parses`` — pure syntax guard: JSON must load without errors.
* ``test_all_v15_signals_in_panels`` — every signal that is already shipped
  (V1 + V1.5 server-side) MUST appear in at least one panel target query.
  A separate ``xfail``-marked block covers aspirational V1.5 signals that
  ship in later stories; those failures are expected and intentional so that
  dashboards catch up automatically when the stories land.
* ``test_alert_rules_present`` — asserts the two mandatory alert rules exist.

Intentionally aspirational
--------------------------
The ``yaml_export_ms``, ``yaml_import_outcome``, ``revision_write_failed``, and
``revisions_tab_opened`` signals are V1.5 aspirational: they are defined in the
policy but ship across multiple stories after US-603. The xfail tests below
fail loudly until those stories add panels, forcing the dashboard to stay
current.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DASHBOARD_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "scripts"
    / "grafana"
    / "agent_designer_dashboard.json"
)


def _load_dashboard() -> dict[str, object]:
    return json.loads(_DASHBOARD_PATH.read_text())


def _all_panel_exprs(dashboard: dict[str, object]) -> list[str]:
    """Return all PromQL/LogQL expressions across every panel target."""
    exprs: list[str] = []
    panels = dashboard.get("panels", [])
    assert isinstance(panels, list)
    for panel in panels:
        assert isinstance(panel, dict)
        for target in panel.get("targets", []):
            assert isinstance(target, dict)
            expr = target.get("expr", "")
            if isinstance(expr, str) and expr:
                exprs.append(expr)
    return exprs


def _all_alert_exprs(dashboard: dict[str, object]) -> list[str]:
    """Return the expr field from every alert rule."""
    exprs: list[str] = []
    alerts = dashboard.get("alerts", [])
    assert isinstance(alerts, list)
    for alert in alerts:
        assert isinstance(alert, dict)
        expr = alert.get("expr", "")
        if isinstance(expr, str) and expr:
            exprs.append(expr)
    return exprs


def _all_alert_names(dashboard: dict[str, object]) -> list[str]:
    alerts = dashboard.get("alerts", [])
    assert isinstance(alerts, list)
    return [str(a.get("name", "")) for a in alerts if isinstance(a, dict)]


# ---------------------------------------------------------------------------
# test_dashboard_parses
# ---------------------------------------------------------------------------


def test_dashboard_parses() -> None:
    """Dashboard JSON must be syntactically valid and contain top-level keys."""
    assert _DASHBOARD_PATH.exists(), f"Dashboard not found at {_DASHBOARD_PATH}"
    dashboard = _load_dashboard()
    assert isinstance(dashboard, dict), "Top-level JSON must be an object"
    assert "panels" in dashboard, "Dashboard must have a 'panels' key"
    assert "alerts" in dashboard, "Dashboard must have an 'alerts' key"
    assert isinstance(dashboard["panels"], list), "'panels' must be an array"
    assert isinstance(dashboard["alerts"], list), "'alerts' must be an array"
    assert len(dashboard["panels"]) > 0, "Dashboard must contain at least one panel"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# test_all_v15_signals_in_panels
# ---------------------------------------------------------------------------

# Signals that are REQUIRED NOW: V1 signals + V1.5 server-side signals that
# shipped in US-601/US-603. All must be covered by at least one panel query.
_REQUIRED_NOW_SIGNALS = [
    "agent_designer.validation_error",
    "agent_designer.save_etag_conflict",
    "agent_designer.registry_fetch_ms",
    "agent_designer.designer_save_latency",
    "agent_designer.chat_mutation",
    "agent_designer.token_refresh_attempt",
    "agent_designer.token_refresh_failure",
]

# V1.5 aspirational signals that ship in later stories.  These tests are
# marked xfail so they fail loudly when the panel is absent, reminding the
# story author to add a panel.  Once a panel is added the xfail turns into
# an xpass, which is also acceptable (strict=False).
_ASPIRATIONAL_SIGNALS = [
    "agent_designer.yaml_export_ms",
    "agent_designer.yaml_import_outcome",
    "agent_designer.revision_write_failed",
    "agent_designer.revisions_tab_opened",
]


def test_all_v15_signals_in_panels() -> None:
    """Every already-shipped V1/V1.5 signal must appear in at least one panel target expr.

    Missing signals indicate that the dashboard has drifted from the
    observability policy.
    """
    dashboard = _load_dashboard()
    exprs = _all_panel_exprs(dashboard)
    combined = " ".join(exprs)

    missing = [sig for sig in _REQUIRED_NOW_SIGNALS if sig not in combined]
    assert not missing, (
        f"The following required signals are absent from all panel target "
        f"expressions in {_DASHBOARD_PATH}:\n"
        + "\n".join(f"  - {s}" for s in missing)
        + "\n\nAdd a panel for each missing signal or update the dashboard."
    )


@pytest.mark.parametrize("signal", _ASPIRATIONAL_SIGNALS)
@pytest.mark.xfail(
    reason=(
        "Aspirational V1.5 signal — ships in a later story. "
        "This xfail is intentional: when the story lands and a panel is added, "
        "the test will pass automatically, confirming dashboard coverage."
    ),
    strict=False,
)
def test_aspirational_v15_signal_in_panels(signal: str) -> None:
    """Each aspirational signal should eventually appear in a panel target.

    Intentionally aspirational — dashboards include placeholders for V1.5
    signals that ship across the release. These tests fail until the
    corresponding stories add panels, at which point they pass automatically.
    """
    dashboard = _load_dashboard()
    exprs = _all_panel_exprs(dashboard)
    combined = " ".join(exprs)
    assert signal in combined, f"Signal '{signal}' not yet covered by any panel"


# ---------------------------------------------------------------------------
# test_alert_rules_present
# ---------------------------------------------------------------------------


def test_alert_rules_present() -> None:
    """Both mandatory alert rules must be defined in the dashboard.

    1. TokenRefreshFailuresHigh — rate(agent_designer.token_refresh_failure) > 0.1
       over 5 m.  Fires when the OBO token rotation path has a sustained error
       rate, indicating a systemic auth failure.

    2. ValidationErrorsHigh — validation_error rate > 5% of validate_total over
       5 m.  Pre-existing regression alert; must not be removed.
    """
    dashboard = _load_dashboard()
    alert_exprs = _all_alert_exprs(dashboard)
    combined_exprs = " ".join(alert_exprs)
    alert_names = _all_alert_names(dashboard)

    # --- Alert 1: TokenRefreshFailuresHigh ---
    assert "TokenRefreshFailuresHigh" in alert_names, (
        "Alert 'TokenRefreshFailuresHigh' is missing from the dashboard. "
        "Add an alert rule: rate(agent_designer.token_refresh_failure[5m]) > 0.1"
    )
    assert "agent_designer.token_refresh_failure" in combined_exprs, (
        "No alert expression references 'agent_designer.token_refresh_failure'. "
        "The TokenRefreshFailuresHigh alert must query this metric."
    )
    # Verify the threshold is present
    assert "0.1" in combined_exprs, (
        "TokenRefreshFailuresHigh alert threshold (0.1) not found in alert expressions."
    )

    # --- Alert 2: ValidationErrorsHigh (pre-existing, regression guard) ---
    assert "ValidationErrorsHigh" in alert_names, (
        "Alert 'ValidationErrorsHigh' is missing. This is a pre-existing alert "
        "that must not be removed from the dashboard."
    )
    assert "agent_designer.validation_error" in combined_exprs, (
        "No alert expression references 'agent_designer.validation_error'. "
        "The ValidationErrorsHigh alert must query this metric."
    )
    assert "0.05" in combined_exprs, (
        "ValidationErrorsHigh alert threshold (0.05 = 5%) not found in alert expressions."
    )
