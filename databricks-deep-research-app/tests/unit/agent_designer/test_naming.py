from __future__ import annotations

from deep_research.agent_designer.naming import semantic_lane_label, semantic_node_label


def test_semantic_node_label_replaces_generic_researcher_ordinal() -> None:
    assert (
        semantic_node_label(
            node_type="agent",
            config={"subtype": "researcher"},
            requested_label="Researcher 2",
        )
        == "Evidence Researcher"
    )


def test_semantic_node_label_uses_config_context_for_generic_label() -> None:
    assert (
        semantic_node_label(
            node_type="agent",
            config={"subtype": "researcher", "output_key": "market_risk_findings"},
            requested_label="Researcher 1",
        )
        == "Market Risk Findings Researcher"
    )


def test_semantic_node_label_keeps_meaningful_label() -> None:
    assert (
        semantic_node_label(
            node_type="agent",
            config={"subtype": "researcher"},
            requested_label="Treasury Calendar Evidence",
        )
        == "Treasury Calendar Evidence"
    )


def test_semantic_lane_label_is_description_first() -> None:
    assert (
        semantic_lane_label("AWS vs Azure cloud revenue and margin evidence", 1)
        == "AWS Vs Azure Cloud Revenue Margin Evidence Researcher"
    )
