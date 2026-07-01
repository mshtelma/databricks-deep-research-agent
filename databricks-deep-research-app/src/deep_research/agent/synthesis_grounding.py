"""Single source of truth for the synthesizer cite-vs-verify configuration.

Citations are ALWAYS produced via a grounded synthesis mode (the cheap
"grounding-only" floor). The per-run "verify sources" choice controls ONLY the
expensive per-claim NLI verification overlay — NOT whether citations exist:

    verify on  -> grounding_mode="reclaim"        (cite + NLI verify + disposition)
    verify off -> grounding_mode="classical_lite" (cite; skip NLI/correction/numeric)

Both grounded modes use the strict-cite prompt and parse ``[N]`` markers. The
framework synthesizer selects the prompt from the node-level ``grounding_mode``
(``agents/grounding.py:resolve_grounding_mode``), and the citation pipeline
independently gates its expensive stages on the ``output_schema`` enable flags
(``citation/pipeline.py``). They MUST agree, so this helper stamps both together.

Used by:
  * ``adapters/config_translator._build_synthesizer`` (built-in deep-research path)
  * ``framework_orchestrator._apply_runtime_overlays_to_workflow`` (custom-agent
    full override: the chat "verify sources" toggle re-stamps every synthesizer).

Kept import-free of the rest of the app so both callers can import it without
risking a cycle.
"""

from __future__ import annotations

from typing import Any

__all__ = ["apply_grounding_to_synth_config"]


def apply_grounding_to_synth_config(
    node_config: dict[str, Any], *, full_verify: bool
) -> None:
    """Stamp ``grounding_mode`` + the citation-pipeline enable flags consistently.

    Mutates *node_config* in place: sets the node-level ``grounding_mode`` and the
    verification flags inside ``node_config["output_schema"]``, PRESERVING any other
    ``output_schema`` keys the caller already set (e.g. ``max_tokens``,
    ``target_word_count``, ``claim_disposition``).

    Args:
        node_config: a synthesizer node config dict (the caller is responsible for
            only calling this on synthesizer nodes).
        full_verify: ``True`` -> ``reclaim`` (run the NLI/correction/numeric overlay);
            ``False`` -> ``classical_lite`` (cite-only; skip the expensive stages).
    """
    schema = node_config.get("output_schema")
    if not isinstance(schema, dict):
        schema = {}
    # ``interleaved`` is the only generation strategy; preserve an author's value.
    schema.setdefault("synthesis_mode", "interleaved")
    schema["enable_citation_verification"] = True

    if full_verify:
        node_config["grounding_mode"] = "reclaim"
        schema["enable_isolated_verification"] = True
        # Clear any stale disables so correction/numeric fall back to the
        # framework defaults (both True). Important when re-stamping a node that
        # was previously classical_lite.
        schema.pop("enable_citation_correction", None)
        schema.pop("enable_numeric_qa_verification", None)
    else:
        node_config["grounding_mode"] = "classical_lite"
        # Cheap grounding-only: generate + link + render citations, then skip the
        # expensive verification overlay (Stage 4 NLI / 5 correction / 6 numeric)
        # and Stage 8 disposition. Claims persist as resolvable-but-unverified.
        schema["enable_isolated_verification"] = False
        schema["enable_citation_correction"] = False
        schema["enable_numeric_qa_verification"] = False

    node_config["output_schema"] = schema
