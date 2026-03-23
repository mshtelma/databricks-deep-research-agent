"""Generic query context extraction.

This module provides the default extraction logic for extracting
structured context from user queries. Plugins can override this
via the ExtractionConfigProvider protocol.

The framework uses this module when no plugin provides extraction
configuration, ensuring a minimal but functional extraction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from deep_research.plugins.base import ExtractionConfig


class GenericQueryExtraction(BaseModel):
    """Truly generic extraction - no domain-specific fields.

    Framework default. Plugins should provide their own model
    via ExtractionConfigProvider for domain-specific extraction.
    """

    primary_entity: str | None = Field(
        default=None,
        description="Main entity/subject being researched",
    )
    secondary_entities: list[str] = Field(
        default_factory=list,
        description="Other entities mentioned",
    )
    query_intent: str | None = Field(
        default=None,
        description="What the user wants to accomplish",
    )


# Generic prompt with NO domain examples
GENERIC_EXTRACTION_PROMPT = """Extract key information from this query.

- primary_entity: The main subject/entity to research
- secondary_entities: Other mentioned entities
- query_intent: What the user wants to learn or accomplish

Be precise. Extract only what is explicitly mentioned. Do not infer or guess."""


def get_default_extraction_config() -> ExtractionConfig:
    """Get default extraction config with no domain bias.

    Returns:
        ExtractionConfig with generic prompt and model.
    """
    from deep_research.plugins.base import ExtractionConfig

    return ExtractionConfig(
        system_prompt=GENERIC_EXTRACTION_PROMPT,
        extraction_model=GenericQueryExtraction,
        field_mapping={
            "primary_entity": "company_name",
            "secondary_entities": "competitors",
            "query_intent": "research_intent",
        },
    )


async def extract_query_context(
    query: str,
    llm: Any,
    plugin_manager: Any | None = None,
) -> dict[str, Any]:
    """Extract structured context from query using LLM.

    Framework function - truly generic. Domain customization
    comes from plugins via ExtractionConfigProvider protocol.

    Args:
        query: User's research query.
        llm: LLM client for extraction.
        plugin_manager: Optional - to get plugin extraction config.

    Returns:
        Dict with extracted context for plugin_data.
    """
    import logging

    from deep_research.plugins.base import ExtractionConfigProvider
    from deep_research.services.llm.types import ModelTier

    logger = logging.getLogger(__name__)

    # 1. Get extraction config from plugin (if available)
    config = None
    if plugin_manager:
        for plugin in plugin_manager._plugins:
            if isinstance(plugin, ExtractionConfigProvider):
                config = plugin.get_extraction_config()
                if config:
                    logger.info(
                        "EXTRACTION_CONFIG_FROM_PLUGIN plugin=%s model=%s",
                        getattr(plugin, "name", type(plugin).__name__),
                        config.extraction_model.__name__,
                    )
                    break

    # 2. Fall back to generic config
    if not config:
        config = get_default_extraction_config()
        logger.info("EXTRACTION_CONFIG_DEFAULT model=%s", config.extraction_model.__name__)

    # 3. Execute extraction with LLM (generic logic, plugin config)
    try:
        response = await llm.complete(
            messages=[
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": query},
            ],
            tier=ModelTier.BULK_ANALYSIS,
            structured_output=config.extraction_model,
        )

        if response.structured:
            extraction = response.structured
        else:
            extraction = config.extraction_model.model_validate_json(response.content)

        # 4. Map extraction fields to plugin_data using config.field_mapping
        result: dict[str, Any] = {}
        extraction_dict = extraction.model_dump()

        for source_field, target_field in config.field_mapping.items():
            if source_field in extraction_dict and extraction_dict[source_field]:
                result[target_field] = extraction_dict[source_field]

        # Also include raw extraction fields for flexibility
        result.update(extraction_dict)

        logger.info(
            "EXTRACTION_RESULT keys=%s primary_entity=%s",
            list(result.keys()),
            result.get("primary_entity"),
        )

        return result

    except Exception as e:
        logger.warning("LLM_EXTRACTION_FAILED error=%s", str(e)[:100])
        return {}
