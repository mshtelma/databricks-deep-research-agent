"""Semantic matching for observation lookup when IDs don't match.

This module provides semantic similarity-based observation matching
as a fallback when exact ID matching fails.
"""

import re
from typing import Dict, List, Optional, Tuple
import numpy as np

from .. import get_logger

logger = get_logger(__name__)


def generate_semantic_query(
    data_id: str,
    extraction_path: str,
    entity: Optional[str] = None
) -> str:
    """Generate semantic search query from extraction path.

    Examples:
        "Spain tax rate" → "Spain Spanish income tax rate percentage 2025"
        "France childcare benefit" → "France French childcare benefit subsidy amount euros"
        "UK salary average" → "United Kingdom UK average salary wage income pounds"

    Args:
        data_id: The data source ID (for context)
        extraction_path: The LLM-provided extraction path
        entity: Optional entity for more specific query

    Returns:
        Enhanced semantic query
    """
    # Parse extraction path components
    components = extraction_path.lower().split()

    # Enhance with synonyms and related terms
    enhancements = {
        "tax": ["tax", "taxation", "income tax", "tax rate", "percentage"],
        "benefit": ["benefit", "subsidy", "allowance", "support", "aid", "payment"],
        "salary": ["salary", "wage", "income", "earnings", "compensation", "pay"],
        "cost": ["cost", "price", "expense", "fee", "amount", "charge"],
        "childcare": ["childcare", "daycare", "nursery", "kindergarten", "child care"],
        "rent": ["rent", "rental", "lease", "housing cost", "accommodation"],
        "net": ["net", "after-tax", "take-home", "post-tax"],
        "gross": ["gross", "before-tax", "pre-tax", "total"],
        "rate": ["rate", "percentage", "percent", "%"],
        "average": ["average", "mean", "typical", "median"],
    }

    # Country/region expansions
    country_expansions = {
        "spain": ["Spain", "Spanish", "ES"],
        "france": ["France", "French", "FR"],
        "germany": ["Germany", "German", "DE"],
        "uk": ["UK", "United Kingdom", "British", "GB"],
        "italy": ["Italy", "Italian", "IT"],
        "netherlands": ["Netherlands", "Dutch", "NL"],
    }

    # Build enhanced query
    query_parts = []

    # Add entity if provided
    if entity:
        query_parts.append(entity)
        # Check if entity is a country and add expansions
        entity_lower = entity.lower()
        for country, expansions in country_expansions.items():
            if country in entity_lower:
                query_parts.extend(expansions[:2])  # Add first 2 expansions
                break

    # Add enhanced terms
    for component in components:
        component_lower = component.lower()
        # Check for country names
        for country, expansions in country_expansions.items():
            if country in component_lower:
                query_parts.extend(expansions[:2])
                break
        # Check for term enhancements
        if component_lower in enhancements:
            query_parts.extend(enhancements[component_lower][:3])  # Top 3 synonyms
        else:
            query_parts.append(component)

    # Add temporal context
    query_parts.append("2025")
    query_parts.append("2024")  # Also check recent year

    # Add value indicators
    query_parts.extend(["amount", "value", "number"])

    # Remove duplicates while preserving order
    seen = set()
    unique_parts = []
    for part in query_parts:
        part_lower = part.lower()
        if part_lower not in seen:
            seen.add(part_lower)
            unique_parts.append(part)

    query = " ".join(unique_parts)

    logger.debug(
        f"[SEMANTIC QUERY] Generated query for '{data_id}': '{query}' "
        f"from extraction_path: '{extraction_path}'"
    )

    return query


def extract_entity_from_path(extraction_path: str) -> Optional[str]:
    """Extract entity (country/region) from extraction path.

    Args:
        extraction_path: The extraction path string

    Returns:
        Extracted entity or None
    """
    # Common country/region patterns
    countries = [
        "Spain", "France", "Germany", "UK", "United Kingdom",
        "Italy", "Netherlands", "Belgium", "Switzerland",
        "Austria", "Portugal", "Ireland", "Sweden",
        "Norway", "Denmark", "Finland"
    ]

    path_lower = extraction_path.lower()
    for country in countries:
        if country.lower() in path_lower:
            return country

    # Check for city names
    cities = [
        "London", "Paris", "Berlin", "Madrid", "Rome",
        "Amsterdam", "Brussels", "Vienna", "Dublin",
        "Stockholm", "Copenhagen", "Helsinki"
    ]

    for city in cities:
        if city.lower() in path_lower:
            return city

    return None


def simple_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity without embeddings.

    This is a fallback when embedding models are not available.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score between 0 and 1
    """
    # Normalize texts
    text1_lower = text1.lower()
    text2_lower = text2.lower()

    # Split into words
    words1 = set(re.findall(r'\w+', text1_lower))
    words2 = set(re.findall(r'\w+', text2_lower))

    # Calculate Jaccard similarity
    if not words1 or not words2:
        return 0.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    jaccard = len(intersection) / len(union) if union else 0.0

    # Boost score if key terms match
    key_terms = ["tax", "rate", "salary", "income", "benefit", "cost", "rent"]
    key_matches = sum(1 for term in key_terms if term in text1_lower and term in text2_lower)

    # Weight Jaccard with key term matches
    score = jaccard * 0.7 + (key_matches / len(key_terms)) * 0.3

    return min(score, 1.0)


def find_best_observation_match(
    query: str,
    observations: List[Dict],
    threshold: float = 0.3
) -> Optional[Dict]:
    """Find best matching observation using simple text similarity.

    This is a lightweight alternative to embedding-based matching.

    Args:
        query: Semantic query string
        observations: List of observations to search
        threshold: Minimum similarity score (0-1)

    Returns:
        Best matching observation or None
    """
    scores = []

    for obs in observations:
        # Combine content and metadata for matching
        obs_text = f"{obs.get('content', '')} {obs.get('step_id', '')}"

        # Calculate similarity
        similarity = simple_text_similarity(query, obs_text)
        scores.append((similarity, obs))

    # Sort by score
    scores.sort(reverse=True, key=lambda x: x[0])

    # Return best match if above threshold
    if scores and scores[0][0] >= threshold:
        best_score, best_obs = scores[0]
        logger.info(
            f"[SEMANTIC MATCH] Found observation for query '{query[:50]}...' "
            f"with similarity {best_score:.3f} "
            f"(obs_id: {best_obs.get('step_id', 'unknown')})"
        )
        return best_obs

    # Log why no match was found
    if scores:
        logger.warning(
            f"[SEMANTIC MATCH] No observation found for query '{query[:50]}...' "
            f"(best score: {scores[0][0]:.3f} < threshold {threshold})"
        )
    else:
        logger.warning(
            f"[SEMANTIC MATCH] No observations to search for query '{query[:50]}...'"
        )

    return None


async def find_observation_semantic(
    data_id: str,
    extraction_path: str,
    observations: List[Dict],
    entity: Optional[str] = None,
    threshold: float = 0.3
) -> Optional[Dict]:
    """Main entry point for semantic observation matching.

    This function orchestrates the semantic matching process.

    Args:
        data_id: The data source ID
        extraction_path: The extraction path from the plan
        observations: List of observations to search
        entity: Optional entity for more specific query
        threshold: Minimum similarity score

    Returns:
        Best matching observation or None
    """
    # Generate enhanced semantic query
    query = generate_semantic_query(data_id, extraction_path, entity)

    # Find best match using simple similarity
    # (In production, this would use embeddings)
    match = find_best_observation_match(query, observations, threshold)

    if match:
        logger.info(
            f"[SEMANTIC MATCH] ✓ Found observation for data_id='{data_id}' "
            f"using semantic query (obs_id: {match.get('step_id')})"
        )
    else:
        logger.warning(
            f"[SEMANTIC MATCH] ✗ No match for data_id='{data_id}' "
            f"with query: '{query[:100]}...'"
        )

    return match