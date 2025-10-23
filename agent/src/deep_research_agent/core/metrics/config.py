"""
Configuration and enums for the unified planning and calculation system.

This module provides type-safe constants and configuration classes
to eliminate hardcoded values throughout the calculation agent code.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, Optional
import yaml


# String constant enums for type safety
class LLMTier(str, Enum):
    """LLM tier for generation operations."""
    SIMPLE = "simple"
    COMPLEX = "complex"
    ADVANCED = "advanced"


class GeographicLevel(str, Enum):
    """Geographic scope levels."""
    CITY = "city"
    REGION = "region"
    COUNTRY = "country"
    GLOBAL = "global"


class AssumptionSeverity(str, Enum):
    """Severity levels for assumptions and caveats."""
    CRITICAL = "critical"
    IMPORTANT = "important"
    MINOR = "minor"


class SourceType(str, Enum):
    """Metric source type."""
    EXTRACT = "extract"
    CALCULATE = "calculate"
    CONSTANT = "constant"


class ObservationType(str, Enum):
    """Type of observation source."""
    PRIMARY_SOURCE = "primary_source"
    LLM_SYNTHESIS = "llm_synthesis"
    CALCULATION_RESULT = "calculation_result"


# Configuration dataclass
@dataclass
class UnifiedPlanningConfig:
    """Configuration for unified planning and calculation system.

    All magic numbers and thresholds centralized here for easy tuning.
    Can be loaded from YAML or created with defaults, and supports
    per-request overrides without global state.
    """
    # Confidence thresholds
    low_confidence_threshold: float = 0.7
    default_synthesized_confidence: float = 0.5
    high_confidence_threshold: float = 0.9

    # Observation processing
    observation_index_size: int = 30
    content_preview_length: int = 150
    max_fallback_searches: int = 100

    # Validation settings
    enforce_user_formula_requests: bool = True
    warn_on_low_confidence_sources: bool = True
    require_verification_for_synthesized: bool = True

    # Calculation settings
    calculation_tolerance: float = 0.01  # 1% tolerance for float comparisons
    max_calculation_retries: int = 3

    @classmethod
    def from_yaml(cls, config_path: str) -> "UnifiedPlanningConfig":
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            UnifiedPlanningConfig instance with values from YAML

        Example:
            config = UnifiedPlanningConfig.from_yaml('config/unified_planning.yaml')
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict.get('unified_planning', {}))

    @classmethod
    def get_default(cls) -> "UnifiedPlanningConfig":
        """Get default configuration.

        Returns:
            UnifiedPlanningConfig with default values

        Example:
            config = UnifiedPlanningConfig.get_default()
        """
        return cls()

    def merge_with_request_overrides(self, overrides: Dict[str, Any]) -> "UnifiedPlanningConfig":
        """
        Create a new config with request-specific overrides.

        This allows per-request configuration without global state,
        enabling different thresholds for different domains or requests.

        Args:
            overrides: Dict of config values to override for this request

        Returns:
            New UnifiedPlanningConfig instance with overrides applied

        Example:
            base_config = UnifiedPlanningConfig()
            request_overrides = {
                'low_confidence_threshold': 0.6,  # More lenient for this domain
                'calculation_tolerance': 0.02     # 2% tolerance for financial
            }
            request_config = base_config.merge_with_request_overrides(request_overrides)
        """
        # Create a copy of current config as dict
        config_dict = {
            'low_confidence_threshold': self.low_confidence_threshold,
            'default_synthesized_confidence': self.default_synthesized_confidence,
            'high_confidence_threshold': self.high_confidence_threshold,
            'observation_index_size': self.observation_index_size,
            'content_preview_length': self.content_preview_length,
            'max_fallback_searches': self.max_fallback_searches,
            'enforce_user_formula_requests': self.enforce_user_formula_requests,
            'warn_on_low_confidence_sources': self.warn_on_low_confidence_sources,
            'require_verification_for_synthesized': self.require_verification_for_synthesized,
            'calculation_tolerance': self.calculation_tolerance,
            'max_calculation_retries': self.max_calculation_retries,
        }

        # Apply overrides
        config_dict.update(overrides)

        # Return new instance with merged config
        return UnifiedPlanningConfig(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary representation of config

        Example:
            config = UnifiedPlanningConfig()
            config_dict = config.to_dict()
        """
        return {
            'low_confidence_threshold': self.low_confidence_threshold,
            'default_synthesized_confidence': self.default_synthesized_confidence,
            'high_confidence_threshold': self.high_confidence_threshold,
            'observation_index_size': self.observation_index_size,
            'content_preview_length': self.content_preview_length,
            'max_fallback_searches': self.max_fallback_searches,
            'enforce_user_formula_requests': self.enforce_user_formula_requests,
            'warn_on_low_confidence_sources': self.warn_on_low_confidence_sources,
            'require_verification_for_synthesized': self.require_verification_for_synthesized,
            'calculation_tolerance': self.calculation_tolerance,
            'max_calculation_retries': self.max_calculation_retries,
        }
