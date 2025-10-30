"""
Report Generation Strategies.

This package provides a modular, extensible system for generating research reports
with multiple strategies optimized for different data availability scenarios.

Strategy Tiers:
- Tier 1 (Hybrid): Best quality, calculation-based tables
- Tier 2 (Section-by-Section): Production workhorse, observation-based
- Tier 3 (Template): Basic fallback, always available

Usage:
    >>> from .factory import ReportGeneratorFactory
    >>> from .types import ReporterConfig, ReportGenerationRequest
    >>>
    >>> factory = ReportGeneratorFactory(llm, config, utilities)
    >>> selection = factory.select_strategy(request)
    >>> generator = factory.create_generator(selection.strategy_name)
    >>> result = await generator.generate(request, config)
"""

from .types import (
    GenerationMode,
    ReportQuality,
    TableGenerationMode,
    ReporterConfig,
    ReportGenerationRequest,
    ReportGenerationResult,
    ReportSection,
    StrategySelectionResult,
    GenerationError,
    ValidationError,
    ConfigurationError,
    TimeoutError,
)

from .base import BaseReportGenerator
from .utilities import ReporterUtilities
from .factory import ReportGeneratorFactory

# Strategy imports (for direct use if needed)
from .hybrid_generator import HybridGenerator
from .section_by_section_generator import SectionBySectionGenerator
from .template_generator import TemplateGenerator

__all__ = [
    # Type system
    'GenerationMode',
    'ReportQuality',
    'TableGenerationMode',
    'ReporterConfig',
    'ReportGenerationRequest',
    'ReportGenerationResult',
    'ReportSection',
    'StrategySelectionResult',
    'GenerationError',
    'ValidationError',
    'ConfigurationError',
    'TimeoutError',

    # Core classes
    'BaseReportGenerator',
    'ReporterUtilities',
    'ReportGeneratorFactory',

    # Strategies
    'HybridGenerator',
    'SectionBySectionGenerator',
    'TemplateGenerator',
]
