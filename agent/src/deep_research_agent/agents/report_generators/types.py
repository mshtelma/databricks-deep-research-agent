"""
Type definitions for report generation strategies.

All report generation uses strongly-typed Pydantic models to ensure:
- Type safety at API boundaries
- Validation of inputs/outputs
- Clear contracts between components
- No Dict[str, Any] antipattern
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime
from uuid import uuid4

# Import core types from the system
from ...core.observation_models import StructuredObservation
from ...core.plan_models import DynamicSection
from ...core.constraint_system import QueryConstraints
# Note: Citation imported via TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ...core.types import Citation


class GenerationMode(str, Enum):
    """Report generation modes (strategy types)."""
    HYBRID = "hybrid"
    SECTION_BY_SECTION = "section_by_section"
    TEMPLATE = "template"


class ReportQuality(str, Enum):
    """Report quality levels."""
    HIGH = "high"        # Hybrid with calculations
    MEDIUM = "medium"    # Section-by-section
    LOW = "low"          # Template fallback


class TableGenerationMode(str, Enum):
    """How tables are generated."""
    CALCULATION_BASED = "calculation_based"  # From CalculationAgent results
    PROGRAMMATIC = "programmatic"      # From structured data
    LLM_STRUCTURED = "llm_structured"  # Pydantic models
    LLM_MARKDOWN = "llm_markdown"      # Raw markdown
    NONE = "none"                       # No tables


class ReporterConfig(BaseModel):
    """
    Strongly typed reporter configuration.

    Replaces Dict[str, Any] config with validated Pydantic model.
    All fields have defaults for backward compatibility.
    """

    # Primary mode selection
    generation_mode: GenerationMode = GenerationMode.SECTION_BY_SECTION
    use_structured_pipeline: bool = True
    enable_structured_generation: bool = True

    # Hybrid-specific settings
    hybrid_enabled: bool = True
    hybrid_fallback_enabled: bool = True
    calc_selector_top_k: int = Field(default=60, ge=1, le=200)
    calc_recent_tail: int = Field(default=20, ge=0, le=100)
    max_calc_prompt_chars: int = Field(default=60000, ge=1000, le=200000)
    table_anchor_format: str = "[TABLE: {id}]"
    enable_async_blocks: bool = False
    max_concurrent_blocks: int = Field(default=2, ge=1, le=10)

    # Section-by-section settings
    max_observations_per_section: int = Field(default=30, ge=1, le=100)
    entity_aware_filtering: bool = True
    two_step_process_enabled: bool = True

    # Shared settings
    default_style: str = "default"
    citation_style: str = "APA"
    include_citations: bool = True
    include_grounding_markers: bool = True
    max_retries: int = Field(default=3, ge=1, le=10)
    timeout_seconds: int = Field(default=180, ge=30, le=600)

    # Observability
    enable_metrics: bool = True
    enable_tracing: bool = True
    enable_performance_monitoring: bool = True

    class Config:
        extra = "forbid"  # Catch typos in config
        use_enum_values = True

    @validator('table_anchor_format')
    def validate_anchor_format(cls, v):
        """Ensure anchor format has {id} placeholder."""
        if '{id}' not in v:
            raise ValueError("table_anchor_format must contain '{id}' placeholder")
        return v


class ReportGenerationRequest(BaseModel):
    """
    Input to report generation strategies.

    This is the single source of truth for what data is available.
    No more checking hasattr or dict.get() - just use typed fields.
    """

    # Core data (required)
    research_topic: str
    observations: List[Any]  # List[StructuredObservation] - using Any to avoid pydantic v1/v2 conflict

    # Optional enhanced data
    citations: List[Any] = Field(default_factory=list)  # List[Citation] to avoid circular import
    dynamic_sections: Optional[List[DynamicSection]] = None
    calculation_context: Optional[Any] = None  # CalculationContext (avoid circular import)
    unified_plan: Optional[Any] = None  # UnifiedPlan (avoid circular import)

    # Constraints and metadata
    query_constraints: Optional[QueryConstraints] = None
    report_template: Optional[str] = None
    report_style: str = "default"

    # Quality metrics
    factuality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)

    # Tracking
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True  # For CalculationContext, UnifiedPlan

    @validator('observations')
    def validate_observations(cls, v):
        """Ensure we have at least some observations."""
        if not v:
            raise ValueError("Cannot generate report with zero observations")
        return v

    def has_calculation_data(self) -> bool:
        """
        Check if calculation data is available.

        IMPORTANT: Expects calculation_context to be a CalculationContext Pydantic object,
        NOT a dict. If you're getting a TypeError here, ensure StateManager.hydrate_state()
        is called at the perimeter (fixture loading, state deserialization, etc.).

        This enforces "Pydantic Everywhere" architecture - dicts should only exist at
        I/O boundaries, never in internal code.
        """
        # AGGRESSIVE LOGGING for debugging
        from ...core import get_logger
        logger = get_logger(__name__)

        if self.calculation_context is None:
            logger.error("🔥 [CALC DATA CHECK] calculation_context is None -> has_calc=False")
            return False

        # PRAGMATIC: Allow both CalculationContext object OR dict with 'calculations' key
        # (Needed due to schema mismatch between metrics.models.DataPoint and report_generation.models.DataPoint)
        has_calculations_attr = hasattr(self.calculation_context, 'calculations')
        has_calculations_key = isinstance(self.calculation_context, dict) and 'calculations' in self.calculation_context

        logger.error(f"🔥 [CALC DATA CHECK] calculation_context type: {type(self.calculation_context).__name__}")
        logger.error(f"🔥 [CALC DATA CHECK] has_calculations_attr: {has_calculations_attr}")
        logger.error(f"🔥 [CALC DATA CHECK] has_calculations_key: {has_calculations_key}")

        if not (has_calculations_attr or has_calculations_key):
            raise TypeError(
                f"calculation_context must be CalculationContext object or dict with 'calculations' key, "
                f"got {type(self.calculation_context).__name__}. "
                f"Ensure StateManager.hydrate_state() is called at perimeter to convert "
                f"dicts to Pydantic models. See core/multi_agent_state.py:hydrate_state()"
            )

        # Access calculations safely (works for both object and dict)
        if has_calculations_attr:
            calculations = self.calculation_context.calculations or []
        elif has_calculations_key:
            calculations = self.calculation_context.get('calculations', [])
        else:
            calculations = []

        # CRITICAL FIX: Also check extracted_data! The calculation agent puts data points in extracted_data
        # even when there are no formulas. Hybrid mode should work with EITHER calculated formulas OR
        # extracted metric values - both provide structured data for table generation.
        if hasattr(self.calculation_context, 'extracted_data'):
            extracted_data = self.calculation_context.extracted_data or []
        elif isinstance(self.calculation_context, dict):
            extracted_data = self.calculation_context.get('extracted_data', [])
        else:
            extracted_data = []

        # Legacy field name support
        if hasattr(self.calculation_context, 'data_points'):
            data_points = self.calculation_context.data_points or []
        elif isinstance(self.calculation_context, dict):
            data_points = self.calculation_context.get('data_points', [])
        else:
            data_points = []

        logger.error(f"🔥 [CALC DATA CHECK] calculations count: {len(calculations)}")
        logger.error(f"🔥 [CALC DATA CHECK] extracted_data count: {len(extracted_data)}")
        logger.error(f"🔥 [CALC DATA CHECK] data_points count: {len(data_points)}")

        # Accept EITHER formulas (calculations) OR extracted metric values (extracted_data/data_points)
        has_formulas = len(calculations) > 0
        has_data = len(extracted_data) > 0 or len(data_points) > 0

        logger.error(f"🔥 [CALC DATA CHECK] has_formulas: {has_formulas}, has_data: {has_data}")
        logger.error(f"🔥 [CALC DATA CHECK] RESULT: has_calc={has_formulas or has_data}")

        return has_formulas or has_data

    def has_dynamic_sections(self) -> bool:
        """Check if dynamic sections are available."""
        return self.dynamic_sections is not None and len(self.dynamic_sections) > 0

    def has_template(self) -> bool:
        """Check if explicit template is provided."""
        return self.report_template is not None and len(self.report_template.strip()) > 0


class ReportSection(BaseModel):
    """A section of the generated report."""

    title: str
    content: str
    level: int = Field(default=2, ge=1, le=6)  # Heading level (## = 2)

    # Metadata
    tables_count: int = Field(default=0, ge=0)
    citations_used: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    observations_used: int = Field(default=0, ge=0)

    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"  # Sections can have extra metadata

    def to_markdown(self) -> str:
        """Convert section to markdown with heading."""
        heading = "#" * self.level
        return f"{heading} {self.title}\n\n{self.content}"


class ReportGenerationResult(BaseModel):
    """
    Output from report generation strategies.

    All strategies return this typed result instead of raw strings.
    """

    # Core output
    final_report: str
    sections: List[ReportSection] = Field(default_factory=list)

    # Metadata about generation
    generation_mode: GenerationMode
    table_mode: TableGenerationMode
    quality: ReportQuality

    # Statistics
    total_sections: int = Field(ge=0)
    total_tables: int = Field(default=0, ge=0)
    total_citations: int = Field(default=0, ge=0)
    observations_used: int = Field(ge=0)
    calculations_used: int = Field(default=0, ge=0)

    # Performance metrics
    generation_time_ms: int = Field(ge=0)
    llm_calls: int = Field(default=0, ge=0)
    tokens_used: int = Field(default=0, ge=0)

    # Tracking
    request_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Error recovery
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

    class Config:
        extra = "forbid"

    @validator('total_sections', always=True)
    def validate_sections_match(cls, v, values):
        """Ensure total_sections matches len(sections)."""
        if 'sections' in values:
            actual_count = len(values['sections'])
            if v != actual_count:
                # Auto-correct to match actual sections
                return actual_count
        return v

    def to_dict_for_state(self) -> Dict[str, Any]:
        """
        Convert to dict for LangGraph state update.

        Only includes serializable fields, excludes complex objects.
        """
        return {
            'generation_mode': self.generation_mode.value,
            'table_mode': self.table_mode.value,
            'quality': self.quality.value,
            'total_sections': self.total_sections,
            'total_tables': self.total_tables,
            'total_citations': self.total_citations,
            'observations_used': self.observations_used,
            'calculations_used': self.calculations_used,
            'generation_time_ms': self.generation_time_ms,
            'llm_calls': self.llm_calls,
            'tokens_used': self.tokens_used,
            'fallback_used': self.fallback_used,
            'fallback_reason': self.fallback_reason,
            'warnings': self.warnings
        }


class GenerationMetrics(BaseModel):
    """
    Metrics for observability and monitoring.

    Captured during generation for performance analysis.
    """

    # Strategy identification
    strategy_name: str
    request_id: str

    # Timing
    start_time: float
    end_time: float
    duration_ms: float

    # Counts
    sections_generated: int = Field(ge=0)
    tables_generated: int = Field(default=0, ge=0)
    llm_invocations: int = Field(default=0, ge=0)
    observations_processed: int = Field(default=0, ge=0)

    # Quality
    avg_section_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    table_validation_failures: int = Field(default=0, ge=0)

    # Resources
    memory_mb_used: float = Field(default=0.0, ge=0.0)
    tokens_consumed: int = Field(default=0, ge=0)

    # Errors and recovery
    errors_encountered: int = Field(default=0, ge=0)
    errors_recovered: int = Field(default=0, ge=0)
    fallback_triggered: bool = False

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


class StrategySelectionResult(BaseModel):
    """
    Result of strategy selection by factory.

    Provides transparency into why a strategy was selected.
    """

    strategy_name: str
    generation_mode: GenerationMode
    reason: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    # Fallback information
    fallback_available: bool
    fallback_chain: List[str] = Field(default_factory=list)

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        extra = "forbid"


# Exceptions

class GenerationError(Exception):
    """
    Base exception for report generation errors.

    Provides structured error information for recovery.
    """

    def __init__(
        self,
        message: str,
        strategy: str,
        recoverable: bool = False,
        fallback_suggestion: Optional[GenerationMode] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.strategy = strategy
        self.recoverable = recoverable
        self.fallback_suggestion = fallback_suggestion
        self.original_error = original_error
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for logging/debugging."""
        return {
            'message': str(self),
            'strategy': self.strategy,
            'recoverable': self.recoverable,
            'fallback_suggestion': self.fallback_suggestion.value if self.fallback_suggestion else None,
            'original_error': str(self.original_error) if self.original_error else None,
            'timestamp': self.timestamp.isoformat()
        }


class ValidationError(GenerationError):
    """Error during output validation."""
    pass


class ConfigurationError(GenerationError):
    """Error in configuration."""
    pass


class TimeoutError(GenerationError):
    """Generation timeout."""
    pass
