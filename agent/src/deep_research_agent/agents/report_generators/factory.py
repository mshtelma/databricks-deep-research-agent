"""
Report Generator Factory: Intelligent strategy selection with fallback chain.

Implements factory pattern for selecting optimal report generation strategy
based on available data and configuration preferences.

Strategy Tier System:
- Tier 1 (Hybrid): Best quality, requires calculation data
- Tier 2 (Section-by-Section): Production workhorse, requires dynamic sections
- Tier 3 (Template): Basic fallback, always available

Selection Algorithm:
1. Check user preference (config.generation_mode)
2. Test if preferred strategy can handle request
3. If not, cascade down fallback chain
4. Return selection result with transparency

Example:
    >>> factory = ReportGeneratorFactory(llm, config, utilities)
    >>> selection = factory.select_strategy(request)
    >>> print(f"Selected: {selection.strategy_name} ({selection.reason})")
    >>> generator = factory.create_generator(selection.strategy_name)
    >>> result = await generator.generate(request, config)
"""

from typing import Dict, Any, Optional, List, Type
import time
from datetime import datetime

from ...core import get_logger
from .types import (
    ReportGenerationRequest,
    ReporterConfig,
    GenerationMode,
    StrategySelectionResult,
    GenerationError,
)
from .base import BaseReportGenerator
from .utilities import ReporterUtilities

# Import strategies (lazy import in methods to avoid circular dependencies)

logger = get_logger(__name__)


class ReportGeneratorFactory:
    """
    Factory for creating and selecting report generation strategies.

    Responsibilities:
    - Register available strategies
    - Select optimal strategy based on request capabilities
    - Provide fallback chain for graceful degradation
    - Track strategy performance metrics

    Strategy Selection Logic:
    1. Prefer user's explicit choice (config.generation_mode)
    2. Validate selected strategy can handle request
    3. If not, cascade through fallback chain
    4. Always return a viable strategy (Template is final fallback)

    Fallback Chain:
        Hybrid (Tier 1) → Section-by-Section (Tier 2) → Template (Tier 3)
    """

    def __init__(
        self,
        llm: Any,
        config: ReporterConfig,
        utilities: Optional[ReporterUtilities] = None,
        **kwargs
    ):
        """
        Initialize factory with strategies.

        Args:
            llm: Language model instance
            config: Typed reporter configuration
            utilities: Shared reporter utilities (created if None)
            **kwargs: Additional args passed to strategies
        """
        self.llm = llm
        self.config = config
        self.utilities = utilities or ReporterUtilities()
        self.kwargs = kwargs

        # Strategy registry: name -> class
        self._strategy_classes: Dict[str, Type[BaseReportGenerator]] = {}

        # Strategy instances cache: name -> instance
        self._strategy_instances: Dict[str, BaseReportGenerator] = {}

        # Performance tracking
        self._selection_history: List[Dict[str, Any]] = []

        # Register all strategies
        self._register_strategies()

        logger.info(
            f"ReportGeneratorFactory initialized: "
            f"{len(self._strategy_classes)} strategies registered"
        )

    def _register_strategies(self) -> None:
        """
        Register all available report generation strategies.

        Strategies are registered in priority order (Tier 1 → Tier 3).
        This order defines the fallback chain.
        """
        # Lazy imports to avoid circular dependencies
        from .hybrid_generator import HybridGenerator
        from .section_by_section_generator import SectionBySectionGenerator
        from .template_generator import TemplateGenerator

        # Register in priority order (fallback chain)
        self._register_strategy("hybrid", HybridGenerator)
        self._register_strategy("section_by_section", SectionBySectionGenerator)
        self._register_strategy("template", TemplateGenerator)

        logger.info(
            f"Registered strategies: {list(self._strategy_classes.keys())}"
        )

    def _register_strategy(
        self,
        name: str,
        strategy_class: Type[BaseReportGenerator]
    ) -> None:
        """
        Register a strategy class.

        Args:
            name: Strategy identifier (must match GenerationMode)
            strategy_class: Strategy class (must extend BaseReportGenerator)
        """
        if not issubclass(strategy_class, BaseReportGenerator):
            raise ValueError(
                f"Strategy {name} must extend BaseReportGenerator, "
                f"got {strategy_class.__name__}"
            )

        self._strategy_classes[name] = strategy_class
        logger.debug(f"Registered strategy: {name} -> {strategy_class.__name__}")

    def create_generator(
        self,
        strategy_name: str,
        use_cache: bool = True
    ) -> BaseReportGenerator:
        """
        Create (or retrieve cached) generator instance.

        Args:
            strategy_name: Name of strategy to create
            use_cache: Whether to use cached instance

        Returns:
            Strategy instance ready for generation

        Raises:
            ValueError: If strategy name unknown
        """
        if strategy_name not in self._strategy_classes:
            available = list(self._strategy_classes.keys())
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. "
                f"Available: {available}"
            )

        # Check cache
        if use_cache and strategy_name in self._strategy_instances:
            logger.debug(f"Returning cached instance: {strategy_name}")
            return self._strategy_instances[strategy_name]

        # Create new instance
        strategy_class = self._strategy_classes[strategy_name]
        instance = strategy_class(
            llm=self.llm,
            config=self.config,
            utilities=self.utilities,
            **self.kwargs
        )

        # Cache instance
        if use_cache:
            self._strategy_instances[strategy_name] = instance

        logger.debug(f"Created new instance: {strategy_name}")
        return instance

    def select_strategy(
        self,
        request: ReportGenerationRequest,
        prefer_mode: Optional[GenerationMode] = None
    ) -> StrategySelectionResult:
        """
        Select optimal strategy for request.

        Selection Algorithm:
        1. Start with preferred mode (config or parameter)
        2. Check if strategy can handle request
        3. If not, try next strategy in fallback chain
        4. Return selection result with transparency

        Args:
            request: Generation request
            prefer_mode: Override config preference (for testing)

        Returns:
            Selection result with strategy name, reason, confidence
        """
        start_time = time.time()

        # Determine preferred mode (handle both enum and string)
        preferred_mode = prefer_mode or self.config.generation_mode

        # Ensure it's an enum
        if isinstance(preferred_mode, str):
            preferred_mode = GenerationMode(preferred_mode)

        logger.info(
            f"Strategy selection started: prefer={preferred_mode.value}, "
            f"has_calc={request.has_calculation_data()}, "
            f"has_sections={request.has_dynamic_sections()}"
        )

        # Define fallback chain based on preferred mode
        if preferred_mode == GenerationMode.HYBRID:
            # Try hybrid first, then section-by-section, then template
            chain = [
                GenerationMode.HYBRID,
                GenerationMode.SECTION_BY_SECTION,
                GenerationMode.TEMPLATE
            ]
        elif preferred_mode == GenerationMode.SECTION_BY_SECTION:
            # Try section-by-section first, then template (skip hybrid)
            chain = [
                GenerationMode.SECTION_BY_SECTION,
                GenerationMode.TEMPLATE
            ]
        else:
            # Template only
            chain = [GenerationMode.TEMPLATE]

        # Try each strategy in chain
        for mode in chain:
            strategy_name = mode.value

            try:
                generator = self.create_generator(strategy_name)
                can_handle, reason = generator.can_handle(request)

                if can_handle:
                    # Success! This strategy can handle the request
                    selection_time_ms = int((time.time() - start_time) * 1000)

                    # Build fallback chain for transparency
                    remaining_fallbacks = [
                        m.value for m in chain[chain.index(mode) + 1:]
                    ]

                    # Determine confidence
                    if mode == preferred_mode:
                        confidence = 1.0  # Got preferred strategy
                    elif mode == GenerationMode.TEMPLATE:
                        confidence = 0.5  # Final fallback (still works)
                    else:
                        confidence = 0.75  # Acceptable fallback

                    # Build selection result
                    result = StrategySelectionResult(
                        strategy_name=strategy_name,
                        generation_mode=mode,
                        reason=(
                            f"Selected {strategy_name}: {reason or 'can handle request'}"
                            if mode == preferred_mode
                            else f"Fallback to {strategy_name}: preferred {preferred_mode.value} unavailable"
                        ),
                        confidence=confidence,
                        fallback_available=len(remaining_fallbacks) > 0,
                        fallback_chain=remaining_fallbacks
                    )

                    # Track selection history
                    self._record_selection(
                        result,
                        request,
                        selection_time_ms,
                        attempted_strategies=[m.value for m in chain[:chain.index(mode) + 1]]
                    )

                    logger.info(
                        f"Strategy selected: {strategy_name} "
                        f"(confidence={confidence:.1%}, time={selection_time_ms}ms)"
                    )

                    return result

                else:
                    # Strategy cannot handle, log reason and continue
                    logger.debug(
                        f"Strategy {strategy_name} cannot handle: {reason}"
                    )

            except Exception as e:
                logger.warning(
                    f"Error testing strategy {strategy_name}: {e}",
                    exc_info=True
                )
                # Continue to next strategy

        # Should never reach here (Template is always available)
        # But provide a safety net
        raise GenerationError(
            "No strategy could handle request (including Template fallback)",
            strategy="factory",
            recoverable=False
        )

    def _record_selection(
        self,
        result: StrategySelectionResult,
        request: ReportGenerationRequest,
        selection_time_ms: int,
        attempted_strategies: List[str]
    ) -> None:
        """
        Record strategy selection for metrics and debugging.

        Args:
            result: Selection result
            request: Original request
            selection_time_ms: Time taken to select
            attempted_strategies: Strategies tested before selection
        """
        # Get config mode as string (handle both enum and string)
        config_mode = self.config.generation_mode
        if isinstance(config_mode, GenerationMode):
            config_mode_str = config_mode.value
        else:
            config_mode_str = str(config_mode)

        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': request.request_id,
            'selected_strategy': result.strategy_name,
            'confidence': result.confidence,
            'selection_time_ms': selection_time_ms,
            'attempted_strategies': attempted_strategies,
            'fallback_used': result.strategy_name != config_mode_str,
            'request_capabilities': {
                'has_observations': len(request.observations) > 0,
                'has_calculation_data': request.has_calculation_data(),
                'has_dynamic_sections': request.has_dynamic_sections(),
                'has_template': request.has_template(),
            }
        }

        self._selection_history.append(record)

        # Keep only last 100 selections (memory management)
        if len(self._selection_history) > 100:
            self._selection_history = self._selection_history[-100:]

    def get_selection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about strategy selection history.

        Returns:
            Dict with selection stats (strategy distribution, fallback rate, etc.)
        """
        if not self._selection_history:
            return {
                'total_selections': 0,
                'strategy_distribution': {},
                'fallback_rate': 0.0,
                'avg_selection_time_ms': 0
            }

        total = len(self._selection_history)

        # Strategy distribution
        strategy_counts = {}
        fallback_count = 0
        total_time = 0

        for record in self._selection_history:
            strategy = record['selected_strategy']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

            if record['fallback_used']:
                fallback_count += 1

            total_time += record['selection_time_ms']

        return {
            'total_selections': total,
            'strategy_distribution': {
                k: v / total for k, v in strategy_counts.items()
            },
            'fallback_rate': fallback_count / total if total > 0 else 0.0,
            'avg_selection_time_ms': total_time / total if total > 0 else 0,
            'strategy_counts': strategy_counts
        }

    def list_strategies(self) -> List[str]:
        """
        List all registered strategy names.

        Returns:
            List of strategy names in priority order
        """
        return list(self._strategy_classes.keys())

    def get_strategy_info(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get information about a specific strategy.

        Args:
            strategy_name: Name of strategy

        Returns:
            Dict with strategy metadata

        Raises:
            ValueError: If strategy unknown
        """
        if strategy_name not in self._strategy_classes:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        generator = self.create_generator(strategy_name, use_cache=False)

        return {
            'name': generator.name,
            'generation_mode': generator.generation_mode.value,
            'quality_level': generator.quality_level.value,
            'class_name': generator.__class__.__name__,
            'docstring': generator.__class__.__doc__
        }
