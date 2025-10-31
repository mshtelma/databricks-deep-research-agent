"""
Model Selector - 4-Tier Model System with Intelligent Endpoint Rotation

Implements tier-based model selection with configurable rotation strategies:
- Round-robin: Even distribution across endpoints
- LRU (Least Recently Used): Maximum spacing between uses
- Random: Natural load spreading

Handles automatic fallback on 429 errors.
"""

import time
import random
import threading
import logging
import asyncio
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from databricks_langchain import ChatDatabricks

from .rate_limiter import get_global_rate_limiter

logger = logging.getLogger(__name__)


# Model-specific max_tokens limits based on Databricks Foundation Model API documentation
# Source: https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/limits
MODEL_MAX_TOKENS: Dict[str, int] = {
    # GPT OSS models
    "databricks-gpt-oss-120b": 25000,
    "databricks-gpt-oss-20b": 25000,

    # Gemma models
    "databricks-gemma-3-12b": 8192,

    # Llama models
    "databricks-llama-4-maverick": 8192,
    "databricks-llama-3-1-405b": 4096,
    "databricks-llama-3-1-70b": 8192,
    "databricks-llama-3-1-8b": 8192,

    # Claude models (high limits, but default is 1000 if not specified)
    "databricks-claude-3-7-sonnet": 8192,
    "databricks-claude-sonnet-4": 8192,

    # Default fallback for unknown models
    "default": 4000,
}


# Models that DO NOT support response_format parameter (JSON mode)
# These models will use prompt-based JSON generation instead
# Source: Databricks API testing - Claude models don't support response_format
MODELS_WITHOUT_JSON_MODE_SUPPORT: set = {
    "databricks-claude-3-7-sonnet",
    "databricks-claude-sonnet-4",
    "databricks-claude-3-5-sonnet",
    "databricks-claude-3-opus",
    "databricks-claude-3-sonnet",
    "databricks-claude-3-haiku",
}


@dataclass
class EndpointHealth:
    """Track health metrics for an endpoint"""
    endpoint: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    last_429_time: Optional[float] = None
    recent_latencies: List[float] = field(default_factory=list)

    def record_success(self, latency: float):
        """Record successful request"""
        self.total_requests += 1
        self.successful_requests += 1
        self.recent_latencies.append(latency)
        if len(self.recent_latencies) > 10:
            self.recent_latencies.pop(0)

    def record_failure(self, is_429: bool = False):
        """Record failed request"""
        self.total_requests += 1
        self.failed_requests += 1
        if is_429:
            self.last_429_time = time.time()

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0-1.0)"""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    @property
    def avg_latency(self) -> float:
        """Calculate average latency"""
        if not self.recent_latencies:
            return 0.0
        return sum(self.recent_latencies) / len(self.recent_latencies)


class EndpointRotator(ABC):
    """Base class for endpoint rotation strategies"""

    def __init__(self, endpoints: List[str]):
        self.endpoints = endpoints
        self.lock = threading.Lock()

    @abstractmethod
    def next_endpoint(self) -> str:
        """Get next endpoint to use"""
        pass


class RoundRobinRotator(EndpointRotator):
    """Round-robin rotation: cycles through endpoints sequentially"""

    def __init__(self, endpoints: List[str]):
        super().__init__(endpoints)
        self.current_index = 0

    def next_endpoint(self) -> str:
        with self.lock:
            endpoint = self.endpoints[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.endpoints)
            return endpoint


class LRURotator(EndpointRotator):
    """
    LRU rotation: selects endpoint that hasn't been used in longest time.
    Best for rate-limited endpoints as it maximizes cooldown time.
    """

    def __init__(self, endpoints: List[str]):
        super().__init__(endpoints)
        self.last_used = {ep: 0.0 for ep in endpoints}

    def next_endpoint(self) -> str:
        with self.lock:
            # Find endpoint with oldest timestamp
            endpoint = min(self.last_used.items(), key=lambda x: x[1])[0]
            self.last_used[endpoint] = time.time()
            return endpoint


class RandomRotator(EndpointRotator):
    """Random rotation: natural load spreading without coordination"""

    def next_endpoint(self) -> str:
        return random.choice(self.endpoints)


class PriorityRotator(EndpointRotator):
    """
    Priority-based endpoint selection: always prefer first endpoint.

    Strategy:
    - Endpoint order in config represents priority (first = highest throughput)
    - Always try endpoints in sequential order
    - Skip endpoints currently in cooldown
    - Return to first endpoint once cooldowns expire

    This rotator respects cooldowns by checking the rate limiter before
    returning an endpoint.
    """

    def __init__(self, endpoints: List[str], rate_limiter=None):
        super().__init__(endpoints)
        self.rate_limiter = rate_limiter
        logger.info(
            f"PriorityRotator initialized: {len(endpoints)} endpoints "
            f"(priority: {' > '.join(endpoints)})"
        )

    def next_endpoint(self) -> str:
        """
        Return highest-priority endpoint not in cooldown.

        Returns:
            First available endpoint, or first endpoint if all in cooldown
        """
        # Try each endpoint in priority order
        for endpoint in self.endpoints:
            if self.rate_limiter and self.rate_limiter.is_in_cooldown(endpoint):
                logger.debug(f"Skipping {endpoint} (in cooldown)")
                continue
            logger.debug(f"Selected {endpoint} (priority-based)")
            return endpoint

        # All endpoints in cooldown - return first anyway
        # (invoke_with_fallback will wait for cooldown)
        logger.warning(
            f"All {len(self.endpoints)} endpoints in cooldown, "
            f"returning highest priority: {self.endpoints[0]}"
        )
        return self.endpoints[0]


class ModelSelector:
    """
    4-Tier model selector with intelligent endpoint rotation.

    Tiers:
    - micro: Pattern matching, entity extraction
    - simple: Query generation, claim extraction
    - analytical: Research synthesis, fact checking
    - complex: Report generation, complex tables
    """

    @staticmethod
    def _estimate_tokens(messages: List, max_tokens: int = 4000) -> int:
        """
        Estimate token count for request.

        Uses heuristic: 1 token ≈ 4 characters for input + max_tokens for output.
        """
        total_chars = 0
        for msg in messages:
            if hasattr(msg, 'content'):
                total_chars += len(str(msg.content))
            elif isinstance(msg, dict) and 'content' in msg:
                total_chars += len(str(msg['content']))

        input_tokens = total_chars // 4
        return input_tokens + max_tokens

    # Operation to tier mapping
    OPERATION_TIER_MAP = {
        # Tier 1: Micro
        "query_classification": "micro",
        "entity_extraction": "micro",
        "pattern_matching": "micro",

        # Tier 2: Simple
        "query_generation": "simple",
        "claim_extraction": "simple",
        "key_term_extraction": "simple",
        "validation": "simple",
        "table_introduction": "simple",
        "brief_analysis": "simple",

        # Tier 3: Analytical
        "step_synthesis": "analytical",
        "planning": "analytical",
        "quality_assessment": "analytical",
        "claim_verification": "analytical",
        "contradiction_detection": "analytical",
        "reflexion": "analytical",

        # Tier 4: Complex
        "section_generation": "complex",
        "table_generation": "complex",
        "final_synthesis": "complex",
        "complex_planning": "complex",
    }

    def __init__(self, config: Dict):
        """
        Initialize model selector.

        Args:
            config: Configuration dict with 'models' key containing tier configs
        """
        self.config = config
        self.models_config = config.get("models", {})
        self.rate_limiter = get_global_rate_limiter()

        # DIAGNOSTIC: Log config structure to debug empty endpoints issue
        logger.info(f"[DIAGNOSTIC] ModelSelector.__init__ called")
        logger.info(f"[DIAGNOSTIC] config keys: {list(config.keys())}")
        logger.info(f"[DIAGNOSTIC] models_config keys: {list(self.models_config.keys())}")
        for tier in ["micro", "simple", "analytical", "complex"]:
            tier_cfg = self.models_config.get(tier, {})
            endpoints = tier_cfg.get("endpoints", [])
            logger.info(f"[DIAGNOSTIC]   Tier '{tier}': {len(endpoints)} endpoints = {endpoints}")

        # Initialize rotators per tier
        self.rotators: Dict[str, EndpointRotator] = {}
        self.endpoint_health: Dict[str, EndpointHealth] = {}

        self._initialize_rotators()
        logger.info(f"ModelSelector initialized with {len(self.rotators)} tiers")

    def _initialize_rotators(self):
        """Initialize endpoint rotators for each tier"""
        for tier_name, tier_config in self.models_config.items():
            endpoints = tier_config.get("endpoints", [])
            if not endpoints:
                logger.warning(f"No endpoints configured for tier: {tier_name}")
                continue

            strategy = tier_config.get("rotation_strategy", "round_robin").lower()

            # Create appropriate rotator
            if strategy == "priority":
                rotator = PriorityRotator(endpoints, self.rate_limiter)
            elif strategy == "lru":
                rotator = LRURotator(endpoints)
            elif strategy == "random":
                rotator = RandomRotator(endpoints)
            else:  # default to round_robin
                rotator = RoundRobinRotator(endpoints)

            self.rotators[tier_name] = rotator
            logger.info(
                f"Tier '{tier_name}': {len(endpoints)} endpoints, "
                f"strategy={strategy}"
            )

            # Initialize health tracking for endpoints
            for endpoint in endpoints:
                if endpoint not in self.endpoint_health:
                    self.endpoint_health[endpoint] = EndpointHealth(endpoint)

    def get_tier_for_operation(self, operation: str) -> str:
        """
        Get model tier for an operation.

        Args:
            operation: Operation name (e.g., "query_generation")

        Returns:
            Tier name (e.g., "simple")
        """
        tier = self.OPERATION_TIER_MAP.get(operation)
        if not tier:
            logger.warning(f"Unknown operation '{operation}', using 'analytical' tier")
            return "analytical"
        return tier

    def get_endpoint_for_tier(self, tier: str) -> Optional[str]:
        """
        Get next endpoint for a tier using configured rotation strategy.

        Args:
            tier: Tier name

        Returns:
            Endpoint name or None if tier not configured
        """
        rotator = self.rotators.get(tier)
        if not rotator:
            logger.error(f"No rotator configured for tier: {tier}")
            return None

        endpoint = rotator.next_endpoint()

        # Check if endpoint is in cooldown
        if self.rate_limiter.is_in_cooldown(endpoint):
            logger.warning(f"Endpoint {endpoint} is in cooldown")
            # Try next endpoint if multiple available
            if len(rotator.endpoints) > 1:
                for _ in range(len(rotator.endpoints) - 1):
                    alt_endpoint = rotator.next_endpoint()
                    if not self.rate_limiter.is_in_cooldown(alt_endpoint):
                        logger.info(f"Using alternative endpoint: {alt_endpoint}")
                        return alt_endpoint
                # All endpoints in cooldown, return original
                logger.warning(f"All endpoints in cooldown for tier {tier}")

        return endpoint

    def get_tier_config(self, tier: str) -> Dict:
        """Get configuration for a tier"""
        return self.models_config.get(tier, {})

    def supports_json_mode(self, endpoint: str) -> bool:
        """
        Check if an endpoint supports JSON mode (response_format parameter).

        Args:
            endpoint: Model endpoint name

        Returns:
            True if model supports response_format, False otherwise
        """
        return endpoint not in MODELS_WITHOUT_JSON_MODE_SUPPORT

    def create_llm(
        self,
        tier: str,
        endpoint: Optional[str] = None,
        **kwargs
    ) -> Optional[ChatDatabricks]:
        """
        Create LLM instance for a tier.

        Args:
            tier: Tier name
            endpoint: Specific endpoint (if None, uses rotation)
            **kwargs: Additional arguments for ChatDatabricks
                     Including 'response_format' for structured output

        Returns:
            ChatDatabricks instance or None
        """
        if endpoint is None:
            endpoint = self.get_endpoint_for_tier(tier)

        if not endpoint:
            return None

        tier_config = self.get_tier_config(tier)

        # Merge tier config with kwargs (kwargs take precedence)
        llm_kwargs = {
            "endpoint": endpoint,
            "temperature": tier_config.get("temperature", 0.7),
            "max_tokens": tier_config.get("max_tokens", 4000),
        }

        # Add reasoning_effort if specified
        if "reasoning_effort" in tier_config:
            llm_kwargs["reasoning_effort"] = tier_config["reasoning_effort"]

        # Handle response_format for structured output (Databricks pattern)
        # Only add if model supports it
        response_format = kwargs.pop("response_format", None)
        if response_format:
            # Check if this model supports JSON mode
            if self.supports_json_mode(endpoint):
                # Add to extra_params for ChatDatabricks
                extra_params = kwargs.get("extra_params", {})
                extra_params["response_format"] = response_format
                kwargs["extra_params"] = extra_params
                # ✅ CRITICAL FIX: response_format can be a Pydantic class or dict
                # Don't call .get() on Pydantic classes - use safe type detection
                if isinstance(response_format, dict):
                    format_type = response_format.get('type', 'unknown')
                elif isinstance(response_format, type):
                    format_type = f"pydantic_model:{response_format.__name__}"
                else:
                    format_type = f"object:{type(response_format).__name__}"
                logger.info(f"🔧 Structured output enabled: {format_type}")
            else:
                # Model doesn't support response_format - log warning and skip
                logger.warning(
                    f"⚠️  Model {endpoint} does not support response_format parameter. "
                    f"Falling back to prompt-based JSON generation. "
                    f"The LLM will be instructed via prompt to return JSON."
                )

        # Override with any provided kwargs
        llm_kwargs.update(kwargs)

        # Validate and cap max_tokens at model's limit
        requested_max_tokens = llm_kwargs.get("max_tokens", 4000)
        model_limit = MODEL_MAX_TOKENS.get(endpoint, MODEL_MAX_TOKENS["default"])

        if requested_max_tokens > model_limit:
            logger.warning(
                f"⚠️  max_tokens ({requested_max_tokens}) exceeds {endpoint} limit ({model_limit}). "
                f"Capping at {model_limit} to prevent API errors."
            )
            llm_kwargs["max_tokens"] = model_limit

        logger.debug(f"Creating LLM for tier={tier}, endpoint={endpoint}")
        return ChatDatabricks(**llm_kwargs)

    def _is_tier_degradation_allowed(self, allow_param: bool) -> bool:
        """
        Check if tier degradation is allowed based on BOTH parameter and config.

        Tier degradation requires:
        1. Parameter must be True (False forces disable for recursion prevention)
        2. Config must enable it (global user setting)

        This implements defense-in-depth: even if RateLimitedChatModel passes True,
        we still check config. Protects against:
        - Direct calls to ModelSelector.invoke_with_smart_fallback
        - Future code changes that might bypass config
        - Configuration mismatches across components

        Args:
            allow_param: Parameter value from caller

        Returns:
            True only if both parameter AND config allow degradation
        """
        # Check 1: Parameter-based control (recursion prevention)
        if not allow_param:
            logger.debug("Tier degradation disabled by parameter (recursion prevention)")
            return False

        # Check 2: Config-based control (user setting)
        tier_fallback_config = self.config.get('tier_fallback', {})
        enable_cross_tier = tier_fallback_config.get('enable_cross_tier_fallback', True)

        if not enable_cross_tier:
            logger.debug("Tier degradation disabled by config (enable_cross_tier_fallback=false)")
            return False

        # Both checks passed
        return True

    def _get_fallback_tier(self, tier: str) -> Optional[str]:
        """
        Get fallback tier based on configuration or defaults.

        Args:
            tier: Current tier name

        Returns:
            Fallback tier name or None
        """
        # Check config for explicit fallback chain
        fallback_config = self.config.get('tier_fallback', {})
        if 'fallback_chain' in fallback_config:
            return fallback_config['fallback_chain'].get(tier)

        # Default fallback chain
        default_chain = {
            'complex': 'analytical',
            'analytical': 'simple',
            'simple': 'micro',
            'micro': None
        }
        return default_chain.get(tier)

    async def invoke_with_smart_fallback(
        self,
        tier: str,
        messages: List,
        operation: Optional[str] = None,
        allow_tier_degradation: bool = True,
        response_format: Optional[Dict] = None,
        **kwargs
    ) -> Any:
        """
        Invoke LLM with intelligent endpoint selection and cross-tier fallback.

        Strategy:
        1. Try ALL available endpoints in primary tier (no waiting)
        2. If all in cooldown, try fallback tier (if allowed)
        3. Only wait if absolutely no endpoints available

        Args:
            tier: Model tier to use
            messages: Messages to send to LLM
            operation: Operation name for logging
            allow_tier_degradation: Allow falling back to lower tier
            response_format: Optional Databricks structured output format (json_schema or json_object)
            **kwargs: Additional arguments for ChatDatabricks

        Returns:
            LLM response

        Raises:
            RuntimeError if all endpoints exhausted
        """
        tier_config = self.get_tier_config(tier)
        endpoints = tier_config.get("endpoints", [])

        if not endpoints:
            raise ValueError(f"No endpoints configured for tier: {tier}")

        # Phase 1: Try all available endpoints (no waiting)
        available = self.rate_limiter.get_available_endpoints(endpoints)
        logger.info(
            f"Tier '{tier}' for operation '{operation or 'unknown'}': "
            f"{len(available)}/{len(endpoints)} endpoints available"
        )

        for idx, endpoint in enumerate(available):
            reservation = None
            try:
                # Estimate tokens for budget tracking
                max_tokens = kwargs.get('max_tokens', tier_config.get('max_tokens', 4000))
                estimated_tokens = self._estimate_tokens(messages, max_tokens)

                # Reserve token budget BEFORE making request
                reservation = await self.rate_limiter.reserve_tokens(
                    endpoint,
                    estimated_tokens,
                    timeout=120.0
                )

                # Forward response_format to create_llm (for structured output)
                if response_format:
                    kwargs['response_format'] = response_format

                llm = self.create_llm(tier, endpoint, **kwargs)
                if not llm:
                    # Release reservation if LLM creation failed
                    if reservation:
                        self.rate_limiter.release_tokens(endpoint, reservation)
                    continue

                start_time = time.time()
                logger.info(f"→ Trying {endpoint} for {operation or 'request'}...")
                response = await llm.ainvoke(messages)

                # Commit actual usage (estimate for now, could be refined later)
                if reservation:
                    self.rate_limiter.commit_tokens(endpoint, reservation, estimated_tokens)

                # Record success
                latency = time.time() - start_time
                if endpoint in self.endpoint_health:
                    self.endpoint_health[endpoint].record_success(latency)

                logger.info(f"✅ Success on {endpoint} ({latency:.2f}s)")
                return response

            except Exception as e:
                # Release reservation on failure
                if reservation:
                    self.rate_limiter.release_tokens(endpoint, reservation)
                error_str = str(e)
                is_429 = "429" in error_str

                # Record failure
                if endpoint in self.endpoint_health:
                    self.endpoint_health[endpoint].record_failure(is_429=is_429)

                if is_429:
                    retry_after = self._extract_retry_after(error_str)
                    self.rate_limiter.report_429(endpoint, retry_after)

                    # Determine next action for logging
                    next_idx = idx + 1
                    if next_idx < len(available):
                        next_endpoint = available[next_idx]
                        logger.warning(
                            f"⚠️  429 Error on {endpoint} (cooldown: {retry_after or 'default'}s). "
                            f"Falling back to {next_endpoint}..."
                        )
                    else:
                        logger.warning(
                            f"⚠️  429 Error on {endpoint} (cooldown: {retry_after or 'default'}s). "
                            f"No more endpoints available in tier '{tier}'."
                        )
                    continue  # Try next endpoint

                # Non-429 error: re-raise immediately
                logger.error(f"❌ Non-rate-limit error on {endpoint}: {e}")
                raise

        # Phase 2: All primary tier endpoints exhausted
        logger.warning(f"All endpoints in tier '{tier}' unavailable or rate-limited")

        # Try fallback tier if allowed (both by parameter AND config)
        if self._is_tier_degradation_allowed(allow_tier_degradation):
            fallback_tier = self._get_fallback_tier(tier)
            if fallback_tier and fallback_tier != tier:
                logger.warning(
                    f"⚠️ TIER DEGRADATION: {tier} → {fallback_tier} for {operation or 'request'}"
                )
                return await self.invoke_with_smart_fallback(
                    fallback_tier,
                    messages,
                    operation,
                    allow_tier_degradation=False,  # Only one level of fallback
                    response_format=response_format,  # Pass through structured output config
                    **kwargs
                )
        else:
            # Degradation disabled - log for observability
            if allow_tier_degradation:  # Only log if would have tried
                logger.info(
                    f"ℹ️  Tier degradation disabled by config for tier '{tier}'. "
                    f"All {len(endpoints)} endpoints exhausted. "
                    f"Will wait for cooldown or fail."
                )

        # Phase 3: Wait for shortest cooldown
        endpoint, wait_time = self.rate_limiter.get_shortest_cooldown(endpoints)
        if endpoint and 0 < wait_time < 60:  # Only wait up to 60s
            logger.warning(
                f"⏳ Waiting {wait_time:.1f}s for {endpoint} cooldown to expire..."
            )
            await asyncio.sleep(wait_time)

            # Try once more after waiting
            reservation = None
            try:
                # Estimate tokens
                max_tokens = kwargs.get('max_tokens', tier_config.get('max_tokens', 4000))
                estimated_tokens = self._estimate_tokens(messages, max_tokens)

                # Reserve tokens
                reservation = await self.rate_limiter.reserve_tokens(
                    endpoint,
                    estimated_tokens,
                    timeout=120.0
                )

                # Forward response_format to create_llm (for structured output) - CRITICAL FIX
                if response_format:
                    kwargs['response_format'] = response_format
                    logger.info(f"📋 Forwarding response_format to cooldown retry: {response_format.get('type', 'unknown')}")

                llm = self.create_llm(tier, endpoint, **kwargs)
                logger.info(f"→ Retrying {endpoint} after cooldown...")
                response = await llm.ainvoke(messages)

                # Commit usage
                if reservation:
                    self.rate_limiter.commit_tokens(endpoint, reservation, estimated_tokens)

                logger.info(f"✅ Success on {endpoint} after waiting ({wait_time:.1f}s cooldown)")
                return response
            except Exception as e:
                # Release on failure
                if reservation:
                    self.rate_limiter.release_tokens(endpoint, reservation)
                logger.error(f"Failed even after waiting for cooldown: {e}")
                raise

        # Phase 4: Give up
        raise RuntimeError(
            f"All endpoints exhausted for tier '{tier}', operation '{operation or 'unknown'}'. "
            f"Available: 0/{len(endpoints)}"
        )

    async def invoke_with_fallback(
        self,
        tier: str,
        messages: List,
        operation: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Invoke LLM with automatic fallback on 429 errors.

        Args:
            tier: Model tier to use
            messages: Messages to send to LLM
            operation: Operation name for logging
            **kwargs: Additional arguments for ChatDatabricks

        Returns:
            LLM response

        Raises:
            Exception if all endpoints fail
        """
        tier_config = self.get_tier_config(tier)
        endpoints = tier_config.get("endpoints", [])
        fallback_enabled = tier_config.get("fallback_on_429", True)

        if not endpoints:
            raise ValueError(f"No endpoints configured for tier: {tier}")

        last_error = None

        # Try each endpoint in the tier
        for attempt, _ in enumerate(endpoints):
            endpoint = self.get_endpoint_for_tier(tier)
            if not endpoint:
                continue

            reservation = None  # Initialize outside try block
            try:
                # Wait for cooldown if needed
                await self.rate_limiter.wait_for_cooldown(endpoint)

                # Estimate tokens
                max_tokens = kwargs.get('max_tokens', tier_config.get('max_tokens', 4000))
                estimated_tokens = self._estimate_tokens(messages, max_tokens)

                # Reserve token budget
                reservation = await self.rate_limiter.reserve_tokens(
                    endpoint,
                    estimated_tokens,
                    timeout=120.0
                )

                # Forward response_format to create_llm (for structured output) - CRITICAL FIX
                if response_format:
                    kwargs['response_format'] = response_format
                    logger.info(f"📋 Forwarding response_format to error recovery: {response_format.get('type', 'unknown')}")

                # Create LLM
                llm = self.create_llm(tier, endpoint, **kwargs)
                if not llm:
                    if reservation:
                        self.rate_limiter.release_tokens(endpoint, reservation)
                    continue

                # Track timing
                start_time = time.time()

                # Make request
                logger.info(
                    f"Invoking {endpoint} for operation={operation or 'unknown'}"
                )
                response = await llm.ainvoke(messages)

                # Commit actual usage
                if reservation:
                    self.rate_limiter.commit_tokens(endpoint, reservation, estimated_tokens)

                # Record success
                latency = time.time() - start_time
                if endpoint in self.endpoint_health:
                    self.endpoint_health[endpoint].record_success(latency)

                logger.info(
                    f"✓ Success: {endpoint} responded in {latency:.2f}s"
                )
                return response

            except Exception as e:
                # Release reservation on failure
                if reservation:
                    self.rate_limiter.release_tokens(endpoint, reservation)
                error_str = str(e)
                is_429 = "429" in error_str

                # Record failure
                if endpoint in self.endpoint_health:
                    self.endpoint_health[endpoint].record_failure(is_429=is_429)

                if is_429:
                    # Report 429 to rate limiter
                    retry_after = self._extract_retry_after(error_str)
                    self.rate_limiter.report_429(endpoint, retry_after)

                    logger.warning(
                        f"429 error on {endpoint} (attempt {attempt + 1}/{len(endpoints)})"
                    )

                    # Try fallback if enabled and more endpoints available
                    if fallback_enabled and attempt < len(endpoints) - 1:
                        logger.info("Trying fallback endpoint...")
                        continue

                # Non-429 error or last endpoint - re-raise
                last_error = e
                logger.error(f"Error on {endpoint}: {e}")

        # All endpoints failed
        if last_error:
            logger.error(f"All endpoints failed for tier {tier}")
            raise last_error
        else:
            raise RuntimeError(f"No valid endpoints for tier {tier}")

    def _extract_retry_after(self, error_str: str) -> Optional[int]:
        """
        Extract Retry-After value from error message or response.

        Checks multiple formats:
        - "Retry-After: 30"
        - "retry_after": 15
        - "wait 10 seconds"
        """
        import re

        # Pattern 1: "Retry-After: 30" (HTTP header format)
        match = re.search(r"retry[- ]after[:\s]+(\d+)", error_str, re.IGNORECASE)
        if match:
            return int(match.group(1))

        # Pattern 2: JSON format "retry_after": 15
        match = re.search(r'"retry_after"[:\s]+(\d+)', error_str)
        if match:
            return int(match.group(1))

        # Pattern 3: Natural language "wait 10 seconds"
        match = re.search(r"wait\s+(\d+)\s+seconds?", error_str, re.IGNORECASE)
        if match:
            return int(match.group(1))

        # Pattern 4: "after 20s" or "after 20 s"
        match = re.search(r"after\s+(\d+)\s*s(?:ec)?", error_str, re.IGNORECASE)
        if match:
            return int(match.group(1))

        return None

    def get_health_report(self) -> Dict:
        """Get health report for all endpoints"""
        report = {}
        for endpoint, health in self.endpoint_health.items():
            report[endpoint] = {
                "total_requests": health.total_requests,
                "success_rate": health.success_rate,
                "avg_latency_ms": health.avg_latency * 1000,
                "last_429": health.last_429_time
            }
        return report


# Global singleton
_global_model_selector: Optional[ModelSelector] = None


def get_model_selector() -> Optional[ModelSelector]:
    """Get global model selector instance"""
    return _global_model_selector


def initialize_model_selector(config: Dict) -> ModelSelector:
    """Initialize global model selector"""
    global _global_model_selector
    _global_model_selector = ModelSelector(config)
    logger.info("Global ModelSelector initialized")
    return _global_model_selector
