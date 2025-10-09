"""
LangChain-compatible wrapper for rate-limited model invocations.

This module provides a drop-in replacement for ChatDatabricks that adds:
- Automatic rate limiting via ModelSelector
- Intelligent endpoint rotation
- 429 error handling with fallback
- Token budget tracking
- Comprehensive logging

The wrapper delegates all complexity to ModelSelector.invoke_with_fallback,
which already handles endpoint selection, rotation, 429 errors, and fallback.
"""

import logging
import asyncio
import time
from typing import List, Any, Optional, Dict
from langchain_core.messages import BaseMessage, AIMessage
from .structured_output import parse_llm_response

logger = logging.getLogger(__name__)


class RateLimitedChatModel:
    """
    LangChain-compatible chat model wrapper with rate limiting.

    This class looks like ChatDatabricks to agents, but internally uses
    ModelSelector for intelligent endpoint selection and rate limiting.

    Key Features:
    - Matches LangChain interface (ainvoke, invoke)
    - Automatic tier-based endpoint selection
    - Rate limiting with token budget tracking
    - 429 error handling with automatic fallback
    - Endpoint rotation (round-robin, LRU, random)
    - Comprehensive logging at every step

    Usage:
        model = RateLimitedChatModel(
            tier="analytical",
            model_selector=selector,
            operation="planning"
        )
        response = await model.ainvoke(messages)
    """

    def __init__(
        self,
        tier: str,
        model_selector: 'ModelSelector',
        operation: str = "default"
    ):
        """
        Initialize rate-limited chat model.

        Args:
            tier: Model tier ("simple", "analytical", "complex")
            model_selector: ModelSelector instance for endpoint management
            operation: Operation name for logging and telemetry
        """
        self.tier = tier
        self.model_selector = model_selector
        self.operation = operation

        workflow_config: Dict[str, Any] = {}
        try:
            if self.model_selector and getattr(self.model_selector, "config", None):
                workflow_config = self.model_selector.config.get("workflow", {}) or {}
        except Exception as exc:  # Defensive: config access should never break initialization
            logger.debug(f"RateLimitedChatModel could not read workflow config: {exc}")

        timeout_override = workflow_config.get("llm_timeout_seconds")
        if timeout_override is None:
            timeout_override = workflow_config.get("max_wall_clock_seconds")

        try:
            default_timeout = float(timeout_override) if timeout_override is not None else 300.0
        except (TypeError, ValueError):
            default_timeout = 300.0

        if default_timeout <= 0:
            default_timeout = 300.0

        self.default_timeout_seconds = default_timeout

        # Read tier degradation config from ModelSelector
        # Default to True for backward compatibility if config missing
        tier_fallback_config: Dict[str, Any] = {}
        try:
            if self.model_selector and getattr(self.model_selector, "config", None):
                tier_fallback_config = self.model_selector.config.get("tier_fallback", {}) or {}
        except Exception as exc:
            # Defensive: config access should never break initialization
            logger.debug(f"RateLimitedChatModel could not read tier_fallback config: {exc}")

        self.enable_tier_degradation = tier_fallback_config.get("enable_cross_tier_fallback", True)

        # Log creation with degradation setting
        logger.info(
            f"🎯 RateLimitedChatModel created: "
            f"tier={tier}, operation={operation}, "
            f"default_timeout={self.default_timeout_seconds:.0f}s, "
            f"tier_degradation={'enabled' if self.enable_tier_degradation else 'disabled'}"
        )

    async def ainvoke(
        self,
        messages: List[BaseMessage],
        **kwargs
    ) -> AIMessage:
        """
        Async invocation with automatic rate limiting and fallback.

        This method matches LangChain's ChatDatabricks.ainvoke interface
        exactly, making it a drop-in replacement. Internally, it delegates
        to ModelSelector.invoke_with_fallback which handles:

        - Endpoint selection (with configured rotation strategy)
        - Token budget checking (prevents 429s)
        - 429 error handling (with cooldowns)
        - Automatic fallback to next endpoint
        - Health tracking and telemetry

        Args:
            messages: List of LangChain message objects
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            AIMessage response from the LLM

        Raises:
            Exception: If all endpoints fail or rate limits exhausted
        """
        # Get tier configuration to check endpoint status
        tier_config = self.model_selector.get_tier_config(self.tier)
        endpoints = tier_config.get("endpoints", [])

        # DIAGNOSTIC: Log what we got from tier config
        logger.info(
            f"[DIAGNOSTIC] RateLimitedChatModel.ainvoke() | "
            f"tier='{self.tier}' | "
            f"operation='{self.operation}' | "
            f"tier_config_keys={list(tier_config.keys())} | "
            f"endpoints={endpoints}"
        )

        # Check endpoint status BEFORE request
        logger.debug(f"[DIAGNOSTIC] About to call get_endpoint_status_summary with endpoints={endpoints}")
        status_list, available_count = self.model_selector.rate_limiter.get_endpoint_status_summary(endpoints)
        logger.debug(f"[DIAGNOSTIC] get_endpoint_status_summary returned: status_list={status_list}, available_count={available_count}")

        # Log invocation start with basic info
        logger.info(
            f"🚀 LLM Request: "
            f"tier={self.tier}, "
            f"operation={self.operation}, "
            f"messages={len(messages)}"
        )

        # Show detailed endpoint status
        status_summary = self.model_selector.rate_limiter.format_endpoint_status(endpoints)
        logger.info(status_summary)

        # Determine and log action
        if available_count > 0:
            # Find first available endpoint
            first_available = next(s for s in status_list if s["available"])
            logger.info(f"→ Proceeding immediately with {first_available['endpoint']}")
        else:
            # All in cooldown - find shortest wait
            shortest = min(status_list, key=lambda s: s["cooldown_remaining"])
            logger.warning(
                f"⏳ All {len(endpoints)} endpoints in cooldown. "
                f"Waiting {shortest['cooldown_remaining']:.1f}s for {shortest['endpoint']} (next available)..."
            )

        # Extract timeout from kwargs or use default derived from workflow config
        timeout_seconds = kwargs.pop('timeout', self.default_timeout_seconds)
        try:
            timeout_seconds = float(timeout_seconds)
        except (TypeError, ValueError):
            timeout_seconds = self.default_timeout_seconds
        if timeout_seconds <= 0:
            timeout_seconds = self.default_timeout_seconds
        llm_start_time = time.time()

        # Extract response_format if present (for structured output)
        response_format = kwargs.pop('response_format', None)

        logger.debug(
            f"[LLM CALL] Starting with timeout={timeout_seconds}s | "
            f"tier={self.tier} | operation={self.operation} | "
            f"structured_output={response_format is not None} | "
            f"(allows retries+cooldowns)"
        )

        try:
            # Wrap LLM call with timeout guard to prevent infinite hangs
            # This is critical for debug mode and when endpoints are slow/unreachable
            response = await asyncio.wait_for(
                self.model_selector.invoke_with_smart_fallback(
                    tier=self.tier,
                    messages=messages,
                    operation=self.operation,
                    allow_tier_degradation=self.enable_tier_degradation,  # Respect config
                    response_format=response_format,  # Forward extracted response_format
                    **kwargs
                ),
                timeout=timeout_seconds
            )

            elapsed = time.time() - llm_start_time
            logger.info(
                f"✅ LLM Response received: "
                f"tier={self.tier}, "
                f"operation={self.operation}, "
                f"elapsed={elapsed:.1f}s"
            )

            return response

        except asyncio.TimeoutError:
            elapsed = time.time() - llm_start_time
            error_msg = (
                f"LLM call timed out after {timeout_seconds}s (elapsed={elapsed:.1f}s) | "
                f"tier={self.tier} | operation={self.operation}. "
                f"This may indicate network issues, endpoint overload, or slow model response. "
                f"Try increasing timeout or check endpoint health."
            )
            logger.error(f"❌ TIMEOUT: {error_msg}")
            raise TimeoutError(error_msg)

        except Exception as e:
            # Log failure
            logger.error(
                f"❌ LLM Request failed: "
                f"tier={self.tier}, "
                f"operation={self.operation}, "
                f"error={type(e).__name__}: {str(e)[:100]}"
            )
            raise

    def invoke(
        self,
        messages: List[BaseMessage],
        **kwargs
    ) -> AIMessage:
        """
        Synchronous invocation wrapper.

        For compatibility with code that doesn't use async.
        Internally wraps ainvoke with asyncio.run or AsyncExecutor.

        IMPORTANT: When called from async context (e.g., async workflow nodes),
        this uses AsyncExecutor.run_async_safe() to avoid nested event loop issues.
        This enables debug mode compatibility and proper async isolation.

        Args:
            messages: List of LangChain message objects
            **kwargs: Additional parameters

        Returns:
            AIMessage response from the LLM
        """
        import asyncio
        import threading
        thread_name = threading.current_thread().name

        try:
            # Check if we're in a running event loop
            loop = asyncio.get_running_loop()

            # CRITICAL FIX: Always use separate thread when event loop detected
            # Never block the event loop thread (causes deadlock with create_task)
            # Thread-3 isolation prevents blocking Thread-2's event loop
            logger.debug(
                f"[INVOKE] Thread={thread_name}, detected running loop={id(loop)}, "
                f"using separate thread (AsyncExecutor) for isolation"
            )

            # Extract timeout from kwargs or fall back to workflow-derived default
            timeout = kwargs.pop('timeout', self.default_timeout_seconds)
            try:
                timeout = float(timeout)
            except (TypeError, ValueError):
                timeout = self.default_timeout_seconds
            if timeout <= 0:
                timeout = self.default_timeout_seconds
            logger.debug(f"[INVOKE] Using LLM timeout={timeout}s (extracted from kwargs or default)")

            from .async_utils import AsyncExecutor
            return AsyncExecutor.run_async_safe(
                self.ainvoke(messages, timeout=timeout, **kwargs),
                timeout=timeout + 60.0  # Add generous buffer for thread creation/cleanup overhead
            )
        except RuntimeError as e:
            if "no running event loop" not in str(e).lower():
                raise
            # No running loop - safe to use asyncio.run() directly
            logger.debug(f"[INVOKE] Thread={thread_name}, no loop detected, using asyncio.run()")
            timeout = kwargs.pop('timeout', self.default_timeout_seconds)
            try:
                timeout = float(timeout)
            except (TypeError, ValueError):
                timeout = self.default_timeout_seconds
            if timeout <= 0:
                timeout = self.default_timeout_seconds
            return asyncio.run(self.ainvoke(messages, timeout=timeout, **kwargs))

    @property
    def _identifying_params(self) -> Dict:
        """
        Return identifying parameters for LangChain compatibility.

        Some LangChain utilities expect this property for model
        introspection and debugging.

        Returns:
            Dictionary with tier and operation information
        """
        return {
            "tier": self.tier,
            "operation": self.operation,
            "model_type": "rate_limited_chat_model"
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"RateLimitedChatModel("
            f"tier={self.tier}, "
            f"operation={self.operation})"
        )

    def with_structured_output(
        self,
        schema,
        **kwargs
    ) -> 'StructuredOutputWrapper':
        """
        Enable structured JSON output following Databricks pattern.

        Returns a wrapper that maintains rate limiting while parsing
        structured output using Databricks response_format.

        Args:
            schema: Pydantic BaseModel class or dict schema
            **kwargs: Additional options (method, include_raw, etc.)

        Returns:
            StructuredOutputWrapper instance

        Example:
            structured_llm = llm.with_structured_output(
                TableBlock,
                method="json_schema"
            )
            result = structured_llm.invoke(messages)
        """
        return StructuredOutputWrapper(
            rate_limited_model=self,
            schema=schema,
            **kwargs
        )


class StructuredOutputWrapper:
    """
    Wrapper for structured output with rate limiting.

    Maintains all rate limiting logic while enabling structured JSON parsing
    using Databricks response_format pattern.
    """

    def __init__(
        self,
        rate_limited_model: RateLimitedChatModel,
        schema,
        method: str = "json_schema",
        include_raw: bool = False,
        **kwargs
    ):
        """
        Initialize structured output wrapper.

        Args:
            rate_limited_model: RateLimitedChatModel instance
            schema: Pydantic BaseModel class or dict schema
            method: "json_schema" (default) or "json_object"
            include_raw: Whether to include raw response (not implemented)
            **kwargs: Additional options
        """
        self.rate_limited_model = rate_limited_model
        self.schema = schema
        self.method = method
        self.include_raw = include_raw
        self.kwargs = kwargs

        # Check if schema is Pydantic model
        try:
            from pydantic import BaseModel
            self.is_pydantic = isinstance(schema, type) and issubclass(schema, BaseModel)
        except ImportError:
            self.is_pydantic = False
            logger.warning("Pydantic not available, using dict-based schema validation")

        # Build Databricks response_format
        self.response_format = self._build_response_format()

    def _get_databricks_schema(self, pydantic_model) -> Dict:
        """
        Get Databricks-compatible schema (strips unsupported features).

        Databricks structured output has strict requirements:
        - Max 64 keys
        - No additionalProperties, $defs, $ref
        - No anyOf, oneOf, allOf, prefixItems
        - No pattern, maxProperties, minProperties, maxLength
        - Prefer flat schemas (less nesting = better quality)

        Args:
            pydantic_model: Pydantic BaseModel class

        Returns:
            Dict: Simplified schema compatible with Databricks

        Note:
            Uses get_databricks_schema() helper from report_models_structured.py
            which strips all unsupported features.
        """
        # Check if model has custom method
        if hasattr(pydantic_model, 'get_databricks_schema'):
            logger.debug(f"Using custom get_databricks_schema() from {pydantic_model.__name__}")
            return pydantic_model.get_databricks_schema()

        # Fall back to automatic simplification
        try:
            from .report_models_structured import get_databricks_schema
            logger.debug(f"Using automatic schema simplification for {pydantic_model.__name__}")
            return get_databricks_schema(pydantic_model)
        except ImportError:
            # Last resort: use raw schema with warning
            logger.warning(
                f"get_databricks_schema not available, using raw schema for {pydantic_model.__name__}. "
                f"This may violate Databricks requirements (additionalProperties, $ref, etc.)"
            )
            return pydantic_model.model_json_schema()

    def _build_response_format(self) -> Dict:
        """Build response_format dict for Databricks API."""

        if self.method == "json_schema":
            # Handle both Pydantic models and dict schemas
            if self.is_pydantic:
                # Get Databricks-compatible schema (strips unsupported features)
                schema_dict = self._get_databricks_schema(self.schema)
                schema_name = self.schema.__name__.lower()
            elif isinstance(self.schema, dict):
                # Dict schema provided directly
                schema_dict = self.schema
                schema_name = self.schema.get('title', 'schema').lower()
                logger.debug(f"Using provided dict schema: {len(schema_dict)} keys")
            else:
                logger.warning(
                    f"Unrecognized schema type: {type(self.schema).__name__}, "
                    f"falling back to json_object mode"
                )
                return {"type": "json_object"}

            return {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema_dict,
                    "strict": True  # Databricks best practice
                }
            }
        else:
            # Generic JSON object fallback (json_mode)
            return {"type": "json_object"}

    async def ainvoke(
        self,
        messages: List[BaseMessage],
        **invoke_kwargs
    ):
        """
        Async invoke with structured output parsing and timeout protection.

        Routes through RateLimitedChatModel.ainvoke to leverage asyncio.wait_for
        timeout protection and diagnostic logging.

        Args:
            messages: List of LangChain message objects
            **invoke_kwargs: Additional invocation parameters

        Returns:
            Parsed and validated Pydantic model or dict

        Raises:
            ValueError: If JSON parsing or validation fails
        """
        # ========== COMPREHENSIVE DEBUG LOGGING ==========
        import json

        logger.info("=" * 80)
        logger.info(f"🚀 STRUCTURED OUTPUT REQUEST | Schema: {self.schema.__name__ if self.is_pydantic else 'dict'}")
        logger.info(f"   Method: {self.method}")

        # Log the exact schema being sent to the model
        if self.response_format:
            try:
                schema_json = json.dumps(self.response_format, indent=2)
                logger.info(f"   Response Format:\n{schema_json}")
            except Exception as e:
                logger.warning(f"Could not serialize response_format: {e}")

        # Log all messages being sent
        for i, msg in enumerate(messages):
            msg_type = msg.__class__.__name__
            content_preview = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
            logger.info(f"   Message[{i}] ({msg_type}):\n{content_preview}")

        # Add response_format to kwargs for forwarding through RateLimitedChatModel
        invoke_kwargs['response_format'] = self.response_format

        # Route through parent model for timeout protection
        # This ensures asyncio.wait_for wraps the call (line 198 of RateLimitedChatModel.ainvoke)
        response = await self.rate_limited_model.ainvoke(messages, **invoke_kwargs)

        # ========== LOG RAW LLM RESPONSE ==========
        logger.info("-" * 80)
        logger.info(f"📨 RAW LLM RESPONSE | Type: {type(response.content).__name__}")

        if isinstance(response.content, list):
            logger.info(f"   Response is LIST with {len(response.content)} items:")
            for i, item in enumerate(response.content[:5]):  # First 5 items
                item_preview = str(item)[:150] + "..." if len(str(item)) > 150 else str(item)
                logger.info(f"   [{i}] {type(item).__name__}: {item_preview}")
            if len(response.content) > 5:
                logger.info(f"   ... and {len(response.content) - 5} more items")
        elif isinstance(response.content, str):
            content_preview = response.content[:300] + "..." if len(response.content) > 300 else response.content
            logger.info(f"   Response STRING (first 300 chars):\n{content_preview}")
        else:
            content_str = str(response.content)
            content_preview = content_str[:300] + "..." if len(content_str) > 300 else content_str
            logger.info(f"   Response: {content_preview}")

        logger.info("=" * 80)

        # Parse and validate response using unified parser
        try:
            logger.debug(f"📋 Parsing structured response with unified parser")

            result = parse_llm_response(
                response,
                schema=self.schema if self.is_pydantic else None,
                repair_json=True
            )

            logger.info(f"✅ Structured output parsed successfully: {type(result).__name__}")
            return result

        except ValueError as e:
            # Enhanced error logging for debugging
            logger.error(f"❌ Structured output parsing/validation failed: {e}")
            logger.error(f"   Raw response type: {type(response.content).__name__}")
            if isinstance(response.content, list):
                block_types = [b.get('type', '?') if isinstance(b, dict) else type(b).__name__ for b in response.content]
                logger.error(f"   Response blocks: {block_types}")
            logger.error(f"   This indicates the model did not follow the provided schema.")
            raise
        except Exception as e:
            logger.error(
                f"❌ Unexpected error in structured output handling | "
                f"Schema: {self.schema.__name__ if self.is_pydantic else 'dict'} | "
                f"Error: {type(e).__name__}: {str(e)[:200]}"
            )
            raise


    def invoke(
        self,
        messages: List[BaseMessage],
        **kwargs
    ):
        """
        Synchronous invoke wrapper.

        Uses same async isolation pattern as RateLimitedChatModel.

        Args:
            messages: List of LangChain message objects
            **kwargs: Additional parameters

        Returns:
            Parsed and validated output
        """
        import asyncio
        import threading

        thread_name = threading.current_thread().name

        try:
            # Check if we're in a running event loop
            loop = asyncio.get_running_loop()

            # Use separate thread for isolation (same pattern as RateLimitedChatModel)
            logger.debug(
                f"[STRUCTURED INVOKE] Thread={thread_name}, detected running loop, "
                f"using AsyncExecutor for isolation"
            )

            timeout = kwargs.pop('timeout', self.rate_limited_model.default_timeout_seconds)
            try:
                timeout = float(timeout)
            except (TypeError, ValueError):
                timeout = self.rate_limited_model.default_timeout_seconds
            if timeout <= 0:
                timeout = self.rate_limited_model.default_timeout_seconds

            from .async_utils import AsyncExecutor
            # DON'T pass timeout to ainvoke (it extracts it from kwargs)
            # AsyncExecutor.run_async_safe has its own timeout guard
            return AsyncExecutor.run_async_safe(
                self.ainvoke(messages, **kwargs),  # timeout already popped, don't re-add
                timeout=timeout + 60.0  # Buffer for overhead
            )

        except RuntimeError as e:
            if "no running event loop" not in str(e).lower():
                raise

            # No running loop - safe to use asyncio.run()
            logger.debug(f"[STRUCTURED INVOKE] Thread={thread_name}, no loop, using asyncio.run()")
            timeout = kwargs.pop('timeout', self.rate_limited_model.default_timeout_seconds)
            try:
                timeout = float(timeout)
            except (TypeError, ValueError):
                timeout = self.rate_limited_model.default_timeout_seconds
            if timeout <= 0:
                timeout = self.rate_limited_model.default_timeout_seconds

            # DON'T pass timeout to ainvoke (it extracts it from kwargs)
            # ainvoke has its own asyncio.wait_for timeout guard (line 202)
            return asyncio.run(self.ainvoke(messages, **kwargs))
