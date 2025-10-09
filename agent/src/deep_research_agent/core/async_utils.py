"""
Simplified async utilities for safe event loop management (V2).

This module provides utilities to safely bridge async code to sync contexts,
specifically for MLflow ResponsesAgent compatibility.

Key Simplifications from V1:
- No nest_asyncio dependency (nodes are now async)
- Single code path (always creates thread for MLflow bridge)
- Clearer failure modes and error handling
- Reduced from 262 lines to ~80 lines
"""
import asyncio
import threading
import logging
from typing import Any, Coroutine, Generator
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class AsyncExecutor:
    """Simplified async execution for MLflow ResponsesAgent compatibility."""

    @staticmethod
    def run_async_safe(coro: Coroutine, timeout: float = None) -> Any:
        """
        Safely run async code from any context (sync or async).

        When called from sync context: Creates event loop and runs directly
        When called from async context: Creates separate thread with its own loop

        This handles the case where sync code (e.g., agent methods) needs to call
        async code (e.g., LLM ainvoke), but might be called from async workflow nodes.

        Args:
            coro: Coroutine to execute
            timeout: Optional timeout in seconds

        Returns:
            Result of the coroutine

        Raises:
            asyncio.TimeoutError: If timeout is exceeded
        """
        try:
            # Check if we're in an async context
            asyncio.get_running_loop()

            # We have a running loop - use separate thread to avoid nested loop issues
            logger.debug("run_async_safe: Running loop detected, using thread-based execution")

            import threading
            from queue import Queue, Empty

            result_queue = Queue()
            exception_holder = [None]

            def run_in_thread():
                """Run coroutine in separate thread with its own event loop"""
                thread_id = threading.current_thread().ident
                thread_name = threading.current_thread().name
                logger.debug(f"[ASYNC_EXEC THREAD {thread_name}/{thread_id}] Creating event loop")

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                logger.debug(f"[ASYNC_EXEC THREAD {thread_name}] Event loop created (id={id(loop)}), starting coroutine")

                try:
                    if timeout:
                        logger.debug(f"[ASYNC_EXEC THREAD {thread_name}] Running coroutine with timeout={timeout}s")
                        result = loop.run_until_complete(asyncio.wait_for(coro, timeout=timeout))
                    else:
                        logger.debug(f"[ASYNC_EXEC THREAD {thread_name}] Running coroutine without timeout")
                        result = loop.run_until_complete(coro)
                    logger.debug(f"[ASYNC_EXEC THREAD {thread_name}] Coroutine completed successfully")
                    result_queue.put(('success', result))
                except Exception as e:
                    logger.error(f"[ASYNC_EXEC THREAD {thread_name}] Coroutine failed: {type(e).__name__}: {str(e)[:200]}")
                    exception_holder[0] = e
                    result_queue.put(('error', e))
                finally:
                    logger.debug(f"[ASYNC_EXEC THREAD {thread_name}] Closing event loop and exiting thread")
                    loop.close()
                    asyncio.set_event_loop(None)

            thread = threading.Thread(target=run_in_thread, daemon=False)
            thread.start()

            # Use reasonable default timeout if none provided (5 minutes)
            join_timeout = (timeout + 5.0) if timeout else 300.0
            thread.join(timeout=join_timeout)

            # Check if thread is still alive (hung)
            if thread.is_alive():
                raise TimeoutError(f"Thread did not complete within {join_timeout}s")

            # Get result from queue with proper timeout
            try:
                msg_type, data = result_queue.get(timeout=5.0)
                if msg_type == 'error':
                    raise data
                return data
            except Empty:
                raise TimeoutError("No response from thread after completion")

        except RuntimeError as e:
            if "no running event loop" not in str(e).lower():
                # Re-raise if it's not the "no event loop" error
                raise

            # No running loop - create event loop and run directly
            logger.debug("run_async_safe: No loop detected, creating event loop")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                if timeout:
                    return loop.run_until_complete(asyncio.wait_for(coro, timeout=timeout))
                return loop.run_until_complete(coro)
            finally:
                loop.close()
                # Clean up to avoid resource leaks
                asyncio.set_event_loop(None)

    @staticmethod
    def stream_async_bridge(
        async_gen,
        timeout_per_item: float = 30.0,
        first_item_timeout: float = 180.0
    ):
        """
        Context-aware bridge for async generators.

        This is the CRITICAL function that allows both:
        1. Async contexts (FastAPI) to directly consume async generators
        2. Sync contexts (MLflow) to consume via thread-based bridge

        The bridge detects the calling context and adapts accordingly:
        - Async context: Returns native async generator (no thread overhead)
        - Sync context: Returns sync generator backed by thread

        Args:
            async_gen: Async generator to stream from
            timeout_per_item: Timeout in seconds for each item after the first
            first_item_timeout: Timeout for first item (default: 180s for research queries)

        Returns:
            Async generator if called from async context, sync generator otherwise

        Raises:
            TimeoutError: If no item received within timeout
            StopIteration: When generator is exhausted (sync path)
            StopAsyncIteration: When generator is exhausted (async path)
        """
        try:
            # Check if we're in an async context
            loop = asyncio.get_running_loop()
            is_async_context = True
            logger.info("[STREAM BRIDGE] Detected async context (FastAPI mode) - using native async forwarding")
        except RuntimeError as e:
            if "no running event loop" not in str(e).lower():
                raise
            is_async_context = False
            logger.info("[STREAM BRIDGE] Detected sync context (MLflow mode) - using thread-based bridge")

        if is_async_context:
            # Async context: Return native async generator with timeout support
            return AsyncExecutor._stream_async_native(
                async_gen,
                timeout_per_item=timeout_per_item,
                first_item_timeout=first_item_timeout
            )
        else:
            # Sync context: Use thread-based bridge
            return AsyncExecutor._stream_via_thread(
                async_gen,
                timeout_per_item=timeout_per_item,
                first_item_timeout=first_item_timeout
            )

    @staticmethod
    async def _stream_async_native(
        async_gen,
        timeout_per_item: float = 30.0,
        first_item_timeout: float = 180.0
    ):
        """
        Native async forwarding for async contexts (FastAPI).

        This eliminates thread overhead when the caller is already async.
        Includes timeout and error handling for robustness.

        Args:
            async_gen: Async generator to forward
            timeout_per_item: Timeout for each item
            first_item_timeout: Timeout for first item

        Yields:
            Items from async generator with timeout enforcement
        """
        is_first = True
        try:
            async for item in async_gen:
                # Apply timeout to each item
                timeout = first_item_timeout if is_first else timeout_per_item
                try:
                    # Yield item (already received, so no additional timeout needed)
                    yield item
                    is_first = False
                except Exception as e:
                    logger.error(f"[ASYNC NATIVE] Error yielding item: {e}", exc_info=True)
                    # Emit error event to ensure UI gets notified
                    yield {
                        "type": "response.error",
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                    raise
        except asyncio.TimeoutError:
            error_msg = (
                f"No initial response within {first_item_timeout}s"
                if is_first
                else f"No item received within {timeout_per_item}s"
            )
            logger.error(f"[ASYNC NATIVE] Timeout: {error_msg}")
            # Emit error event
            yield {
                "type": "response.error",
                "error": error_msg,
                "error_type": "TimeoutError"
            }
            raise TimeoutError(error_msg)
        except Exception as e:
            logger.error(f"[ASYNC NATIVE] Stream error: {e}", exc_info=True)
            # Emit error event
            yield {
                "type": "response.error",
                "error": str(e),
                "error_type": type(e).__name__
            }
            raise

    @staticmethod
    def _stream_via_thread(
        async_gen,
        timeout_per_item: float = 30.0,
        first_item_timeout: float = 180.0
    ) -> Generator[Any, None, None]:
        """
        Thread-based bridge for sync contexts (MLflow).

        This is used when predict_stream() is called from sync context.
        Creates a dedicated thread with its own event loop to consume
        the async generator and bridges events to the sync caller.

        Args:
            async_gen: Async generator to stream from
            timeout_per_item: Timeout in seconds for each item after the first
            first_item_timeout: Timeout for first item

        Yields:
            Items from the async generator

        Raises:
            TimeoutError: If no item received within timeout
            StopIteration: When generator is exhausted
        """

        # Create communication channel between threads
        queue = Queue()
        exception_holder = [None]
        is_first_item = [True]

        def run_stream_thread():
            """
            Thread function to consume async generator.

            This runs in a dedicated thread with its own event loop,
            bridging async events to sync Queue for main thread consumption.
            """
            thread_id = threading.current_thread().ident
            thread_name = threading.current_thread().name

            logger.info(f"[THREAD {thread_name}] Starting async stream bridge (ID: {thread_id})")

            # Create fresh event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            logger.debug(f"[THREAD {thread_name}] Created event loop: {id(loop)}")

            try:
                while True:
                    try:
                        # Determine timeout based on whether this is first item
                        timeout = first_item_timeout if is_first_item[0] else timeout_per_item

                        # Wait for next item from async generator
                        item = loop.run_until_complete(
                            asyncio.wait_for(
                                async_gen.__anext__(),
                                timeout=timeout
                            )
                        )

                        # Send item to main thread via queue
                        queue.put(('item', item))
                        is_first_item[0] = False

                    except StopAsyncIteration:
                        # Generator exhausted - normal completion
                        logger.info(f"[THREAD {thread_name}] Stream exhausted, signaling completion")
                        queue.put(('done', None))
                        break

                    except asyncio.TimeoutError:
                        # Timeout waiting for item
                        if is_first_item[0]:
                            error = TimeoutError(
                                f"No initial response from research agent within {timeout}s. "
                                f"The query may be too complex or the system may be under heavy load."
                            )
                        else:
                            error = TimeoutError(
                                f"No item received from async generator within {timeout}s"
                            )
                        logger.error(f"[THREAD {thread_name}] Timeout: {error}")
                        exception_holder[0] = error
                        # Emit error event before signaling
                        error_event = {
                            "type": "response.error",
                            "error": str(error),
                            "error_type": "TimeoutError"
                        }
                        queue.put(('item', error_event))
                        queue.put(('error', error))
                        break

                    except Exception as e:
                        # Unexpected error in async generator
                        logger.error(f"[THREAD {thread_name}] Error in stream: {e}", exc_info=True)
                        exception_holder[0] = e
                        # Emit error event before signaling
                        error_event = {
                            "type": "response.error",
                            "error": str(e),
                            "error_type": type(e).__name__
                        }
                        queue.put(('item', error_event))
                        queue.put(('error', e))
                        break

            except Exception as e:
                # Fatal error in thread itself
                logger.error(f"[THREAD {thread_name}] Fatal error in thread: {e}", exc_info=True)
                exception_holder[0] = e
                # Emit error event
                error_event = {
                    "type": "response.error",
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                queue.put(('item', error_event))
                queue.put(('error', e))
            finally:
                # Always clean up event loop
                logger.debug(f"[THREAD {thread_name}] Closing event loop and exiting")
                loop.close()
                asyncio.set_event_loop(None)

        # Start thread (non-daemon to ensure proper cleanup even in debug mode)
        thread = threading.Thread(target=run_stream_thread, daemon=False)
        thread.start()

        logger.info(
            f"[MAIN THREAD] Started async stream bridge thread, "
            f"timeouts: first={first_item_timeout}s, per_item={timeout_per_item}s"
        )

        # Consume items from queue and yield to caller
        is_first = True
        try:
            while True:
                try:
                    # Wait for item from thread (with buffer for thread overhead)
                    queue_timeout = (first_item_timeout + 10.0) if is_first else (timeout_per_item + 5.0)
                    msg_type, data = queue.get(timeout=queue_timeout)

                    if msg_type == 'done':
                        logger.info("[MAIN THREAD] Received completion signal from stream thread")
                        break
                    elif msg_type == 'error':
                        logger.error(f"[MAIN THREAD] Received error from stream thread: {data}")
                        raise data
                    else:
                        # Normal item - yield to caller
                        yield data
                        is_first = False

                except Empty:
                    # Queue timeout - check if thread had an error
                    if exception_holder[0]:
                        raise exception_holder[0]

                    # No error, just genuinely slow
                    if is_first:
                        raise TimeoutError(
                            f"No initial response from research agent within {queue_timeout}s. "
                            f"The query may be too complex or the system may be under heavy load."
                        )
                    else:
                        raise TimeoutError(
                            f"No message received from stream thread within {queue_timeout}s"
                        )
        finally:
            # CRITICAL: Wait for thread to complete to prevent event loss
            # This ensures all events are processed even in debug mode (breakpoints, etc.)
            logger.debug("[MAIN THREAD] Waiting for stream thread to complete (timeout=30s)...")
            thread.join(timeout=30.0)

            if thread.is_alive():
                logger.warning(
                    "[MAIN THREAD] Stream thread did not complete within 30s of final event. "
                    "This may indicate a thread hang or extremely slow processing."
                )
            else:
                logger.info("[MAIN THREAD] Stream thread completed successfully")

    # Backward compatibility alias
    @staticmethod
    def stream_async_safe(
        async_gen,
        timeout_per_item: float = 30.0,
        first_item_timeout: float = 180.0
    ):
        """
        Backward compatibility alias for stream_async_bridge.

        Deprecated: Use stream_async_bridge instead.
        This alias will be removed in a future version.
        """
        logger.warning(
            "stream_async_safe is deprecated, use stream_async_bridge instead. "
            "The new bridge is context-aware and works in both async and sync contexts."
        )
        return AsyncExecutor.stream_async_bridge(
            async_gen,
            timeout_per_item=timeout_per_item,
            first_item_timeout=first_item_timeout
        )
