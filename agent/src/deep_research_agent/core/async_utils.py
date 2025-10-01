"""
Async utilities for safe event loop management.

This module provides utilities to safely execute async code in any context,
handling conflicts between pytest, FastAPI, and agent code event loops.
"""
import asyncio
import threading
import logging
from typing import Any, Coroutine, Optional, Generator
from queue import Queue, Empty

logger = logging.getLogger(__name__)

# Apply nest_asyncio to allow nested event loops
try:
    import nest_asyncio
    nest_asyncio.apply()
    logger.info("nest_asyncio applied successfully")
except ImportError:
    logger.warning("nest_asyncio not available - nested event loops may fail")


class AsyncExecutor:
    """Safe async execution that works in any context."""

    @staticmethod
    def run_async_safe(coro: Coroutine, timeout: Optional[float] = None) -> Any:
        """
        Safely run async code regardless of existing event loop.

        Handles three scenarios:
        1. No existing loop -> Creates new loop
        2. Existing loop in same thread -> Uses nest_asyncio
        3. Existing loop in different thread -> Not supported (raises error)

        Args:
            coro: Coroutine to execute
            timeout: Optional timeout in seconds

        Returns:
            Result of the coroutine

        Raises:
            asyncio.TimeoutError: If timeout is exceeded
            RuntimeError: If called from different thread with existing loop
        """
        try:
            # Try to get running loop
            loop = asyncio.get_running_loop()

            # We have a loop in this thread - use nest_asyncio
            logger.debug("Existing event loop detected, using nest_asyncio")
            if timeout:
                future = asyncio.ensure_future(asyncio.wait_for(coro, timeout=timeout))
            else:
                future = asyncio.ensure_future(coro)

            # This works because nest_asyncio patches asyncio
            return asyncio.get_event_loop().run_until_complete(future)

        except RuntimeError as e:
            if "no running event loop" in str(e).lower():
                # No loop exists - create one safely
                logger.debug("No existing event loop, creating new one")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    if timeout:
                        return loop.run_until_complete(
                            asyncio.wait_for(coro, timeout=timeout)
                        )
                    return loop.run_until_complete(coro)
                finally:
                    loop.close()
            else:
                # Some other runtime error
                raise

    @staticmethod
    def stream_async_safe(
        async_gen,
        timeout_per_item: float = 30.0,
        first_item_timeout: float = None
    ) -> Generator[Any, None, None]:
        """
        Safely stream from async generator with timeout per item.

        Args:
            async_gen: Async generator to stream from
            timeout_per_item: Timeout in seconds for each item after the first
            first_item_timeout: Timeout for first item (default: 180s for complex queries)

        Yields:
            Items from the async generator

        Raises:
            TimeoutError: If no item received within timeout
            StopIteration: When generator is exhausted
        """
        # Default first item timeout is longer as research may take time to start
        if first_item_timeout is None:
            first_item_timeout = 180.0  # 3 minutes for first item

        try:
            # Try to get running loop
            loop = asyncio.get_running_loop()

            # Existing loop - yield items with timeout using nest_asyncio
            logger.debug("Streaming with existing event loop")
            is_first = True
            while True:
                try:
                    timeout = first_item_timeout if is_first else timeout_per_item
                    # Use nest_asyncio to run async code in existing loop
                    item = asyncio.get_event_loop().run_until_complete(
                        asyncio.wait_for(
                            async_gen.__anext__(),
                            timeout=timeout
                        )
                    )
                    yield item
                    is_first = False
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    if is_first:
                        raise TimeoutError(
                            f"No initial response from research agent within {timeout}s. "
                            f"The query may be too complex or the system may be under heavy load."
                        )
                    else:
                        raise TimeoutError(
                            f"No item received from async generator within {timeout}s"
                        )

        except RuntimeError as e:
            if "no running event loop" not in str(e).lower():
                raise

            # No loop - use dedicated thread for streaming
            logger.debug("No existing event loop, using dedicated thread for streaming")
            queue = Queue()
            exception_holder = [None]
            is_first_item = [True]  # Track if we're waiting for first item

            def run_stream_thread():
                """Thread function to consume async generator."""
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    while True:
                        try:
                            # Use longer timeout for first item
                            timeout = first_item_timeout if is_first_item[0] else timeout_per_item
                            item = loop.run_until_complete(
                                asyncio.wait_for(
                                    async_gen.__anext__(),
                                    timeout=timeout
                                )
                            )
                            queue.put(('item', item))
                            is_first_item[0] = False
                        except StopAsyncIteration:
                            queue.put(('done', None))
                            break
                        except asyncio.TimeoutError:
                            if is_first_item[0]:
                                error = TimeoutError(
                                    f"No initial response from research agent within {timeout}s. "
                                    f"The query may be too complex or the system may be under heavy load."
                                )
                            else:
                                error = TimeoutError(
                                    f"No item received from async generator within {timeout}s"
                                )
                            exception_holder[0] = error
                            queue.put(('error', error))
                            break
                        except Exception as e:
                            logger.error(f"Error in stream thread: {e}")
                            exception_holder[0] = e
                            queue.put(('error', e))
                            break
                finally:
                    loop.close()

            thread = threading.Thread(target=run_stream_thread, daemon=True)
            thread.start()

            # Yield items from queue with timeout
            is_first = True
            while True:
                try:
                    # Wait longer for first item from queue
                    queue_timeout = (first_item_timeout + 10.0) if is_first else (timeout_per_item + 5.0)
                    msg_type, data = queue.get(timeout=queue_timeout)
                    if msg_type == 'done':
                        break
                    elif msg_type == 'error':
                        raise data
                    else:
                        yield data
                        is_first = False
                except Empty:
                    if exception_holder[0]:
                        raise exception_holder[0]
                    if is_first:
                        raise TimeoutError(
                            f"No initial response from research agent within {queue_timeout}s. "
                            f"The query may be too complex or the system may be under heavy load."
                        )
                    else:
                        raise TimeoutError(
                            f"No message received from stream thread within {queue_timeout}s"
                        )
