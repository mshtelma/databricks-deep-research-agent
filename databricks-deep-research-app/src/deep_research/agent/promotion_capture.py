"""App-side capture of promotion traces (spec Wave 5 / feature 6.1).

A thin, **fail-soft** wrapper over the framework's
:class:`~databricks_deep_research.promotion.PromotionTraceBuilder`. It is fed the
live framework event stream during a run and built at completion, then persisted
onto ``research_session.promotion_trace`` for later promotion (feature 6.2).

Trace capture is observability, not the product — it must NEVER break a run. Every
method swallows and logs its own errors; ``build`` returns ``None`` (→ persist
nothing, column stays NULL = "not promotable") on any failure or when no
structural steps were observed.
"""

from __future__ import annotations

import logging
from typing import Any

from databricks_deep_research.events.types import StreamEvent
from databricks_deep_research.promotion import PromotionTraceBuilder

logger = logging.getLogger(__name__)


class PromotionTraceCollector:
    """Accumulates a run's events into a persistable promotion-trace dict."""

    def __init__(self, *, run_id: str = "") -> None:
        self._run_id = run_id
        try:
            self._builder: PromotionTraceBuilder | None = PromotionTraceBuilder()
        except Exception:  # pragma: no cover - defensive
            logger.debug("PROMOTION_CAPTURE_INIT_FAILED", exc_info=True)
            self._builder = None

    def observe(self, event: StreamEvent) -> None:
        """Project one framework event into the trace (fail-soft, never raises)."""
        if self._builder is None:
            return
        try:
            self._builder.observe(event)
        except Exception:  # pragma: no cover - defensive
            logger.debug("PROMOTION_CAPTURE_OBSERVE_FAILED", exc_info=True)

    def build(self, *, query_shape: str = "") -> dict[str, Any] | None:
        """Finalize the trace as a JSON-able dict, or ``None`` if not worth persisting.

        Returns ``None`` on any error or when no structural steps were captured
        (e.g. a simple/degenerate run), so the caller persists nothing.
        """
        if self._builder is None:
            return None
        try:
            trace = self._builder.build(run_id=self._run_id, query_shape=query_shape)
            if not trace.steps:
                return None
            return trace.model_dump(mode="json")
        except Exception:
            logger.warning("PROMOTION_CAPTURE_BUILD_FAILED", exc_info=True)
            return None
