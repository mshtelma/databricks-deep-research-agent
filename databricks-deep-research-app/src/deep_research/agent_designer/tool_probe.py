"""SafeProbe-only sampling for Agent Designer tool catalogs."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from databricks_deep_research.tools.catalog_types import (
    CatalogProvider,
    ProbeSample,
    SafeProbe,
)
from databricks_deep_research.tools.protocol import ToolContext
from databricks_deep_research.workflow.definition import ToolDeclaration

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")
_AWS_KEY_RE = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")


@dataclass(frozen=True)
class ProbeConfig:
    timeout_seconds: float = 30.0
    max_concurrent_probes: int = 4
    max_output_chars: int = 800
    persist: bool = False


def sanitize_probe_output(value: Any, *, max_chars: int) -> str:
    """Scrub common PII/secret patterns before truncation."""
    text = str(value)
    text = _EMAIL_RE.sub("[redacted-email]", text)
    text = _SSN_RE.sub("[redacted-ssn]", text)
    text = _JWT_RE.sub("[redacted-jwt]", text)
    text = _AWS_KEY_RE.sub("[redacted-aws-key]", text)
    return text[:max_chars]


class ProbeOrchestrator:
    """Run factory-declared SafeProbes with isolation and bounded concurrency."""

    def __init__(
        self,
        *,
        providers: Iterable[CatalogProvider],
        config: ProbeConfig | None = None,
        clock: Any | None = None,
    ) -> None:
        self._config = config or ProbeConfig()
        self._clock = clock or (lambda: datetime.now(UTC))
        probes: dict[str, SafeProbe | None] = {}
        for provider in providers:
            for kind, probe in provider.safe_probes.items():
                probes.setdefault(kind, probe)
        self._safe_probes: Mapping[str, SafeProbe | None] = probes

    @classmethod
    def from_default_factories(
        cls,
        *,
        config: ProbeConfig | None = None,
    ) -> ProbeOrchestrator:
        from databricks_deep_research.tools.factories import BUILTIN_FACTORIES

        seen: set[type] = set()
        providers: list[CatalogProvider] = []
        for factory_cls in BUILTIN_FACTORIES.values():
            if factory_cls in seen:
                continue
            seen.add(factory_cls)
            providers.append(factory_cls())
        return cls(providers=providers, config=config)

    async def probe(
        self,
        declarations: Iterable[ToolDeclaration],
        *,
        ctx: ToolContext,
        user_query: str | None,
    ) -> list[ProbeSample]:
        semaphore = asyncio.Semaphore(max(1, self._config.max_concurrent_probes))
        decls = list(declarations)

        read_only_ctx = ToolContext(
            query=ctx.query,
            url_registry=ctx.url_registry,
            table_registry=ctx.table_registry,
            current_step=ctx.current_step,
            background_summary=ctx.background_summary,
            recent_observations=list(ctx.recent_observations),
            discovered_sources=list(ctx.discovered_sources),
            read_only=True,
            extras={**ctx.extras},
        )

        async def run_one(decl: ToolDeclaration) -> ProbeSample:
            probe = self._safe_probes.get(decl.kind)
            if probe is None:
                logger.info(
                    "TOOL_PROBE_SKIPPED tool_name=%s kind=%s reason=no SafeProbe declared",
                    decl.name,
                    decl.kind,
                )
                return ProbeSample(
                    sample_input={},
                    sample_output="",
                    probed_at=self._clock(),
                    status="skipped",
                    reason="no SafeProbe declared",
                )
            async with semaphore:
                logger.info("TOOL_PROBE_STARTED tool_name=%s kind=%s", decl.name, decl.kind)
                try:
                    sample = await asyncio.wait_for(
                        probe.run(
                            config=decl.config,
                            ctx=read_only_ctx,
                            user_query=user_query,
                        ),
                        timeout=self._config.timeout_seconds,
                    )
                except TimeoutError:
                    logger.warning(
                        "TOOL_PROBE_FAILED tool_name=%s kind=%s error=timeout",
                        decl.name,
                        decl.kind,
                    )
                    return ProbeSample(
                        sample_input={},
                        sample_output="",
                        probed_at=self._clock(),
                        status="error",
                        reason="timeout",
                    )
                except Exception as exc:  # noqa: BLE001 - isolate per tool
                    logger.warning(
                        "TOOL_PROBE_FAILED tool_name=%s kind=%s error=%s",
                        decl.name,
                        decl.kind,
                        exc,
                        exc_info=True,
                    )
                    return ProbeSample(
                        sample_input={},
                        sample_output="",
                        probed_at=self._clock(),
                        status="error",
                        reason=str(exc),
                    )

                safe_output = sanitize_probe_output(
                    sample.sample_output,
                    max_chars=self._config.max_output_chars,
                )
                logger.info(
                    "TOOL_PROBE_CAPTURED tool_name=%s kind=%s output_chars=%d",
                    decl.name,
                    decl.kind,
                    len(safe_output),
                )
                return sample.model_copy(
                    update={
                        "sample_output": safe_output,
                        "probed_at": sample.probed_at or self._clock(),
                    }
                )

        return await asyncio.gather(
            *(run_one(decl) for decl in decls),
        )


__all__ = [
    "ProbeConfig",
    "ProbeOrchestrator",
    "sanitize_probe_output",
]
