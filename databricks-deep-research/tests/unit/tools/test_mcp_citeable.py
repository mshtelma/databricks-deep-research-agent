"""Tests for the CITEABLE MCP toolset (spec §4.3 #11) + declarative server config.

Covers:

* A citeable MCP tool result reaches the pool as a source through the SAME
  admission path (``admit_tool_result`` + ``ReactLoop._run_tool_gated``) used by
  every non-builtin tool. This is the §4.3 #11 fix: ``_MCPTool`` used to set
  ``source_kind=builtin``, which BYPASSED admission and returned NO sources.
* A ``builtin`` MCP tool (opt-out) still bypasses admission (no source).
* ``MCPServerConfig`` validation: secret-ref-only credentials, fast vs deep.
* ``build_mcp_toolset`` builds from config with a STUB client (no network),
  SSRF rejects a disallowed host, and auth headers are NEVER logged.

All tests run without network access (a fake duck-typed MCP client is injected).
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from databricks_deep_research.agents.react_loop import ReactLoop
from databricks_deep_research.agents.source_aware import admit_tool_result
from databricks_deep_research.llm.client import ToolCall
from databricks_deep_research.tools.mcp import (
    MCPSecurityError,
    MCPServerConfig,
    MCPToolset,
    build_mcp_auth,
    build_mcp_toolset,
)
from databricks_deep_research.tools.protocol import SourceKind, ToolContext

# ---------------------------------------------------------------------------
# Fakes (no network)
# ---------------------------------------------------------------------------


class _FakeTextResult:
    def __init__(self, text: str, *, is_error: bool = False) -> None:
        self.content = [type("P", (), {"type": "text", "text": text})()]
        self.isError = is_error  # noqa: N815 — match MCP spec field name


class _FakeResearchClient:
    """Duck-typed MCP client returning a prose answer (qa_assistant modality)."""

    def __init__(self, answer: str = "The 2024 revenue was 45.2 billion USD.") -> None:
        self._answer = answer
        self.last_arguments: dict[str, object] | None = None

    def list_tools(self):  # type: ignore[no-untyped-def]
        return [
            type("T", (), {
                "name": "ask_corp_filings",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
                "description": "Answer questions over corporate filings.",
            })()
        ]

    def call_tool(self, name, arguments):  # type: ignore[no-untyped-def]
        self.last_arguments = dict(arguments or {})
        return _FakeTextResult(self._answer)


class _CountingDiscoveryClient(_FakeResearchClient):
    """Counts how many times list_tools() is called (fast vs deep cadence)."""

    def __init__(self) -> None:
        super().__init__()
        self.discovery_calls = 0

    def list_tools(self):  # type: ignore[no-untyped-def]
        self.discovery_calls += 1
        return super().list_tools()


class _SyncClientWithInternalAsyncRun(_FakeResearchClient):
    """Mimics sync SDK clients that drive async internals themselves."""

    def call_tool(self, name, arguments):  # type: ignore[no-untyped-def]
        self.last_arguments = dict(arguments or {})
        return asyncio.run(self._call_tool_async(name, arguments))

    async def _call_tool_async(self, name, arguments):  # type: ignore[no-untyped-def]
        await asyncio.sleep(0)
        return _FakeTextResult(self._answer)


# ---------------------------------------------------------------------------
# §4.3 #11 — citeable source emission
# ---------------------------------------------------------------------------


def test_citeable_mcp_tool_emits_source() -> None:
    """A default (citeable) MCP tool attaches a SourceInfo with a URL."""
    ts = MCPToolset(client=_FakeResearchClient(), name_prefix="corp_")
    tool = ts.tools[0]
    # The tool is non-builtin so it is NOT bypassed by the ReAct builtin branch.
    assert tool.definition.source_kind == SourceKind.qa_assistant
    assert tool.definition.source_type == "mcp"


@pytest.mark.asyncio
async def test_citeable_mcp_result_carries_source() -> None:
    ts = MCPToolset(client=_FakeResearchClient(), name_prefix="corp_")
    tool = ts.tools[0]
    result = await tool.execute({"query": "2024 revenue"}, ToolContext())
    assert result.success
    assert "45.2 billion" in result.content
    assert len(result.sources) == 1
    src = result.sources[0]
    assert src.url.startswith("mcp://")
    assert src.content == result.content
    assert src.source_kind == SourceKind.qa_assistant


@pytest.mark.asyncio
async def test_sync_mcp_client_with_internal_asyncio_run_executes_in_thread() -> None:
    """Regression: Databricks MCP sync call_tool internally uses asyncio.run()."""
    client = _SyncClientWithInternalAsyncRun()
    ts = MCPToolset(client=client, name_prefix="corp_")
    tool = ts.tools[0]

    result = await tool.execute({"query": "2024 revenue"}, ToolContext())

    assert result.success
    assert "45.2 billion" in result.content
    assert client.last_arguments == {"query": "2024 revenue"}


@pytest.mark.asyncio
async def test_mcp_result_admitted_as_pool_source() -> None:
    """THE §4.3 #11 fix: an MCP result reaches the pool as a citeable source.

    Before the fix, ``_MCPTool`` set ``source_kind=builtin`` so
    ``admit_tool_result`` was never reached and zero sources flowed to the
    pool. Now the qa_assistant kind is admitted unconditionally.
    """
    ts = MCPToolset(client=_FakeResearchClient(), name_prefix="corp_")
    tool = ts.tools[0]
    result = await tool.execute({"query": "2024 revenue"}, ToolContext())

    admitted = admit_tool_result(
        tool.definition,
        result,
        current_step=None,
        root_query="What was the 2024 revenue?",
    )
    assert admitted.accepted_count == 1
    assert admitted.accepted_sources[0]["url"].startswith("mcp://")


@pytest.mark.asyncio
async def test_mcp_result_reaches_pool_through_react_gate() -> None:
    """End-to-end through ReactLoop._run_tool_gated — the real gating spine.

    A non-builtin MCP tool routes through admission + source/pool writes, so
    the gated call returns the admitted source (parity with web/vector tools).
    """
    ts = MCPToolset(client=_FakeResearchClient(), name_prefix="corp_")
    tool = ts.tools[0]

    loop = ReactLoop(
        llm_client=_StubLLM(),
        tools=[tool],
        tool_context=ToolContext(query="What was the 2024 revenue?"),
        node_id="n1",
    )
    tc = ToolCall(id="c1", function_name=tool.definition.name, arguments='{"query": "2024 revenue"}')
    tc_id, content, sources, meta = await loop._run_tool_gated(
        tc, tool, {"query": "2024 revenue"}, origin="toolcall",
    )
    assert tc_id == "c1"
    assert meta["accepted_source_count"] == 1
    assert len(sources) == 1
    assert sources[0]["url"].startswith("mcp://")


@pytest.mark.asyncio
async def test_builtin_mcp_tool_bypasses_admission() -> None:
    """A ``builtin`` (opt-out) MCP tool emits NO source — bypasses admission."""
    ts = MCPToolset(
        client=_FakeResearchClient(),
        name_prefix="side_",
        source_kind=SourceKind.builtin,
    )
    tool = ts.tools[0]
    assert tool.definition.source_kind == SourceKind.builtin
    result = await tool.execute({"query": "x"}, ToolContext())
    assert result.sources == []

    loop = ReactLoop(
        llm_client=_StubLLM(), tools=[tool],
        tool_context=ToolContext(query="x"), node_id="n1",
    )
    tc = ToolCall(id="c1", function_name=tool.definition.name, arguments='{"query": "x"}')
    _id, _content, sources, meta = await loop._run_tool_gated(
        tc, tool, {"query": "x"}, origin="toolcall",
    )
    assert sources == []
    assert meta["evidence_quality"] == "builtin"


# ---------------------------------------------------------------------------
# MCPServerConfig validation
# ---------------------------------------------------------------------------


def test_config_defaults_are_citeable_and_fast() -> None:
    cfg = MCPServerConfig(name="corp", url="https://mcp.example.com/sse")
    assert cfg.citeable is True
    assert cfg.strategy == "fast"
    assert cfg.transport == "http"
    assert cfg.source_kind() == SourceKind.qa_assistant


def test_config_builtin_opt_out_source_kind() -> None:
    cfg = MCPServerConfig(name="corp", url="https://mcp.example.com", citeable=False)
    assert cfg.source_kind() == SourceKind.builtin


def test_config_rejects_inline_credentials() -> None:
    """auth_type bearer/api_key MUST carry a secret_ref, never an inline token."""
    with pytest.raises(ValueError, match="requires a secret_ref"):
        MCPServerConfig(name="corp", url="https://mcp.example.com", auth_type="bearer")
    with pytest.raises(ValueError, match="requires a secret_ref"):
        MCPServerConfig(name="corp", url="https://mcp.example.com", auth_type="api_key")


def test_config_forbids_unknown_fields() -> None:
    with pytest.raises(ValueError):
        MCPServerConfig(name="corp", url="https://mcp.example.com", token="inline-secret")  # type: ignore[call-arg]


def test_config_rejects_stdio_transport() -> None:
    with pytest.raises(ValueError):
        MCPServerConfig(name="corp", url="https://mcp.example.com", transport="stdio")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# build_mcp_toolset — stub client, SSRF, secret redaction
# ---------------------------------------------------------------------------


def test_build_toolset_from_config_with_stub_client() -> None:
    cfg = MCPServerConfig(name="corp", url="https://mcp.example.com/sse", name_prefix="corp_")
    ts = build_mcp_toolset(cfg, client=_FakeResearchClient())
    assert len(ts) == 1
    assert ts.tools[0].definition.name == "corp_ask_corp_filings"
    assert ts.tools[0].definition.source_kind == SourceKind.qa_assistant


def test_build_toolset_ssrf_rejects_disallowed_host() -> None:
    """SSRF guard rejects a loopback transport URL (no client injected)."""
    cfg = MCPServerConfig(name="bad", url="http://127.0.0.1/sse")
    with pytest.raises(MCPSecurityError, match="loopback"):
        build_mcp_toolset(cfg)


def test_build_toolset_ssrf_rejects_imds() -> None:
    cfg = MCPServerConfig(name="bad", url="http://169.254.169.254/latest/meta-data/")
    with pytest.raises(MCPSecurityError, match="link_local"):
        build_mcp_toolset(cfg)


def test_secret_ref_resolved_via_resolver() -> None:
    cfg = MCPServerConfig(
        name="corp",
        url="https://mcp.example.com",
        auth_type="bearer",
        secret_ref="my_scope/mcp_token",
    )
    auth = build_mcp_auth(cfg, lambda ref: "tok-123" if ref == "my_scope/mcp_token" else None)
    assert auth is not None
    assert auth.headers() == {"Authorization": "Bearer tok-123"}


def test_missing_secret_resolver_raises() -> None:
    cfg = MCPServerConfig(
        name="corp", url="https://mcp.example.com",
        auth_type="bearer", secret_ref="s/k",
    )
    with pytest.raises(ValueError, match="needs a secret resolver"):
        build_mcp_auth(cfg, None)


def test_auth_secret_never_logged(caplog: pytest.LogCaptureFixture) -> None:
    """The resolved credential must NEVER appear in any log line."""
    secret = "super-secret-token-value-xyz"
    cfg = MCPServerConfig(
        name="corp",
        url="https://mcp.example.com/sse",
        auth_type="bearer",
        secret_ref="my_scope/mcp_token",
        name_prefix="corp_",
    )
    with caplog.at_level(logging.DEBUG):
        ts = build_mcp_toolset(
            cfg,
            client=_FakeResearchClient(),
            secret_resolver=lambda _ref: secret,
        )
    assert len(ts) == 1
    # The secret value and the bearer header must not leak into logs.
    combined = " ".join(rec.getMessage() for rec in caplog.records)
    assert secret not in combined
    assert "Bearer" not in combined


# ---------------------------------------------------------------------------
# fast vs deep discovery cadence
# ---------------------------------------------------------------------------


def test_fast_strategy_discovers_once() -> None:
    """``fast``: the toolset is built once; one discovery call covers reuse."""
    client = _CountingDiscoveryClient()
    cfg = MCPServerConfig(name="corp", url="https://mcp.example.com", strategy="fast")
    build_mcp_toolset(cfg, client=client)
    assert client.discovery_calls == 1


def test_deep_strategy_rediscovers_per_build() -> None:
    """``deep``: the host rebuilds per step, each build re-runs discovery."""
    client = _CountingDiscoveryClient()
    cfg = MCPServerConfig(name="corp", url="https://mcp.example.com", strategy="deep")
    # Simulate the host re-building the toolset for two successive steps.
    build_mcp_toolset(cfg, client=client)
    build_mcp_toolset(cfg, client=client)
    assert client.discovery_calls == 2


# ---------------------------------------------------------------------------
# Stub LLM for the ReactLoop gating-spine tests (never actually called — we
# invoke ``_run_tool_gated`` directly, but ReactLoop requires a client).
# ---------------------------------------------------------------------------


class _StubLLM:
    async def complete(self, *args, **kwargs):  # type: ignore[no-untyped-def]  # pragma: no cover
        raise AssertionError("LLM should not be called in a direct gated-call test")
