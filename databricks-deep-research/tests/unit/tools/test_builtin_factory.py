"""Unit tests for BuiltinToolFactory."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from databricks_deep_research.tools.factories.builtin import BuiltinToolFactory
from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.workflow.definition import ToolDeclaration


class TestBuiltinFactoryWebCrawl:
    """BuiltinToolFactory wires web_crawl with or without an injected crawler."""

    @pytest.mark.asyncio
    async def test_web_crawl_allows_default_crawler(self) -> None:
        """web_crawl creation should succeed when ctx.crawler is None."""
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(name="web_crawl", kind="web_crawl", config={})
        ctx = ToolFactoryContext()  # crawler=None by default

        tool = await factory.create(decl, ctx)

        assert tool.definition.name == "web_crawl"
        assert tool._crawler is None  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_web_crawl_passes_config(self) -> None:
        """timeout and max_content_length from decl.config are forwarded."""
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="web_crawl",
            kind="web_crawl",
            config={"timeout": 15.0, "max_content_length": 10_000},
        )
        ctx = ToolFactoryContext(crawler=MagicMock())

        tool = await factory.create(decl, ctx)

        # Access internals to verify config was passed through
        assert tool._timeout == 15.0  # type: ignore[attr-defined]
        assert tool._max_content_length == 10_000  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_web_crawl_default_config(self) -> None:
        """Default timeout/max_content_length when not in decl.config."""
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(name="web_crawl", kind="web_crawl", config={})
        ctx = ToolFactoryContext(crawler=MagicMock())

        tool = await factory.create(decl, ctx)

        assert tool._timeout == 30.0  # type: ignore[attr-defined]
        assert tool._max_content_length == 50_000  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_web_search_still_requires_search_client(self) -> None:
        """web_search still raises ValueError when search_client is None."""
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(name="web_search", kind="web_search", config={})
        ctx = ToolFactoryContext()

        with pytest.raises(ValueError, match="search_client required"):
            await factory.create(decl, ctx)


class TestProviderResolution:
    """Provider-based tool creation via config.provider."""

    @pytest.mark.asyncio
    async def test_jina_search_provider(self) -> None:
        pytest.importorskip(
            "databricks_deep_research.tools.builtins.jina_search",
            reason="jina_search module not yet available",
        )
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="web_search", kind="web_search",
            config={"provider": "jina"},
        )
        ctx = ToolFactoryContext(api_keys={"jina": "test-key"})

        tool = await factory.create(decl, ctx)

        assert tool.definition.name == "web_search"
        from databricks_deep_research.tools.builtins.jina_search import JinaSearchAdapter
        assert isinstance(tool._client, JinaSearchAdapter)  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_jina_search_no_key_ok(self) -> None:
        """Jina works without API key."""
        pytest.importorskip(
            "databricks_deep_research.tools.builtins.jina_search",
            reason="jina_search module not yet available",
        )
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="web_search", kind="web_search",
            config={"provider": "jina"},
        )
        ctx = ToolFactoryContext()

        tool = await factory.create(decl, ctx)
        assert tool.definition.name == "web_search"

    @pytest.mark.asyncio
    async def test_brave_provider_with_api_keys(self) -> None:
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="web_search", kind="web_search",
            config={"provider": "brave"},
        )
        ctx = ToolFactoryContext(api_keys={"brave": "brave-key-123"})

        tool = await factory.create(decl, ctx)

        assert tool.definition.name == "web_search"
        from databricks_deep_research.tools.builtins.brave_search import BraveSearchAdapter
        assert isinstance(tool._client, BraveSearchAdapter)  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_brave_provider_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="web_search", kind="web_search",
            config={"provider": "brave"},
        )
        ctx = ToolFactoryContext()

        with pytest.raises(ValueError, match="BRAVE_API_KEY"):
            await factory.create(decl, ctx)

    @pytest.mark.asyncio
    async def test_unknown_search_provider_raises(self) -> None:
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="web_search", kind="web_search",
            config={"provider": "google"},
        )
        ctx = ToolFactoryContext()

        with pytest.raises(ValueError, match="Unknown search provider.*google"):
            await factory.create(decl, ctx)

    @pytest.mark.asyncio
    async def test_no_provider_uses_legacy_search_client(self) -> None:
        """No provider key → legacy path using ctx.search_client."""
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(name="web_search", kind="web_search", config={})
        mock_client = MagicMock()
        ctx = ToolFactoryContext(search_client=mock_client)

        tool = await factory.create(decl, ctx)

        assert tool._client is mock_client  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_max_content_per_result_passed(self) -> None:
        pytest.importorskip(
            "databricks_deep_research.tools.builtins.jina_search",
            reason="jina_search module not yet available",
        )
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="web_search", kind="web_search",
            config={"provider": "jina", "max_content_per_result": 2000},
        )
        ctx = ToolFactoryContext()

        tool = await factory.create(decl, ctx)
        assert tool._max_content_per_result == 2000  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_jina_crawl_provider(self) -> None:
        pytest.importorskip(
            "databricks_deep_research.tools.builtins.jina_crawl",
            reason="jina_crawl module not yet available",
        )
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="web_crawl", kind="web_crawl",
            config={"provider": "jina"},
        )
        ctx = ToolFactoryContext(api_keys={"jina": "key"})

        tool = await factory.create(decl, ctx)

        assert tool.definition.name == "web_crawl"
        from databricks_deep_research.tools.builtins.jina_crawl import JinaCrawlAdapter
        assert isinstance(tool._crawler, JinaCrawlAdapter)  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_web_crawl_no_provider_uses_ctx_crawler(self) -> None:
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(name="web_crawl", kind="web_crawl", config={})
        mock_crawler = MagicMock()
        ctx = ToolFactoryContext(crawler=mock_crawler)

        tool = await factory.create(decl, ctx)
        assert tool._crawler is mock_crawler  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_unknown_crawl_provider_raises(self) -> None:
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="web_crawl", kind="web_crawl",
            config={"provider": "bing"},
        )
        ctx = ToolFactoryContext()

        with pytest.raises(ValueError, match="Unknown crawl provider.*bing"):
            await factory.create(decl, ctx)


class TestBuiltinFactoryTableKinds:
    """BuiltinToolFactory wires the 6 text_table kinds.

    Each kind requires a subset of {table_registry, schema_cache,
    sql_executor, table_discovery_provider}. We assert (a) supports()
    returns True for each kind, (b) catalog_cards exposes the kind, (c)
    create() builds the right tool when deps are present, and (d)
    create() raises ValueError when a required dep is missing.
    """

    _TABLE_KINDS = (
        "table_discovery",
        "table_search",
        "table_read",
        "table_neighbors",
        "table_load",
        "table_aggregate",
    )

    def test_supports_all_table_kinds(self) -> None:
        factory = BuiltinToolFactory()
        for kind in self._TABLE_KINDS:
            assert factory.supports(kind), f"factory does not support {kind!r}"

    def test_catalog_cards_present_for_all_table_kinds(self) -> None:
        for kind in self._TABLE_KINDS:
            assert kind in BuiltinToolFactory.catalog_cards
            card = BuiltinToolFactory.catalog_cards[kind]
            assert card.summary
            assert card.input_prose
            assert card.output_prose

    def test_safe_probes_present_for_all_table_kinds(self) -> None:
        for kind in self._TABLE_KINDS:
            assert kind in BuiltinToolFactory.safe_probes
            # All currently None — this asserts the keys exist.
            assert BuiltinToolFactory.safe_probes[kind] is None

    def test_legacy_delta_keys_absent(self) -> None:
        for legacy in ("delta_read", "delta_grep", "delta_context", "delta_table_read"):
            assert legacy not in BuiltinToolFactory.catalog_cards
            assert legacy not in BuiltinToolFactory.safe_probes

    @pytest.mark.asyncio
    async def test_table_discovery_create(self) -> None:
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="table_discovery", kind="table_discovery", config={}
        )
        registry = MagicMock()
        ctx = ToolFactoryContext(
            table_registry=registry,
            schema_cache=MagicMock(),
            table_discovery_provider=MagicMock(),
        )
        tool = await factory.create(decl, ctx)
        assert tool.definition.name == "table_discovery"
        assert tool.definition.source_kind == "text_table"

    @pytest.mark.asyncio
    async def test_table_discovery_requires_registry(self) -> None:
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="table_discovery", kind="table_discovery", config={}
        )
        ctx = ToolFactoryContext()  # no table_registry
        with pytest.raises(ValueError, match="table_registry required"):
            await factory.create(decl, ctx)

    @pytest.mark.asyncio
    async def test_table_search_create(self) -> None:
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="table_search", kind="table_search", config={}
        )
        ctx = ToolFactoryContext(
            table_registry=MagicMock(),
            schema_cache=MagicMock(),
            sql_executor=MagicMock(),
        )
        tool = await factory.create(decl, ctx)
        assert tool.definition.name == "table_search"
        assert tool.definition.source_kind == "text_table"

    @pytest.mark.asyncio
    async def test_table_decl_with_table_name_registers_bound_default_binding(
        self,
    ) -> None:
        from databricks_deep_research.tools.builtins.text_table import (
            BindingSource,
            Schema,
            SchemaColumn,
            TableBindingRegistry,
        )
        from databricks_deep_research.tools.protocol import ToolContext

        class _SchemaCache:
            def get(self, fqn: str, user_token: str) -> Schema:
                return Schema(
                    fqn=fqn,
                    columns=(
                        SchemaColumn(name="chunk_id", data_type="string"),
                        SchemaColumn(name="content", data_type="string"),
                        SchemaColumn(name="file_name", data_type="string"),
                    ),
                )

        seen_sql: list[str] = []

        def sql_executor(sql: str, params: list[object], token: str) -> list[dict[str, str]]:
            del params, token
            seen_sql.append(sql)
            return [
                {
                    "chunk_id": "r1",
                    "content": "needle in text",
                    "file_name": "doc.txt",
                }
            ]

        registry = TableBindingRegistry()
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="treasury_grep",
            kind="table_search",
            config={
                "table_name": "cat.sch.docs",
                "content_column": "content",
                "columns": ["chunk_id", "content", "file_name"],
            },
            description="Search selected table rows.",
        )
        ctx = ToolFactoryContext(
            table_registry=registry,
            schema_cache=_SchemaCache(),
            sql_executor=sql_executor,
        )

        tool = await factory.create(decl, ctx)

        assert tool.definition.name == "treasury_grep"
        assert tool.definition.parameters["required"] == ["query"]
        binding = registry.get("treasury_grep")
        assert binding.source is BindingSource.BOUND
        assert binding.roles is not None
        assert binding.roles.id_column == "chunk_id"
        assert binding.roles.content_column == "content"

        args = tool.validate_arguments({"query": "needle"})
        result = await tool.execute(args, ToolContext())
        assert result.success is True
        assert result.data["rows"][0]["id"] == "r1"
        assert "LIKE" in seen_sql[0]

    @pytest.mark.asyncio
    async def test_table_decl_without_roles_registers_discovered_default_binding(
        self,
    ) -> None:
        from databricks_deep_research.tools.builtins.text_table import (
            BindingSource,
            Schema,
            SchemaColumn,
            TableBindingRegistry,
        )

        class _SchemaCache:
            def get(self, fqn: str, user_token: str) -> Schema:
                del user_token
                return Schema(
                    fqn=fqn,
                    columns=(
                        SchemaColumn(name="id", data_type="string"),
                        SchemaColumn(name="body", data_type="string"),
                    ),
                )

        registry = TableBindingRegistry()
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="table_reader",
            kind="table_read",
            config={"table_name": "cat.sch.raw"},
        )

        def sql_executor(sql: str, params: list[object], token: str) -> list[dict[str, str]]:
            del sql, params, token
            return []

        ctx = ToolFactoryContext(
            table_registry=registry,
            schema_cache=_SchemaCache(),
            sql_executor=sql_executor,
        )

        tool = await factory.create(decl, ctx)

        assert tool.definition.name == "table_reader"
        assert tool.definition.parameters["required"] == []
        binding = registry.get("table_reader")
        assert binding.source is BindingSource.DISCOVERED
        assert binding.roles is None
        assert tool.validate_arguments({})["binding"] == "table_reader"

    @pytest.mark.asyncio
    async def test_table_search_requires_sql_executor(self) -> None:
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="table_search", kind="table_search", config={}
        )
        ctx = ToolFactoryContext(
            table_registry=MagicMock(),
            schema_cache=MagicMock(),
            # sql_executor missing
        )
        with pytest.raises(ValueError, match="sql_executor required"):
            await factory.create(decl, ctx)

    @pytest.mark.asyncio
    async def test_table_search_requires_schema_cache(self) -> None:
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="table_search", kind="table_search", config={}
        )
        ctx = ToolFactoryContext(
            table_registry=MagicMock(),
            sql_executor=MagicMock(),
            # schema_cache missing
        )
        with pytest.raises(ValueError, match="schema_cache required"):
            await factory.create(decl, ctx)

    @pytest.mark.asyncio
    async def test_table_read_create(self) -> None:
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(name="table_read", kind="table_read", config={})
        ctx = ToolFactoryContext(
            table_registry=MagicMock(),
            schema_cache=MagicMock(),
            sql_executor=MagicMock(),
        )
        tool = await factory.create(decl, ctx)
        assert tool.definition.name == "table_read"
        assert tool.definition.source_kind == "text_table"

    @pytest.mark.asyncio
    async def test_table_read_requires_table_registry(self) -> None:
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(name="table_read", kind="table_read", config={})
        ctx = ToolFactoryContext(
            schema_cache=MagicMock(),
            sql_executor=MagicMock(),
        )
        with pytest.raises(ValueError, match="table_registry required"):
            await factory.create(decl, ctx)

    @pytest.mark.asyncio
    async def test_table_neighbors_create(self) -> None:
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="table_neighbors", kind="table_neighbors", config={}
        )
        ctx = ToolFactoryContext(
            table_registry=MagicMock(),
            schema_cache=MagicMock(),
            sql_executor=MagicMock(),
        )
        tool = await factory.create(decl, ctx)
        assert tool.definition.name == "table_neighbors"
        assert tool.definition.source_kind == "text_table"

    @pytest.mark.asyncio
    async def test_table_load_create_no_compute_resolver(self) -> None:
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(name="table_load", kind="table_load", config={})
        ctx = ToolFactoryContext(
            table_registry=MagicMock(),
            schema_cache=MagicMock(),
            sql_executor=MagicMock(),
        )
        tool = await factory.create(decl, ctx)
        assert tool.definition.name == "table_load"
        # No compute tool in the resolver cache → namespace_setter is None.
        assert tool._namespace_setter is None  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_table_load_wires_compute_namespace_setter(self) -> None:
        """When sibling 'compute' tool is in the resolver cache, table_load
        wires its inject_variable as the namespace_setter."""
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(name="table_load", kind="table_load", config={})

        compute_stub = MagicMock()
        compute_stub.inject_variable = MagicMock()
        ctx = ToolFactoryContext(
            table_registry=MagicMock(),
            schema_cache=MagicMock(),
            sql_executor=MagicMock(),
            extras={"_resolver_cache": {"compute": compute_stub}},
        )
        tool = await factory.create(decl, ctx)
        assert tool._namespace_setter is compute_stub.inject_variable  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_table_aggregate_create(self) -> None:
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="table_aggregate", kind="table_aggregate", config={}
        )
        ctx = ToolFactoryContext(
            table_registry=MagicMock(),
            schema_cache=MagicMock(),
            sql_executor=MagicMock(),
        )
        tool = await factory.create(decl, ctx)
        assert tool.definition.name == "table_aggregate"
        assert tool.definition.source_kind == "text_table"

    @pytest.mark.asyncio
    async def test_table_aggregate_requires_sql_executor(self) -> None:
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="table_aggregate", kind="table_aggregate", config={}
        )
        ctx = ToolFactoryContext(
            table_registry=MagicMock(),
            schema_cache=MagicMock(),
        )
        with pytest.raises(ValueError, match="sql_executor required"):
            await factory.create(decl, ctx)


class TestBuiltinFactoryNoLegacyKinds:
    """The legacy delta_* and pre-text_table table_read kinds must not be
    accepted any longer."""

    _LEGACY = ("delta_read", "delta_grep", "delta_context", "delta_table_read")

    def test_supports_returns_false_for_legacy(self) -> None:
        factory = BuiltinToolFactory()
        for kind in self._LEGACY:
            assert not factory.supports(kind)

    @pytest.mark.asyncio
    async def test_create_raises_for_legacy(self) -> None:
        factory = BuiltinToolFactory()
        for kind in self._LEGACY:
            decl = ToolDeclaration(name=kind, kind=kind, config={})
            ctx = ToolFactoryContext()
            with pytest.raises(ValueError, match=f"Unsupported kind: {kind}"):
                await factory.create(decl, ctx)


class TestDatabricksSearchProviderIdentity:
    """Built-in web search authenticates as the app/SP serving client, NOT the
    OBO ``user_token`` — model serving runs as the app, and the OBO token need
    not carry the ``model-serving`` foundation-model passthrough scope."""

    def test_prefers_serving_client_provider_over_obo(self) -> None:
        from databricks_deep_research.tools.factories.builtin import (
            _build_databricks_search_provider,
        )

        sp_client = object()  # sentinel SP serving client
        ctx = ToolFactoryContext(
            user_token="obo-token-must-be-ignored",
            serving_client_provider=lambda: sp_client,
        )
        adapter = _build_databricks_search_provider(
            ctx, {"model": "databricks-gemini-3-1-flash-lite"}
        )
        assert adapter._client_provider() is sp_client

    def test_falls_back_to_workspace_client(self) -> None:
        from databricks_deep_research.tools.factories.builtin import (
            _build_databricks_search_provider,
        )

        ws = MagicMock()
        ws.config.host = "https://example.cloud.databricks.com"
        ws.config.authenticate.return_value = {"Authorization": "Bearer sp-from-ws"}
        ctx = ToolFactoryContext(workspace_client=ws)  # no serving_client_provider
        adapter = _build_databricks_search_provider(ctx, {"model": "databricks-gpt-5"})
        assert adapter._client_provider().api_key == "sp-from-ws"

    def test_raises_without_serving_provider_or_workspace_client(self) -> None:
        from databricks_deep_research.tools.factories.builtin import (
            _build_databricks_search_provider,
        )

        ctx = ToolFactoryContext()  # neither serving_client_provider nor workspace_client
        with pytest.raises(
            ValueError, match="serving_client_provider or workspace_client"
        ):
            _build_databricks_search_provider(ctx, {"model": "databricks-gpt-5"})
