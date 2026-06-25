"""Unit tests for ExportService.

Tests the batch source loading optimization (N+1 query fix).
"""

import contextlib
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest


# Use delayed import to avoid circular import issues
@pytest.fixture
def export_service_class():
    """Import ExportService class lazily to avoid circular imports."""
    from deep_research.services.export_service import ExportService
    return ExportService


@pytest.fixture
def message_role():
    """Import MessageRole enum lazily."""
    from deep_research.models.message import MessageRole
    return MessageRole


class TestExportServiceBatchLoading:
    """Tests for batch source loading optimization."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = AsyncMock()
        session.execute = AsyncMock()
        return session

    @pytest.fixture
    def export_service(self, mock_session, export_service_class):
        """Create export service with mocked dependencies."""
        return export_service_class(session=mock_session)

    @pytest.mark.asyncio
    async def test_get_sources_for_messages_empty_list(self, export_service):
        """Test batch loading with empty message list."""
        result = await export_service._get_sources_for_messages([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_sources_for_messages_returns_grouped_sources(
        self, mock_session, export_service
    ):
        """Test that sources are correctly grouped by message_id."""
        # Arrange
        msg1_id = uuid4()
        msg2_id = uuid4()

        # Create mock Source objects
        source1 = MagicMock()
        source1.title = "Source 1"
        source1.url = "https://example.com/1"

        source2 = MagicMock()
        source2.title = "Source 2"
        source2.url = "https://example.com/2"

        source3 = MagicMock()
        source3.title = None  # Test fallback to URL
        source3.url = "https://example.com/3"

        # Mock query result - returns (Source, message_id) tuples
        mock_result = MagicMock()
        mock_result.all.return_value = [
            (source1, msg1_id),
            (source2, msg1_id),  # Two sources for msg1
            (source3, msg2_id),  # One source for msg2
        ]
        mock_session.execute.return_value = mock_result

        # Act
        result = await export_service._get_sources_for_messages([msg1_id, msg2_id])

        # Assert
        assert len(result) == 2
        assert len(result[msg1_id]) == 2
        assert len(result[msg2_id]) == 1
        assert result[msg1_id][0]["title"] == "Source 1"
        assert result[msg1_id][0]["url"] == "https://example.com/1"
        # Test title fallback to URL when title is None
        assert result[msg2_id][0]["title"] == "https://example.com/3"

    @pytest.mark.asyncio
    async def test_get_sources_for_messages_single_query(
        self, mock_session, export_service
    ):
        """Test that only one query is executed for batch loading."""
        # Arrange
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute.return_value = mock_result

        # Act
        await export_service._get_sources_for_messages([uuid4(), uuid4(), uuid4()])

        # Assert - only ONE query should be executed
        assert mock_session.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_export_markdown_uses_batch_loading(
        self, mock_session, message_role, export_service_class
    ):
        """Test that export_markdown uses batch loading instead of N+1 queries."""
        # Arrange
        chat_id = uuid4()
        user_id = "test-user"

        # Mock chat
        mock_chat = MagicMock()
        mock_chat.id = chat_id
        mock_chat.title = "Test Chat"
        mock_chat.created_at = None

        # Mock messages - 3 agent messages
        messages = []
        for i in range(3):
            msg = MagicMock()
            msg.id = uuid4()
            msg.role = message_role.AGENT
            msg.content = f"Agent response {i}"
            messages.append(msg)

        # Add a user message
        user_msg = MagicMock()
        user_msg.id = uuid4()
        user_msg.role = message_role.USER
        user_msg.content = "User question"
        messages.insert(0, user_msg)

        # Mock chat service
        mock_chat_service = AsyncMock()
        mock_chat_service.get_for_user = AsyncMock(return_value=mock_chat)

        # Mock message service
        mock_message_service = AsyncMock()
        mock_message_service.list_messages = AsyncMock(
            return_value=(messages, len(messages))
        )

        # Mock batch source loading - return empty for all
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute.return_value = mock_result

        # Create service with mocks
        service = export_service_class(
            session=mock_session,
            chat_service=mock_chat_service,
            message_service=mock_message_service,
        )

        # Act
        await service.export_markdown(chat_id, user_id, include_sources=True)

        # Assert - Should only execute ONE query for batch source loading
        # Not 2 * 3 = 6 queries (one for session + one for sources per agent message)
        assert mock_session.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_export_markdown_skips_batch_when_sources_disabled(
        self, mock_session, message_role, export_service_class
    ):
        """Test that batch loading is skipped when include_sources=False."""
        # Arrange
        chat_id = uuid4()
        user_id = "test-user"

        mock_chat = MagicMock()
        mock_chat.id = chat_id
        mock_chat.title = "Test Chat"
        mock_chat.created_at = None

        mock_msg = MagicMock()
        mock_msg.id = uuid4()
        mock_msg.role = message_role.USER
        mock_msg.content = "User question"

        mock_chat_service = AsyncMock()
        mock_chat_service.get_for_user = AsyncMock(return_value=mock_chat)

        mock_message_service = AsyncMock()
        mock_message_service.list_messages = AsyncMock(return_value=([mock_msg], 1))

        service = export_service_class(
            session=mock_session,
            chat_service=mock_chat_service,
            message_service=mock_message_service,
        )

        # Act
        await service.export_markdown(chat_id, user_id, include_sources=False)

        # Assert - No source queries should be executed
        mock_session.execute.assert_not_called()


class TestExportServiceSourcesInOutput:
    """Tests for source inclusion in markdown output."""

    @pytest.mark.asyncio
    async def test_sources_included_in_markdown_output(
        self, export_service_class, message_role
    ):
        """Test that sources appear in the markdown output."""
        # Arrange
        chat_id = uuid4()
        user_id = "test-user"
        msg_id = uuid4()

        mock_session = AsyncMock()

        mock_chat = MagicMock()
        mock_chat.id = chat_id
        mock_chat.title = "Test Research"
        mock_chat.created_at = None

        mock_msg = MagicMock()
        mock_msg.id = msg_id
        mock_msg.role = message_role.AGENT
        mock_msg.content = "Research findings."

        mock_chat_service = AsyncMock()
        mock_chat_service.get_for_user = AsyncMock(return_value=mock_chat)

        mock_message_service = AsyncMock()
        mock_message_service.list_messages = AsyncMock(return_value=([mock_msg], 1))

        # Mock sources returned by batch loading
        source = MagicMock()
        source.title = "Important Paper"
        source.url = "https://arxiv.org/paper"

        mock_result = MagicMock()
        mock_result.all.return_value = [(source, msg_id)]
        mock_session.execute.return_value = mock_result

        service = export_service_class(
            session=mock_session,
            chat_service=mock_chat_service,
            message_service=mock_message_service,
        )

        # Act
        markdown = await service.export_markdown(chat_id, user_id, include_sources=True)

        # Assert
        assert "#### Sources" in markdown
        assert "[Important Paper](https://arxiv.org/paper)" in markdown


class TestBodyHasSourcesSection:
    """Unit tests for the ``_body_has_sources_section`` double-append guard."""

    def test_detects_sources_heading(self):
        from deep_research.services.export_service import _body_has_sources_section

        body = "Report text [0].\n\n## Sources\n\n[0] [A](https://a.com)\n"
        assert _body_has_sources_section(body) is True

    def test_detects_heading_any_level(self):
        from deep_research.services.export_service import _body_has_sources_section

        assert _body_has_sources_section("Body\n\n### Sources\n") is True

    def test_no_false_positive_on_inline_word(self):
        from deep_research.services.export_service import _body_has_sources_section

        assert _body_has_sources_section("We consulted several sources inline.") is False

    def test_empty_or_none(self):
        from deep_research.services.export_service import _body_has_sources_section

        assert _body_has_sources_section(None) is False
        assert _body_has_sources_section("") is False


class TestExportReportSourcesGuard:
    """End-to-end guard: export must not double-append ``## Sources`` when the
    report body already carries a synthesizer-rendered section."""

    @pytest.fixture
    def export_service_class(self):
        from deep_research.services.export_service import ExportService

        return ExportService

    def _make_session_obj(self):
        session = MagicMock()
        session.query = "What is revenue?"
        session.research_depth = None
        session.id = uuid4()
        return session

    def _execute_returns(self, mock_session, session_obj, sources):
        """Wire two sequential ``execute`` calls: ResearchSession then Source."""
        session_result = MagicMock()
        session_result.scalar_one_or_none.return_value = session_obj
        sources_result = MagicMock()
        sources_scalars = MagicMock()
        sources_scalars.all.return_value = sources
        sources_result.scalars.return_value = sources_scalars
        mock_session.execute = AsyncMock(side_effect=[session_result, sources_result])

    @pytest.mark.asyncio
    async def test_no_double_append_when_body_has_sources(self, export_service_class):
        mock_session = AsyncMock()
        message = MagicMock()
        message.content = (
            "Revenue was $4.2B [0].\n\n## Sources\n\n"
            "### Cited\n\n[0] [Body Source](https://body.com)\n"
        )

        db_source = MagicMock()
        db_source.title = "DB Source"
        db_source.url = "https://db.com"

        service = export_service_class(session=mock_session)
        service._get_message_with_auth = AsyncMock(return_value=message)
        self._execute_returns(mock_session, self._make_session_obj(), [db_source])

        markdown = await service.export_report_markdown(uuid4(), "user-1")

        # Exactly one "## Sources" heading (the body's). The DB-driven one is
        # skipped, so the DB source title must NOT appear.
        assert markdown.count("## Sources") == 1
        assert "Body Source" in markdown
        assert "DB Source" not in markdown

    @pytest.mark.asyncio
    async def test_appends_when_body_lacks_sources(self, export_service_class):
        mock_session = AsyncMock()
        message = MagicMock()
        message.content = "Revenue was $4.2B [0]. (no sources section in body)"

        db_source = MagicMock()
        db_source.title = "DB Source"
        db_source.url = "https://db.com"

        service = export_service_class(session=mock_session)
        service._get_message_with_auth = AsyncMock(return_value=message)
        self._execute_returns(mock_session, self._make_session_obj(), [db_source])

        markdown = await service.export_report_markdown(uuid4(), "user-1")

        # Body had none -> export appends its DB-driven section.
        assert markdown.count("## Sources") == 1
        assert "DB Source" in markdown


class TestMarkdownToHtml:
    """Unit tests for the ``_markdown_to_html`` helper (skipped if markdown absent)."""

    def test_renders_headings_and_links(self):
        pytest.importorskip("markdown")
        from deep_research.services.export_service import _markdown_to_html

        md = "# Title\n\nBody text [0].\n\n## Sources\n\n[0] [A](https://a.com)\n"
        html = _markdown_to_html(md, title="My Report")
        assert "<!DOCTYPE html>" in html
        assert "<title>My Report</title>" in html
        assert "<h1>Title</h1>" in html
        assert 'href="https://a.com"' in html
        # Citation marker survives as plain text.
        assert "[0]" in html

    def test_escapes_title(self):
        pytest.importorskip("markdown")
        from deep_research.services.export_service import _markdown_to_html

        html = _markdown_to_html("body", title="A <script> & B")
        assert "<script>" not in html.split("<body>")[0]
        assert "&lt;script&gt;" in html

    def test_raises_dependency_error_when_markdown_missing(self, monkeypatch):
        """When the markdown lib is absent, the helper raises (not crashes)."""
        import builtins

        from deep_research.services.export_service import (
            ExportDependencyError,
            _markdown_to_html,
        )

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "markdown":
                raise ImportError("No module named 'markdown'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        with pytest.raises(ExportDependencyError, match="markdown"):
            _markdown_to_html("# x", title="t")


class TestExportPdfDocx:
    """PDF/DOCX export: round-trip when libs present, graceful degrade when not."""

    def _service_with_report(self, export_service_class, body="# Report\n\nText [0]."):
        mock_session = AsyncMock()
        message = MagicMock()
        message.content = body

        session_obj = MagicMock()
        session_obj.query = "What is revenue?"
        session_obj.research_depth = None
        session_obj.id = uuid4()

        service = export_service_class(session=mock_session)
        service._get_message_with_auth = AsyncMock(return_value=message)

        # export_report_markdown executes: ResearchSession lookup, then Source
        # lookup; _report_title executes a third ResearchSession lookup.
        def _session_result():
            r = MagicMock()
            r.scalar_one_or_none.return_value = session_obj
            return r

        def _sources_result():
            r = MagicMock()
            scalars = MagicMock()
            scalars.all.return_value = []
            r.scalars.return_value = scalars
            return r

        mock_session.execute = AsyncMock(
            side_effect=[_session_result(), _sources_result(), _session_result()]
        )
        return service

    @pytest.fixture
    def export_service_class(self):
        from deep_research.services.export_service import ExportService

        return ExportService

    @pytest.mark.asyncio
    async def test_export_pdf_roundtrip_or_degrade(self, export_service_class):
        from deep_research.services.export_service import ExportDependencyError

        service = self._service_with_report(export_service_class)
        try:
            import markdown  # noqa: F401
            import weasyprint  # noqa: F401

            have_pdf = True
        except ImportError:
            have_pdf = False

        if have_pdf:
            pdf = await service.export_pdf(uuid4(), "user-1")
            assert isinstance(pdf, bytes)
            assert pdf[:4] == b"%PDF"
        else:
            with pytest.raises(ExportDependencyError):
                await service.export_pdf(uuid4(), "user-1")

    @pytest.mark.asyncio
    async def test_export_docx_roundtrip_or_degrade(self, export_service_class):
        from deep_research.services.export_service import ExportDependencyError

        service = self._service_with_report(export_service_class)
        try:
            import docx  # noqa: F401
            import htmldocx  # noqa: F401
            import markdown  # noqa: F401

            have_docx = True
        except ImportError:
            have_docx = False

        if have_docx:
            docx_bytes = await service.export_docx(uuid4(), "user-1")
            assert isinstance(docx_bytes, bytes)
            # A .docx is a ZIP container -> PK magic bytes.
            assert docx_bytes[:2] == b"PK"
        else:
            with pytest.raises(ExportDependencyError):
                await service.export_docx(uuid4(), "user-1")

    @pytest.mark.asyncio
    async def test_export_pdf_degrades_when_weasyprint_missing(
        self, export_service_class, monkeypatch
    ):
        """Even with markdown present, a missing PDF backend degrades cleanly."""
        pytest.importorskip("markdown")
        import builtins

        from deep_research.services.export_service import ExportDependencyError

        service = self._service_with_report(export_service_class)
        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "weasyprint":
                raise ImportError("No module named 'weasyprint'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        with pytest.raises(ExportDependencyError, match="weasyprint"):
            await service.export_pdf(uuid4(), "user-1")

    @pytest.mark.asyncio
    async def test_export_pdf_inherits_sources_dedup(self, export_service_class):
        """PDF export must not double-append Sources when body already has one."""
        pytest.importorskip("markdown")
        body = (
            "Revenue was $4.2B [0].\n\n## Sources\n\n[0] [Body Source](https://body.com)\n"
        )
        service = self._service_with_report(export_service_class, body=body)
        # Capture the markdown the PDF is built from.
        captured = {}
        orig = service.export_report_markdown

        async def _wrap(message_id, user_id):
            md = await orig(message_id, user_id)
            captured["md"] = md
            return md

        service.export_report_markdown = _wrap  # type: ignore[method-assign]
        # weasyprint may be absent; we only assert on the captured markdown.
        with contextlib.suppress(Exception):
            await service.export_pdf(uuid4(), "user-1")
        assert captured["md"].count("## Sources") == 1
