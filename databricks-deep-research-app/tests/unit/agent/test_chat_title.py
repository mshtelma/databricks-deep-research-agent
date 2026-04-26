"""Unit tests for the canonical chat-title derivation helper."""

from __future__ import annotations

from deep_research.agent.chat_title import derive_chat_title_from_query


class TestDeriveChatTitleFromQuery:
    def test_short_input(self) -> None:
        assert derive_chat_title_from_query("Hello") == "Hello"

    def test_returns_empty_string_for_none(self) -> None:
        assert derive_chat_title_from_query(None) == ""

    def test_returns_empty_string_for_empty(self) -> None:
        assert derive_chat_title_from_query("") == ""

    def test_returns_empty_string_for_whitespace(self) -> None:
        assert derive_chat_title_from_query("   ") == ""
        assert derive_chat_title_from_query("\t\n  ") == ""

    def test_strips_leading_and_trailing_whitespace(self) -> None:
        assert derive_chat_title_from_query("  hello  ") == "hello"

    def test_threshold_boundary_exactly_50_chars(self) -> None:
        s = "x" * 50
        assert derive_chat_title_from_query(s) == s

    def test_threshold_boundary_51_chars_truncates(self) -> None:
        s = "x" * 51
        result = derive_chat_title_from_query(s)
        assert result == "x" * 47 + "..."
        assert len(result) == 50

    def test_long_input_uses_first_47_chars_plus_ellipsis(self) -> None:
        s = "A" * 100
        result = derive_chat_title_from_query(s)
        assert result == "A" * 47 + "..."
        assert len(result) == 50

    def test_idempotent_on_short_input(self) -> None:
        once = derive_chat_title_from_query("short")
        twice = derive_chat_title_from_query(once)
        assert once == twice == "short"

    def test_idempotent_on_already_truncated(self) -> None:
        once = derive_chat_title_from_query("x" * 100)
        # "xxxxx...xxx..." is 50 chars, which equals _MAX_RAW_LENGTH (50),
        # so the second call should return the 50-char input raw (not re-truncate).
        twice = derive_chat_title_from_query(once)
        assert once == twice
        assert len(once) == 50

    def test_matches_orchestrator_inline_rule(self) -> None:
        """Byte-for-byte parity with framework_orchestrator.py:841 expression:
        chat_title = query[:47] + "..." if len(query) > 50 else query
        """

        def orchestrator_rule(query: str) -> str:
            # Exactly the expression used at framework_orchestrator.py:841
            return query[:47] + "..." if len(query) > 50 else query

        # Parametrize over a range of inputs -- only assert for non-empty,
        # non-whitespace inputs (the helper adds a strip/guard layer that the
        # inline rule lacks; that is intentional defensive behavior).
        for length in (1, 10, 49, 50, 51, 60, 100):
            probe = "a" * length
            assert derive_chat_title_from_query(probe) == orchestrator_rule(probe)

    def test_handles_multibyte_unicode(self) -> None:
        # 'é' is a single char in Python; len counts codepoints, not bytes.
        probe = "é" * 60
        result = derive_chat_title_from_query(probe)
        assert result == "é" * 47 + "..."
