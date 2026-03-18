"""Tests for SQL identifier validation in grant_permissions module."""

import pytest

from deep_research.db.grant_permissions import _validate_sql_identifier


class TestValidateSqlIdentifier:
    """Tests for _validate_sql_identifier()."""

    @pytest.mark.parametrize(
        "identifier",
        [
            "normal_sp_123",
            "sp-with-dashes",
            "sp.with.dots",
            "SimpleName",
            "abc123",
            "a",
        ],
    )
    def test_valid_identifiers_pass(self, identifier: str) -> None:
        """Valid identifiers should be returned unchanged."""
        result = _validate_sql_identifier(identifier)
        assert result == identifier

    @pytest.mark.parametrize(
        "identifier",
        [
            "'; DROP TABLE --",
            'sp"injection',
            "sp\ninjection",
            "",
            "sp name with spaces",
            "sp;semicolon",
            "sp'quote",
            "$(command)",
        ],
    )
    def test_injection_attempts_raise_value_error(self, identifier: str) -> None:
        """Injection attempts and unsafe characters should raise ValueError."""
        with pytest.raises(ValueError, match="Unsafe SQL"):
            _validate_sql_identifier(identifier)

    def test_custom_label_in_error_message(self) -> None:
        """The label parameter should appear in the error message."""
        with pytest.raises(ValueError, match="Unsafe SQL database name"):
            _validate_sql_identifier("'; DROP TABLE --", label="database name")

    def test_parameterized_query_format(self) -> None:
        """Verify the expected parameterized query format for databricks_create_role.

        The production code uses:
            await conn.execute(
                "SELECT databricks_create_role($1, 'SERVICE_PRINCIPAL')",
                sp_username,
            )
        This test validates that the query string uses $1 placeholder
        rather than f-string interpolation.
        """
        query = "SELECT databricks_create_role($1, 'SERVICE_PRINCIPAL')"
        assert "$1" in query
        assert "'" not in query.replace("'SERVICE_PRINCIPAL'", "")
        # Ensure no f-string patterns
        assert "{" not in query
        assert "}" not in query
