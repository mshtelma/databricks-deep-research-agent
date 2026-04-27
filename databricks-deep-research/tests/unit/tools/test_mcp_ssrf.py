"""Unit tests for the SSRF defenses in :mod:`databricks_deep_research.tools.mcp_security`.

All tests run without real network access. DNS-dependent tests
monkey-patch ``socket.getaddrinfo`` via ``unittest.mock.patch``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from databricks_deep_research.tools.mcp_security import (
    MCPSecurityError,
    is_allowed_host,
    validate_mcp_redirect,
    validate_mcp_url,
)


def _addrinfo(ip: str) -> list[tuple[int, int, int, str, tuple[str, int]]]:
    """Build a fake ``getaddrinfo`` return list for a single IP literal."""
    # Minimal shape: getaddrinfo returns
    # ``[(family, type, proto, canonname, sockaddr)]``. The validator
    # only reads ``sockaddr[0]``.
    return [(2, 1, 0, "", (ip, 0))]


# --- Literal-IP tests (no DNS involved) -------------------------------


def test_blocks_loopback_v4() -> None:
    with pytest.raises(MCPSecurityError, match="loopback"):
        validate_mcp_url("http://127.0.0.1/x")


def test_blocks_loopback_v6() -> None:
    with pytest.raises(MCPSecurityError, match="loopback"):
        validate_mcp_url("https://[::1]/x")


def test_blocks_link_local_v4() -> None:
    # IMDS lives at 169.254.169.254 — must always be blocked.
    with pytest.raises(MCPSecurityError, match="link_local"):
        validate_mcp_url("http://169.254.169.254/latest/meta-data/")


def test_blocks_link_local_v6() -> None:
    with pytest.raises(MCPSecurityError, match="link_local"):
        validate_mcp_url("https://[fe80::1]/x")


@pytest.mark.parametrize(
    "host",
    [
        "10.0.0.5",
        "172.16.0.5",
        "192.168.0.5",
    ],
)
def test_blocks_private_v4(host: str) -> None:
    with pytest.raises(MCPSecurityError):
        validate_mcp_url(f"http://{host}/x")


def test_blocks_cgnat() -> None:
    with pytest.raises(MCPSecurityError, match="cgnat"):
        validate_mcp_url("http://100.64.0.5/x")


def test_blocks_ipv4_mapped_v6() -> None:
    # ``::ffff:127.0.0.1`` is the IPv4-mapped IPv6 spelling of 127.0.0.1.
    # A naive v4-only check would miss it. Our validator must catch it.
    with pytest.raises(MCPSecurityError, match="ipv4_mapped_v6|loopback"):
        validate_mcp_url("https://[::ffff:127.0.0.1]/x")


def test_blocks_6to4() -> None:
    with pytest.raises(MCPSecurityError, match="6to4"):
        validate_mcp_url("https://[2002::1]/x")


def test_blocks_zero_addr() -> None:
    with pytest.raises(MCPSecurityError, match="unspecified"):
        validate_mcp_url("http://0.0.0.0/x")


def test_blocks_v6_unspecified() -> None:
    with pytest.raises(MCPSecurityError, match="unspecified"):
        validate_mcp_url("https://[::]/x")


# --- Scheme tests -----------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "gopher://example.com/x",
        "ftp://example.com/x",
        "ws://example.com/x",
        "javascript:alert(1)",
    ],
)
def test_blocks_unsupported_scheme(url: str) -> None:
    with pytest.raises(MCPSecurityError, match="scheme"):
        validate_mcp_url(url)


# --- Hostname / DNS tests ---------------------------------------------


def test_allows_https_public_ip() -> None:
    """Public DNS resolution succeeds and returns a pinned-IP URL."""
    with patch(
        "databricks_deep_research.tools.mcp_security.socket.getaddrinfo",
        return_value=_addrinfo("8.8.8.8"),
    ):
        result = validate_mcp_url("https://example.com/x")
    # Hostname has been replaced with the IP literal.
    assert "8.8.8.8" in result
    assert "example.com" not in result
    assert result.startswith("https://")
    assert result.endswith("/x")


def test_dns_resolution_to_private_blocked(caplog: pytest.LogCaptureFixture) -> None:
    with patch(
        "databricks_deep_research.tools.mcp_security.socket.getaddrinfo",
        return_value=_addrinfo("10.0.0.5"),
    ), pytest.raises(MCPSecurityError, match="resolves to blocked address"):
        validate_mcp_url("http://evil.example.com/x")
    # Structured log carries the dns: prefix so operators can filter.
    assert any("dns:" in rec.message for rec in caplog.records)


def test_dns_rebind_blocked() -> None:
    """Pinned-IP architecture neutralises DNS rebinding.

    The validator resolves once at validation time. The returned URL
    contains the resolved IP literal — there is no hostname for any
    later code to re-resolve. A subsequent DNS rebind to an internal
    address is therefore irrelevant: the transport will only contact
    the validation-time IP.
    """
    with patch(
        "databricks_deep_research.tools.mcp_security.socket.getaddrinfo",
        return_value=_addrinfo("8.8.8.8"),
    ):
        result = validate_mcp_url("https://maybe-evil.example.com/x")
    # The URL no longer carries the hostname; subsequent DNS resolution
    # of "maybe-evil.example.com" by an attacker is moot.
    assert "8.8.8.8" in result
    assert "maybe-evil.example.com" not in result


def test_allowed_hosts_overrides_block() -> None:
    """Hostnames in the allowlist skip IP-range checks entirely."""
    # No DNS mock needed — the allowlist short-circuits before resolution.
    result = validate_mcp_url(
        "http://localhost:8888/x", allowed_hosts=["localhost"]
    )
    assert result == "http://localhost:8888/x"


def test_allowed_hosts_case_insensitive() -> None:
    result = validate_mcp_url(
        "http://Localhost/x", allowed_hosts=["LOCALHOST"]
    )
    assert "localhost" in result.lower()


# --- Redirect tests ---------------------------------------------------


def test_redirect_cross_host_blocked() -> None:
    with pytest.raises(MCPSecurityError, match="host mismatch"):
        validate_mcp_redirect(
            "https://example.com/a", "https://other.example/b"
        )


def test_redirect_cross_scheme_blocked() -> None:
    with pytest.raises(MCPSecurityError, match="scheme mismatch"):
        validate_mcp_redirect(
            "https://example.com/a", "http://example.com/a"
        )


def test_redirect_cross_port_blocked() -> None:
    with pytest.raises(MCPSecurityError, match="port mismatch"):
        validate_mcp_redirect(
            "https://example.com:443/a", "https://example.com:8443/a"
        )


def test_redirect_same_host_allowed() -> None:
    # Should not raise.
    validate_mcp_redirect("https://example.com/a", "https://example.com/b")


# --- Empty / malformed -------------------------------------------------


def test_blocks_empty_host() -> None:
    with pytest.raises(MCPSecurityError, match="host"):
        validate_mcp_url("http:///x")


def test_dns_resolution_failure_blocked() -> None:
    import socket as _socket

    def _raise(*args: object, **kwargs: object) -> None:
        raise _socket.gaierror("no such host")

    with patch(
        "databricks_deep_research.tools.mcp_security.socket.getaddrinfo",
        side_effect=_raise,
    ), pytest.raises(MCPSecurityError, match="DNS resolution failed"):
        validate_mcp_url("http://nonexistent.invalid/x")


# --- is_allowed_host helper -------------------------------------------


def test_is_allowed_host_none() -> None:
    assert is_allowed_host("anything", None) is False


def test_is_allowed_host_empty() -> None:
    assert is_allowed_host("anything", []) is False


def test_is_allowed_host_match() -> None:
    assert is_allowed_host("localhost", ["LOCALHOST", "127.0.0.1"]) is True


def test_is_allowed_host_no_match() -> None:
    assert is_allowed_host("evil.com", ["good.com"]) is False
