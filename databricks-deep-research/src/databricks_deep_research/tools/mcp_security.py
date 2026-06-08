"""SSRF defenses for ``MCPToolset``.

This module provides URL canonicalization, deny-by-default IP-range
validation, pinned-IP DNS substitution (to defeat DNS rebinding), and a
same-origin redirect validator. It is invoked from
:class:`databricks_deep_research.tools.mcp.MCPToolset` whenever a caller
constructs a toolset with an explicit ``url=`` argument.

See ``docs/security/mcp-ssrf.md`` for the full threat model, ADR, and
configuration reference.

Public surface (re-exported via ``databricks_deep_research.api``):

- :class:`MCPSecurityError` — raised on every rejected URL/redirect.
- :func:`validate_mcp_url` — main entry point; returns the URL with the
  hostname replaced by a validated IP literal (pinned-IP).
- :func:`validate_mcp_redirect` — utility for cross-origin redirect
  rejection. Currently exposed for callers that wrap their own transport;
  there is no MCP transport hook in this PR.
- :func:`is_allowed_host` — case-insensitive allowlist match used as the
  dev escape hatch for ``localhost`` etc.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)


class MCPSecurityError(ValueError):
    """Raised when an MCP URL or redirect target violates SSRF policy."""


_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})

# Explicit blocked nets. Many of these are also caught by the
# ``is_private`` / ``is_loopback`` / etc. attributes on
# :class:`ipaddress.IPv4Address`, but enumerating them gives precise
# rejection reasons in logs and protects against future stdlib drift.
_BLOCKED_NETS_V4: tuple[ipaddress.IPv4Network, ...] = (
    ipaddress.IPv4Network("127.0.0.0/8"),  # loopback
    ipaddress.IPv4Network("10.0.0.0/8"),  # RFC1918 private
    ipaddress.IPv4Network("172.16.0.0/12"),  # RFC1918 private
    ipaddress.IPv4Network("192.168.0.0/16"),  # RFC1918 private
    ipaddress.IPv4Network("169.254.0.0/16"),  # link-local (IMDS lives here)
    ipaddress.IPv4Network("100.64.0.0/10"),  # CGNAT
    ipaddress.IPv4Network("0.0.0.0/8"),  # "this network" / unspecified
    ipaddress.IPv4Network("224.0.0.0/4"),  # multicast
    ipaddress.IPv4Network("240.0.0.0/4"),  # reserved (class E)
)

_BLOCKED_NETS_V6: tuple[ipaddress.IPv6Network, ...] = (
    ipaddress.IPv6Network("::1/128"),  # loopback
    ipaddress.IPv6Network("fc00::/7"),  # ULA (private)
    ipaddress.IPv6Network("fe80::/10"),  # link-local
    ipaddress.IPv6Network("fec0::/10"),  # site-local (deprecated)
    ipaddress.IPv6Network("ff00::/8"),  # multicast
    ipaddress.IPv6Network("::/128"),  # unspecified
    ipaddress.IPv6Network("::ffff:0:0/96"),  # IPv4-mapped IPv6
    ipaddress.IPv6Network("2002::/16"),  # 6to4
)


def _is_blocked_ip(ip_str: str) -> tuple[bool, str]:
    """Return ``(blocked, reason)`` for an IP literal.

    The reason string is a short slug suitable for structured logging
    (e.g. ``"loopback"``, ``"link_local"``, ``"cgnat"``,
    ``"ipv4_mapped_v6"``, ``"6to4"``, ``"unspecified"``,
    ``"private"``, ``"reserved"``, ``"multicast"``).
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True, "invalid_ip"

    # Special-case the unspecified addresses up-front for clear
    # reporting; they would also be caught by the explicit nets.
    if isinstance(ip, ipaddress.IPv4Address) and str(ip) == "0.0.0.0":
        return True, "unspecified"
    if isinstance(ip, ipaddress.IPv6Address) and str(ip) == "::":
        return True, "unspecified"

    # Explicit nets first so we can attribute precise reasons (cgnat,
    # ipv4_mapped_v6, 6to4) before the broader stdlib classifiers
    # short-circuit with a less informative reason like "private".
    if isinstance(ip, ipaddress.IPv4Address):
        for net4 in _BLOCKED_NETS_V4:
            if ip in net4:
                if net4.network_address == ipaddress.IPv4Address("100.64.0.0"):
                    return True, "cgnat"
                if net4.network_address == ipaddress.IPv4Address("127.0.0.0"):
                    return True, "loopback"
                if net4.network_address == ipaddress.IPv4Address("169.254.0.0"):
                    return True, "link_local"
                if net4.network_address == ipaddress.IPv4Address("224.0.0.0"):
                    return True, "multicast"
                if net4.network_address == ipaddress.IPv4Address("240.0.0.0"):
                    return True, "reserved"
                if net4.network_address == ipaddress.IPv4Address("0.0.0.0"):
                    return True, "unspecified"
                return True, "private"
    else:
        for net6 in _BLOCKED_NETS_V6:
            if ip in net6:
                if net6.network_address == ipaddress.IPv6Address("::ffff:0:0"):
                    return True, "ipv4_mapped_v6"
                if net6.network_address == ipaddress.IPv6Address("2002::"):
                    return True, "6to4"
                if net6.network_address == ipaddress.IPv6Address("::1"):
                    return True, "loopback"
                if net6.network_address == ipaddress.IPv6Address("fe80::"):
                    return True, "link_local"
                if net6.network_address == ipaddress.IPv6Address("ff00::"):
                    return True, "multicast"
                if net6.network_address == ipaddress.IPv6Address("::"):
                    return True, "unspecified"
                return True, "private"

    # Fall-through: stdlib classifiers catch anything not in our explicit
    # nets (e.g. future RFCs adding new private blocks).
    if ip.is_loopback:
        return True, "loopback"
    if ip.is_link_local:
        return True, "link_local"
    if ip.is_multicast:
        return True, "multicast"
    if ip.is_private:
        return True, "private"
    if ip.is_reserved:
        return True, "reserved"
    if ip.is_unspecified:
        return True, "unspecified"

    return False, ""


def _resolve_dns(hostname: str) -> list[str]:
    """Return the set of resolved A + AAAA IP literals for ``hostname``.

    Wraps :func:`socket.getaddrinfo` so tests can monkey-patch the
    resolver without monkey-patching stdlib globally.
    """
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise MCPSecurityError(f"DNS resolution failed for host: {hostname}") from exc
    seen: list[str] = []
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        raw = sockaddr[0]
        if not isinstance(raw, str) or not raw:
            continue
        if raw not in seen:
            seen.append(raw)
    if not seen:
        raise MCPSecurityError(f"DNS resolution returned no addresses for host: {hostname}")
    return seen


def is_allowed_host(host: str, allowed_hosts: list[str] | None) -> bool:
    """Case-insensitive allowlist match for hostnames.

    When ``allowed_hosts`` is ``None`` or empty the function returns
    ``False``; the caller must apply normal IP-range checks. When the
    host matches an entry the caller skips IP-range checks entirely
    (the dev escape hatch for ``localhost``).
    """
    if not allowed_hosts:
        return False
    needle = host.lower()
    return any(needle == entry.lower() for entry in allowed_hosts)


def _log_block(url: str, reason: str) -> None:
    logger.warning("MCP_URL_BLOCKED url=%s reason=%s", url, reason)


def validate_mcp_url(url: str, allowed_hosts: list[str] | None = None) -> str:
    """Validate ``url`` against SSRF policy and return a pinned-IP URL.

    Steps:

    1. Reject non-``http``/``https`` schemes.
    2. Reject empty hostnames.
    3. If the host is a literal IP, run :func:`_is_blocked_ip`.
    4. Otherwise, if the host is in ``allowed_hosts``, return the URL
       unchanged (no IP check, no DNS pin — the operator opted in).
    5. Otherwise resolve the host via DNS, run :func:`_is_blocked_ip`
       on every resolved address, and return the URL with the hostname
       replaced by the first validated IP literal. This pinned-IP
       substitution defeats DNS rebinding because the transport never
       re-resolves the hostname.

    Raises :class:`MCPSecurityError` on any failure.
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        _log_block(url, f"scheme:{scheme or 'empty'}")
        raise MCPSecurityError(
            f"MCP URL scheme '{scheme or 'empty'}' is not allowed; permitted: "
            f"{sorted(_ALLOWED_SCHEMES)}"
        )

    host = parsed.hostname
    if not host:
        _log_block(url, "empty_host")
        raise MCPSecurityError("MCP URL has no host component")

    # Literal IP path — no DNS, no allowlist semantics.
    try:
        ipaddress.ip_address(host)
        is_literal_ip = True
    except ValueError:
        is_literal_ip = False

    if is_literal_ip:
        blocked, reason = _is_blocked_ip(host)
        if blocked:
            _log_block(url, reason)
            raise MCPSecurityError(
                f"MCP URL host {host!r} is blocked by SSRF policy ({reason})"
            )
        return url

    # Hostname path — apply the allowlist as a hard escape hatch first.
    if is_allowed_host(host, allowed_hosts):
        return url

    resolved = _resolve_dns(host)
    for ip_literal in resolved:
        blocked, reason = _is_blocked_ip(ip_literal)
        if blocked:
            _log_block(url, f"dns:{reason}")
            raise MCPSecurityError(
                f"MCP URL host {host!r} resolves to blocked address "
                f"{ip_literal} ({reason})"
            )

    # Pinned-IP substitution. Wrap IPv6 in brackets per RFC 3986.
    chosen = resolved[0]
    try:
        chosen_ip = ipaddress.ip_address(chosen)
    except ValueError:
        # Defensive: if getaddrinfo handed us something unparseable,
        # fail closed.
        _log_block(url, "pin_unparseable")
        raise MCPSecurityError(
            f"MCP URL pinned address {chosen!r} is not a valid IP literal"
        ) from None
    pinned_host = f"[{chosen_ip}]" if isinstance(chosen_ip, ipaddress.IPv6Address) else str(chosen_ip)

    # urllib.parse splits user/password/host/port into ``netloc``; rebuild
    # it preserving userinfo and port (if present).
    userinfo = ""
    if parsed.username is not None:
        userinfo = parsed.username
        if parsed.password is not None:
            userinfo += f":{parsed.password}"
        userinfo += "@"
    port_part = f":{parsed.port}" if parsed.port is not None else ""
    new_netloc = f"{userinfo}{pinned_host}{port_part}"
    return urlunparse(parsed._replace(netloc=new_netloc))


def validate_mcp_redirect(original_url: str, redirect_url: str) -> None:
    """Reject cross-origin redirects.

    A same-origin redirect (matching scheme, host, and port) is allowed.
    Any difference triggers :class:`MCPSecurityError`. This is exposed
    as a utility for callers that want to enforce same-origin on their
    own transport; PR3c does not wire it into the MCP SDK because the
    SDK does not currently expose a redirect hook.
    """
    a = urlparse(original_url)
    b = urlparse(redirect_url)
    if a.scheme.lower() != b.scheme.lower():
        _log_block(redirect_url, "redirect_scheme_mismatch")
        raise MCPSecurityError(
            f"MCP redirect scheme mismatch: {a.scheme!r} -> {b.scheme!r}"
        )
    a_host = (a.hostname or "").lower()
    b_host = (b.hostname or "").lower()
    if a_host != b_host:
        _log_block(redirect_url, "redirect_host_mismatch")
        raise MCPSecurityError(
            f"MCP redirect host mismatch: {a_host!r} -> {b_host!r}"
        )
    if a.port != b.port:
        _log_block(redirect_url, "redirect_port_mismatch")
        raise MCPSecurityError(
            f"MCP redirect port mismatch: {a.port!r} -> {b.port!r}"
        )


__all__ = [
    "MCPSecurityError",
    "validate_mcp_url",
    "validate_mcp_redirect",
    "is_allowed_host",
]
