# MCP SSRF hardening

`MCPToolset` accepts a remote `url=` argument. Without validation this
is a classic Server-Side Request Forgery (SSRF) vector — a caller can
point the transport at internal services (loopback, link-local IMDS,
RFC1918 ranges) or non-HTTP schemes. PR3c hardens the entry point with
a deny-by-default URL validator implemented in
`databricks_deep_research.tools.mcp_security`.

## Threat model

The framework runs inside Databricks workspaces and other cloud
environments where:

- Cloud metadata services (IMDS) live at `169.254.169.254` and expose
  short-lived credentials.
- Lakebase, internal DBs, and service meshes use private RFC1918
  address space.
- Operators may configure the allowlist conservatively, but cannot
  predict every internal IP.

A malicious or compromised MCP URL — supplied by an end user, an
agent prompt, or a misconfigured config file — must not be able to:

1. Reach loopback/private/link-local services.
2. Pivot via DNS rebinding (validation-time public, request-time
   private).
3. Use non-`http(s)` schemes such as `file://`, `gopher://`, `ftp://`,
   `ws://`, `javascript:`.
4. Pivot via cross-origin HTTP redirects.

## Current callers (verified during PR3c planning)

- `databricks_deep_research/api/__init__.py:68` — re-export only.
- `databricks_deep_research/tests/unit/tools/test_mcp_auth.py:59` —
  `client=fake`.
- `databricks_deep_research/tests/unit/tools/test_mcp_discovery.py` —
  `client=fake` or expected-to-raise.
- `databricks_deep_research/tests/unit/tools/test_mcp_image_content.py`
  — `client=fake`.
- `databricks-deep-research-app/` — **zero callers**.

All test callers use `client=...` (no URL) so SSRF validation only
triggers when `url=` is explicit. Existing tests are unaffected by the
validator.

## Policy: deny by default

URLs are rejected with `MCPSecurityError` (a `ValueError` subclass)
unless they pass every check.

### Allowed schemes

| Scheme | Allowed |
|--------|---------|
| `http` | yes |
| `https` | yes |
| anything else | no |

This excludes `file`, `gopher`, `ftp`, `ws`, `wss`, `javascript`,
`data`, etc.

### Blocked IP ranges

| Range | Family | Reason |
|-------|--------|--------|
| `127.0.0.0/8` | v4 | loopback |
| `10.0.0.0/8` | v4 | private (RFC1918) |
| `172.16.0.0/12` | v4 | private (RFC1918) |
| `192.168.0.0/16` | v4 | private (RFC1918) |
| `169.254.0.0/16` | v4 | link-local (IMDS) |
| `100.64.0.0/10` | v4 | CGNAT |
| `0.0.0.0/8` | v4 | "this network" / unspecified |
| `224.0.0.0/4` | v4 | multicast |
| `240.0.0.0/4` | v4 | reserved (class E) |
| `::1/128` | v6 | loopback |
| `fc00::/7` | v6 | unique local (private) |
| `fe80::/10` | v6 | link-local |
| `fec0::/10` | v6 | site-local (deprecated) |
| `ff00::/8` | v6 | multicast |
| `::/128` | v6 | unspecified |
| `::ffff:0:0/96` | v6 | IPv4-mapped IPv6 (catches v4 spelled as v6) |
| `2002::/16` | v6 | 6to4 |

The IPv4-mapped and 6to4 entries are critical: a naive v4-only check
would miss `::ffff:127.0.0.1` and `2002:7f00:1::`, both of which would
let an attacker reach loopback through an IPv6 transport.

## DNS pinning (anti-rebinding)

When the URL contains a hostname (not a literal IP), the validator:

1. Resolves the hostname via `socket.getaddrinfo` (A + AAAA records).
2. Validates every resolved address against the IP-range policy.
3. **Substitutes** the first validated IP literal back into the URL,
   replacing the hostname.

The pinned-IP URL is what the transport receives. Because the URL no
longer carries the original hostname, the transport never re-resolves
it. This eliminates the canonical DNS rebinding attack:

> Validation-time DNS resolves to public 8.8.8.8; attacker rebinds the
> name to 169.254.169.254 between validation and request.

After pinning, a rebind is moot: the transport contacts `8.8.8.8` (the
literal), not `evil.example.com`.

IPv6 pinned addresses are wrapped in brackets (RFC 3986).

## Redirect handling

`validate_mcp_redirect(original, redirect)` rejects any redirect whose
scheme, host, or port differs from the original URL. The function is
exposed as a utility for callers that wish to enforce same-origin on
their own transport — PR3c does not wire it into the MCP SDK because
the third-party SDK does not currently expose a redirect hook.

The pinned-IP architecture provides defence-in-depth even without a
redirect hook: the transport contacts the validated IP directly, so a
302 to `http://169.254.169.254` from an attacker-controlled MCP server
results in the transport opening a connection to the original pinned
IP (which is not the IMDS) and receiving the redirect response. If the
underlying HTTP client is configured with `follow_redirects=False`
(recommended), no redirect pivot is possible.

## Configuration

### Programmatic

```python
from databricks_deep_research.api import MCPToolset

# Default: deny-by-default
MCPToolset(url="https://mcp.example.com/sse")

# Dev escape hatch: explicit allowlist for localhost MCP stub
MCPToolset(url="http://localhost:8888/sse", allowed_hosts=["localhost"])
```

### Environment variable

`DDR_MCP_ALLOWED_HOSTS` is a comma-separated list of hostnames that
bypass the IP-range check. It is consulted only when the
`allowed_hosts` kwarg is not supplied.

```bash
export DDR_MCP_ALLOWED_HOSTS=localhost,127.0.0.1,my-internal-mcp.lan
```

## Dev escape hatch — when and how

The allowlist is intended for local development and integration tests
that point at a stubbed MCP server on `localhost`. When an entry
matches:

- The IP-range check is **skipped entirely**.
- DNS resolution is also skipped — the hostname goes to the transport
  unmodified.

This means an allowlisted hostname is fully trusted. **Do not put
production hostnames in the allowlist** — they are validated normally,
which is what you want for production.

## ADR

### Decision

Adopt a deny-by-default URL validator with pinned-IP DNS substitution
for `MCPToolset` URLs, with an opt-in `allowed_hosts` allowlist as a
dev escape hatch.

### Drivers

1. **Real SSRF risk.** `MCPToolset(url=...)` passes URL straight to
   `ClientSession`. Public API exposes `MCPToolset` via
   `api/__init__.py`.
2. **Zero production callers today.** All test callers use
   `client=fake`, so deny-by-default is backward-compatible.
3. **DNS rebinding is the canonical SSRF defeater.** Pinned-IP
   substitution is the industry-standard mitigation.
4. **Cloud-native deployment context.** IMDS is at
   `169.254.169.254`; default-unsafe is unacceptable.

### Alternatives considered

- **Allow-by-default with denylist.** Rejected: framework runs in
  cloud workspaces where IMDS is reachable. Default-unsafe is not
  acceptable.
- **Trust the MCP SDK to handle SSRF.** Rejected: the SDK is
  third-party and not designed for SSRF defence. The framework owns
  this risk.
- **No DNS pinning, validate hostname only.** Rejected: this leaves
  DNS rebinding open, which is the most-cited cloud SSRF vector.

### Why chosen

1. Smallest blast radius (zero production callers affected).
2. Pinned-IP eliminates the entire class of DNS-rebinding attacks at
   the cost of one `getaddrinfo` per toolset construction.
3. The allowlist provides a clear, narrow escape hatch for legitimate
   local-development use cases.

### Consequences

- Future MCP integrations that require non-public hostnames must add
  them to `allowed_hosts` or `DDR_MCP_ALLOWED_HOSTS`.
- A small DNS lookup latency at toolset construction time (one-shot,
  cached for the lifetime of the toolset).
- IPv6 hostnames are pinned to their resolved address — operators who
  rely on RR-DNS for failover should be aware that the toolset binds
  to the resolved IP for its lifetime. Re-construct the toolset to
  pick up new resolutions.
- `MCPSecurityError` is exported from `databricks_deep_research.api`
  and `databricks_deep_research` for callers that want to handle
  rejection programmatically.
