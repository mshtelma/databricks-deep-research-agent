"""GitHub ref reachability probe for the Deploy Here feature (Section S4a).

Before uploading source to the workspace, we verify that the pinned framework
Git ref actually exists in the upstream repository. A ref may be a tag or a
branch. A 404 for both is the only definitive negative; all other outcomes
(rate-limits, network errors, private repos, non-GitHub URLs) fail-open so we
never block a deploy on a transient probe failure.

The ``http_client`` parameter is injectable so tests can run without real
network calls.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

# Strict regex: only plain ``https://github.com/owner/repo`` (with optional
# ``.git`` suffix and trailing slash).
_GITHUB_URL_RE = re.compile(
    r"^https://github\.com/([^/]+)/([^/.]+)(?:\.git)?/?$"
)


@dataclass(frozen=True)
class TagProbeResult:
    """Result of :func:`probe_framework_tag`.

    Attributes
    ----------
    reachable:
        ``True`` when the ref is confirmed present or when we cannot determine
        its status (fail-open). ``False`` only on definitive HTTP 404s for both
        tag and branch refs.
    error_kind:
        ``"framework_tag_unreachable"`` when ``reachable=False``; ``None``
        otherwise.
    note:
        Human-readable reason for the decision, useful for log triage.
    ref_kind:
        ``"tag"`` when the ref matched ``git/refs/tags/...``; ``"branch"`` when
        it matched ``git/refs/heads/...``; ``None`` when the kind could not be
        determined (fail-open paths, 401/403, network errors). Callers that
        require tag-immutability semantics should reject ``"branch"`` results
        because branches can be force-pushed underneath deployed apps.
    """

    reachable: bool
    error_kind: Literal["framework_tag_unreachable"] | None
    note: str | None
    ref_kind: Literal["tag", "branch"] | None = None


async def probe_framework_tag(
    *,
    git_url: str,
    git_tag: str,
    github_token: str | None = None,
    timeout_seconds: float = 5.0,
    http_client: Any | None = None,
) -> TagProbeResult:
    """Check whether ``git_tag`` exists as a tag or branch in ``git_url``.

    Decision matrix
    ---------------
    * ``200`` → reachable (tag or branch found).
    * tag ``404`` + branch ``404`` → ``framework_tag_unreachable``.
    * ``401``/``403`` → reachable (fail-open), ``note="rate_limited_or_unauthorized"``.
    * ``5xx`` / network error / timeout → reachable (fail-open), ``note="probe_unavailable"``.
    * Non-GitHub URL → reachable (fail-open), ``note="non_github_url_skip"``.

    Parameters
    ----------
    git_url:
        The framework repository URL (e.g. ``"https://github.com/owner/repo"``).
    git_tag:
        The Git ref to verify (for example, ``"main"`` or ``"v0.3.0"``).
    github_token:
        Optional Bearer token for authenticated requests (higher rate limits,
        required for private repos).
    timeout_seconds:
        HTTP request timeout.  Defaults to 5 s — tight enough to not block
        the deploy UI for long but enough for a cold DNS lookup.
    http_client:
        Injectable ``httpx.AsyncClient`` (or compatible mock) for testing.
        When ``None`` a short-lived client is created internally.
    """
    match = _GITHUB_URL_RE.match(git_url)
    if match is None:
        return TagProbeResult(
            reachable=True,
            error_kind=None,
            note="non_github_url_skip",
        )

    owner, repo = match.group(1), match.group(2)
    # Probe order is tag-first so callers that care about tag-vs-branch
    # immutability (see ``deploy_here_require_tag_only`` setting) get the
    # tag answer without needing a second round trip when the ref happens to
    # exist as both.
    api_targets: tuple[tuple[str, Literal["tag", "branch"]], ...] = (
        (f"https://api.github.com/repos/{owner}/{repo}/git/refs/tags/{git_tag}", "tag"),
        (f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{git_tag}", "branch"),
    )
    headers: dict[str, str] = {"Accept": "application/vnd.github+json"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    try:
        # Lazy import so the module can be loaded without httpx installed.
        import httpx  # noqa: PLC0415

        async def _probe_with_client(client: Any) -> TagProbeResult:
            for api_url, ref_kind in api_targets:
                response = await client.get(
                    api_url,
                    headers=headers,
                    timeout=timeout_seconds,
                )
                status = response.status_code
                if status == 200:
                    return TagProbeResult(
                        reachable=True,
                        error_kind=None,
                        note=None,
                        ref_kind=ref_kind,
                    )
                if status == 404:
                    continue
                if status in (401, 403):
                    return TagProbeResult(
                        reachable=True,
                        error_kind=None,
                        note="rate_limited_or_unauthorized",
                    )
                # 5xx or unexpected status — fail open.
                return TagProbeResult(
                    reachable=True,
                    error_kind=None,
                    note=f"probe_unavailable:http_{status}",
                )
            return TagProbeResult(
                reachable=False,
                error_kind="framework_tag_unreachable",
                note=f"ref_not_found:{git_tag}",
            )

        if http_client is not None:
            return await _probe_with_client(http_client)

        async with httpx.AsyncClient() as client:
            return await _probe_with_client(client)

    except Exception:  # noqa: BLE001
        # Network error, DNS failure, timeout, httpx not installed, etc.
        return TagProbeResult(
            reachable=True,
            error_kind=None,
            note="probe_unavailable",
        )
