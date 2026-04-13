"""Security response headers middleware.

Injects standard security headers on every HTTP response:
- X-Content-Type-Options: nosniff
- X-Frame-Options: DENY
- X-XSS-Protection: 0  (disable legacy auditor; CSP is the modern defence)
- Referrer-Policy: strict-origin-when-cross-origin
- Content-Security-Policy (configurable, optional)
"""

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class SecurityHeadersMiddleware:
    """Append security headers to every HTTP response."""

    def __init__(
        self,
        app: ASGIApp,
        csp_policy: str | None = None,
        report_only: bool = False,
        enable_hsts: bool = False,
    ) -> None:
        self.app = app
        self.csp_policy = csp_policy
        self._csp_header_name = (
            "Content-Security-Policy-Report-Only" if report_only else "Content-Security-Policy"
        )
        self._enable_hsts = enable_hsts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers.append("X-Content-Type-Options", "nosniff")
                headers.append("X-Frame-Options", "DENY")
                headers.append("X-XSS-Protection", "0")
                headers.append("Referrer-Policy", "strict-origin-when-cross-origin")
                if self.csp_policy:
                    headers.append(self._csp_header_name, self.csp_policy)
                if self._enable_hsts:
                    headers.append(
                        "Strict-Transport-Security",
                        "max-age=31536000; includeSubDomains",
                    )
            await send(message)

        await self.app(scope, receive, send_wrapper)
