"""Custom exception classes and error handling."""

from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse


class AppException(Exception):
    """Base application exception."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: dict[str, Any] | None = None,
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)

    def to_response(self) -> dict[str, Any]:
        """Convert to API error response format."""
        response: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.details:
            response["details"] = self.details
        return response


class NotFoundError(AppException):
    """Resource not found error."""

    def __init__(self, resource: str, resource_id: str):
        super().__init__(
            message=f"{resource} not found: {resource_id}",
            code="NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"resource": resource, "id": resource_id},
        )


class ValidationError(AppException):
    """Validation error."""

    def __init__(self, message: str, field: str | None = None):
        details = {"field": field} if field else {}
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details,
        )


class AuthenticationError(AppException):
    """Authentication error."""

    def __init__(self, message: str = "Authentication required"):
        super().__init__(
            message=message,
            code="AUTHENTICATION_ERROR",
            status_code=status.HTTP_401_UNAUTHORIZED,
        )


class AuthorizationError(AppException):
    """Authorization error."""

    def __init__(self, message: str = "Access denied"):
        super().__init__(
            message=message,
            code="AUTHORIZATION_ERROR",
            status_code=status.HTTP_403_FORBIDDEN,
        )


class PermissionDeniedError(AppException):
    """Permission denied error (OBO access required).

    Used when a specific permission (like OBO token) is required
    but not available.
    """

    def __init__(self, message: str = "Permission denied"):
        super().__init__(
            message=message,
            code="PERMISSION_DENIED",
            status_code=status.HTTP_403_FORBIDDEN,
        )


class RateLimitError(AppException):
    """Rate limit exceeded error."""

    def __init__(
        self,
        retry_after: int = 60,
        endpoint: str | None = None,
        checked_endpoints: list[str] | None = None,
    ):
        details: dict[str, Any] = {"retry_after": retry_after}
        if endpoint:
            details["endpoint"] = endpoint
        if checked_endpoints:
            details["checked_endpoints"] = checked_endpoints
        super().__init__(
            message="Rate limit exceeded",
            code="RATE_LIMITED",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=details,
        )
        self.retry_after = retry_after
        self.endpoint = endpoint
        self.checked_endpoints = checked_endpoints or []

    @property
    def endpoint_display(self) -> str:
        """Get a display string for the endpoint(s) involved.

        Returns the specific endpoint if one failed, otherwise lists
        all checked endpoints that were unavailable.
        """
        if self.endpoint:
            return self.endpoint
        if self.checked_endpoints:
            return f"all_unavailable({','.join(self.checked_endpoints)})"
        return "unknown"


class ConflictError(AppException):
    """Resource conflict error."""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            code="CONFLICT",
            status_code=status.HTTP_409_CONFLICT,
        )


class GoneError(AppException):
    """Resource permanently deleted error."""

    def __init__(self, resource: str, resource_id: str):
        super().__init__(
            message=f"{resource} has been permanently deleted: {resource_id}",
            code="GONE",
            status_code=status.HTTP_410_GONE,
            details={"resource": resource, "id": resource_id},
        )


class LLMError(AppException):
    """LLM service error."""

    def __init__(self, message: str, endpoint: str | None = None):
        details = {"endpoint": endpoint} if endpoint else {}
        super().__init__(
            message=message,
            code="LLM_ERROR",
            status_code=status.HTTP_502_BAD_GATEWAY,
            details=details,
        )


class ExternalServiceError(AppException):
    """External service error (Brave Search, etc.)."""

    def __init__(self, service: str, message: str):
        super().__init__(
            message=f"{service} error: {message}",
            code="EXTERNAL_SERVICE_ERROR",
            status_code=status.HTTP_502_BAD_GATEWAY,
            details={"service": service},
        )


# ---------------------------------------------------------------------------
# OBO (On-Behalf-Of) Authentication Exceptions
# ---------------------------------------------------------------------------


class OBOError(AppException):
    """Base exception for OBO authentication errors.

    OBO (On-Behalf-Of) authentication allows the application to access
    Databricks resources with the user's permissions. These exceptions
    provide user-friendly error messages with remediation steps.
    """

    def __init__(
        self,
        message: str,
        code: str = "OBO_ERROR",
        status_code: int = status.HTTP_403_FORBIDDEN,
        details: dict[str, Any] | None = None,
        remediation: str | None = None,
    ):
        self.remediation = remediation
        super().__init__(
            message=message,
            code=code,
            status_code=status_code,
            details=details or {},
        )
        if remediation:
            self.details["remediation"] = remediation

    def to_response(self) -> dict[str, Any]:
        """Convert to API error response format with remediation."""
        response = super().to_response()
        if self.remediation:
            response["remediation"] = self.remediation
        return response


class OBOPermissionError(OBOError):
    """User lacks permission to access a resource via OBO.

    Raised when the user's token is valid but they don't have
    the required permissions on the target resource.
    """

    def __init__(
        self,
        resource_type: str,
        resource_name: str,
        required_permission: str | None = None,
    ):
        remediation_steps = [
            f"Verify you have access to the {resource_type} '{resource_name}'",
            "Contact your workspace administrator to request access",
        ]
        if required_permission:
            remediation_steps.insert(
                1, f"Required permission: {required_permission}"
            )

        super().__init__(
            message=(
                f"Permission denied: You don't have access to {resource_type} "
                f"'{resource_name}'."
            ),
            code="OBO_PERMISSION_DENIED",
            status_code=status.HTTP_403_FORBIDDEN,
            details={
                "resource_type": resource_type,
                "resource_name": resource_name,
                "required_permission": required_permission,
            },
            remediation="\n".join(f"- {step}" for step in remediation_steps),
        )
        self.resource_type = resource_type
        self.resource_name = resource_name
        self.required_permission = required_permission


class OBOTokenExpiredError(OBOError):
    """User's OBO token has expired.

    OAuth tokens have a limited lifetime. This error indicates
    the token needs to be refreshed.
    """

    def __init__(self, token_type: str = "access_token"):
        super().__init__(
            message="Your authentication session has expired.",
            code="OBO_TOKEN_EXPIRED",
            status_code=status.HTTP_401_UNAUTHORIZED,
            details={"token_type": token_type},
            remediation=(
                "- Refresh the page to obtain a new authentication token\n"
                "- If the issue persists, try logging out and back in\n"
                "- Contact support if you continue to experience issues"
            ),
        )
        self.token_type = token_type


class OBOResourceNotFoundError(OBOError):
    """Resource not found when accessed via OBO.

    The resource may have been deleted, renamed, or the user
    may be looking at the wrong workspace.
    """

    def __init__(
        self,
        resource_type: str,
        resource_name: str,
        workspace_hint: str | None = None,
    ):
        remediation_steps = [
            f"Verify the {resource_type} name is correct: '{resource_name}'",
            f"Check that the {resource_type} exists in your workspace",
            f"The {resource_type} may have been deleted or renamed",
        ]
        if workspace_hint:
            remediation_steps.append(f"Workspace: {workspace_hint}")

        super().__init__(
            message=f"{resource_type} not found: '{resource_name}'",
            code="OBO_RESOURCE_NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
            details={
                "resource_type": resource_type,
                "resource_name": resource_name,
                "workspace_hint": workspace_hint,
            },
            remediation="\n".join(f"- {step}" for step in remediation_steps),
        )
        self.resource_type = resource_type
        self.resource_name = resource_name


class OBOTokenMissingError(OBOError):
    """OBO token required but not provided.

    Enterprise data sources require OBO authentication. This error
    indicates the request is missing the required user token.
    """

    def __init__(self, source_name: str | None = None):
        details: dict[str, Any] = {}
        if source_name:
            details["source_name"] = source_name

        super().__init__(
            message="User authentication required for this data source.",
            code="OBO_TOKEN_MISSING",
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details,
            remediation=(
                "- Ensure you are logged in to Databricks\n"
                "- The x-forwarded-access-token header must be present\n"
                "- If running locally, ensure your environment is configured "
                "for OBO authentication"
            ),
        )
        self.source_name = source_name


class OBOServiceUnavailableError(OBOError):
    """OBO service is temporarily unavailable.

    The underlying Databricks service (Vector Search, Genie, etc.)
    may be experiencing issues or undergoing maintenance.
    """

    def __init__(
        self,
        service_name: str,
        retry_after_seconds: int | None = None,
    ):
        details: dict[str, Any] = {"service_name": service_name}
        if retry_after_seconds:
            details["retry_after_seconds"] = retry_after_seconds

        remediation_parts = [
            f"The {service_name} service is temporarily unavailable",
            "Try again in a few minutes",
            "Check Databricks status page for service updates",
        ]
        if retry_after_seconds:
            remediation_parts[1] = f"Try again after {retry_after_seconds} seconds"

        super().__init__(
            message=f"{service_name} service is temporarily unavailable.",
            code="OBO_SERVICE_UNAVAILABLE",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details,
            remediation="\n".join(f"- {step}" for step in remediation_parts),
        )
        self.service_name = service_name
        self.retry_after_seconds = retry_after_seconds


class StructuredSynthesisError(Exception):
    """Raised when structured synthesis fails and needs streaming fallback.

    This exception signals to the orchestrator that structured output generation
    failed (e.g., validation errors like max_length exceeded) and the system
    should fall back to streaming synthesis to emit SSE chunks.
    """

    def __init__(self, message: str, state: Any):
        super().__init__(message)
        self.state = state


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Handle AppException and return JSON response."""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_response(),
        headers={"Retry-After": str(exc.retry_after)} if isinstance(exc, RateLimitError) else None,
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTPException and return consistent JSON response."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": "HTTP_ERROR",
            "message": exc.detail,
        },
    )
