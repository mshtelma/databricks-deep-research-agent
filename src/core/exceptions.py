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
        response = {
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


class RateLimitError(AppException):
    """Rate limit exceeded error."""

    def __init__(self, retry_after: int = 60):
        super().__init__(
            message="Rate limit exceeded",
            code="RATE_LIMITED",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details={"retry_after": retry_after},
        )
        self.retry_after = retry_after


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
