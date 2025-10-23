"""Helpers for returning OpenAI-compatible error payloads."""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import HTTPException, status


def openai_http_error(
    status_code: int,
    message: str,
    *,
    error_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> HTTPException:
    """Return an ``HTTPException`` matching OpenAI's error schema."""

    error_payload: Dict[str, Any] = {
        "message": message,
        "type": error_type,
        "param": param,
        "code": code,
    }
    return HTTPException(status_code=status_code, detail=error_payload)


def model_not_found(model: str) -> HTTPException:
    """Return a 404 error for missing models."""

    return openai_http_error(
        status.HTTP_404_NOT_FOUND,
        message=f"Model '{model}' does not exist or is not available.",
        error_type="model_not_found",
        param="model",
        code="model_not_found",
    )


def feature_not_available(feature: str, message: str) -> HTTPException:
    """Return a 501 error when a feature is disabled or unsupported."""

    return openai_http_error(
        status.HTTP_501_NOT_IMPLEMENTED,
        message=message,
        error_type="not_implemented_error",
        code=f"{feature}_not_available",
    )
