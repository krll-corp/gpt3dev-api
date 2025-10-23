"""Common Pydantic schemas used across endpoints."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ErrorResponse(BaseModel):
    class Error(BaseModel):
        message: str
        type: str
        param: Optional[str] = None
        code: Optional[str] = None

    error: Error
