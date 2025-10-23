"""Embeddings endpoint stub."""
from __future__ import annotations

from fastapi import APIRouter, Response, status

from ..core.errors import openai_http_error
from ..core.settings import get_settings

router = APIRouter(prefix="/v1", tags=["embeddings"])


@router.post("/embeddings")
async def create_embeddings() -> Response:
    settings = get_settings()
    if not settings.enable_embeddings_backend:
        raise openai_http_error(
            status.HTTP_501_NOT_IMPLEMENTED,
            "Embeddings backend is not configured. Set ENABLE_EMBEDDINGS_BACKEND=1 to enable.",
            error_type="not_implemented_error",
            code="embeddings_backend_unavailable",
        )
    raise openai_http_error(
        status.HTTP_501_NOT_IMPLEMENTED,
        "Embeddings backend configuration is pending implementation.",
        error_type="not_implemented_error",
        code="embeddings_backend_pending",
    )
