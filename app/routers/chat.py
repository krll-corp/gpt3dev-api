"""Chat completions endpoint."""
from __future__ import annotations

from fastapi import APIRouter

from ..core.errors import feature_not_available
from ..schemas.chat import ChatCompletionRequest, ChatCompletionResponse

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(payload: ChatCompletionRequest) -> ChatCompletionResponse:
    """Return a structured error while chat completions are disabled."""

    raise feature_not_available(
        "chat_completions",
        "Chat completions are currently disabled; please use /v1/completions instead.",
    )
