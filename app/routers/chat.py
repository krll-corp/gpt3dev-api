"""Chat completions endpoint."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Generator, List

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..core import engine
from ..core.errors import model_not_found, openai_http_error
from ..core.model_registry import get_model_spec
from ..schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkChoiceDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)
from ..schemas.common import UsageInfo

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(payload: ChatCompletionRequest) -> ChatCompletionResponse:
    """Generate chat completions using instruct-tuned models."""
    try:
        spec = get_model_spec(payload.model)
    except KeyError:
        raise model_not_found(payload.model)

    if not spec.is_instruct:
        raise openai_http_error(
            400,
            f"Model '{payload.model}' is not an instruct model and cannot be used with chat completions. "
            "Please use /v1/completions instead, or choose an instruct model like 'GPT4-dev-177M-1511-Instruct'.",
            error_type="invalid_request_error",
            param="model",
        )

    # Convert messages to dict format for apply_chat_template
    messages_dict = [{"role": m.role, "content": m.content} for m in payload.messages]
    
    # Apply chat template using tokenizer
    prompt = engine.apply_chat_template(payload.model, messages_dict)
    
    stop_sequences = payload.stop if isinstance(payload.stop, list) else (
        [payload.stop] if payload.stop else []
    )

    if payload.stream:
        return _streaming_chat_completion(payload, prompt, stop_sequences)

    try:
        result = await asyncio.to_thread(
            engine.generate,
            payload.model,
            prompt,
            temperature=payload.temperature,
            top_p=payload.top_p,
            max_tokens=payload.max_tokens,
            stop=stop_sequences,
            n=payload.n,
        )
    except Exception as exc:
        raise openai_http_error(
            500,
            f"Generation error: {exc}",
            error_type="server_error",
            code="generation_error",
        )

    choices: List[ChatCompletionChoice] = []
    total_completion_tokens = 0
    for idx, item in enumerate(result.completions):
        total_completion_tokens += item.tokens
        choices.append(
            ChatCompletionChoice(
                index=idx,
                message=ChatMessage(role="assistant", content=item.text.strip()),
                finish_reason=item.finish_reason,
            )
        )

    usage = UsageInfo(
        prompt_tokens=result.prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=result.prompt_tokens + total_completion_tokens,
    )
    return ChatCompletionResponse(model=payload.model, choices=choices, usage=usage)


def _streaming_chat_completion(
    payload: ChatCompletionRequest,
    prompt: str,
    stop_sequences: List[str],
) -> StreamingResponse:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"

    def event_stream() -> Generator[bytes, None, None]:
        stream = engine.create_stream(
            payload.model,
            prompt,
            temperature=payload.temperature,
            top_p=payload.top_p,
            max_tokens=payload.max_tokens,
            stop=stop_sequences,
        )
        
        # Send initial role delta
        initial_chunk = ChatCompletionChunk(
            id=completion_id,
            created=int(time.time()),
            model=payload.model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkChoiceDelta(role="assistant"),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {initial_chunk.model_dump_json()}\n\n".encode()
        
        for token in stream.iter_tokens():
            chunk = ChatCompletionChunk(
                id=completion_id,
                created=int(time.time()),
                model=payload.model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkChoiceDelta(content=token),
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n".encode()
        
        # Send final chunk with finish_reason
        final_chunk = ChatCompletionChunk(
            id=completion_id,
            created=int(time.time()),
            model=payload.model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkChoiceDelta(),
                    finish_reason=stream.finish_reason,
                )
            ],
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )
