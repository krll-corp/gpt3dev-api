"""Legacy completions endpoint."""
from __future__ import annotations

import json
import time
import uuid
import asyncio
from typing import Generator, List

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..core import engine
from ..core.errors import model_not_found
from ..core.errors import openai_http_error
from ..core.model_registry import get_model_spec
from ..schemas.common import UsageInfo
from ..schemas.completions import (
    CompletionChoice,
    CompletionChunk,
    CompletionChunkChoice,
    CompletionRequest,
    CompletionResponse,
)

router = APIRouter(prefix="/v1", tags=["completions"])


@router.post("/completions", response_model=CompletionResponse)
async def create_completion(payload: CompletionRequest):
    try:
        get_model_spec(payload.model)
    except KeyError:
        raise model_not_found(payload.model)
    prompt = payload.prompt if isinstance(payload.prompt, str) else "\n".join(payload.prompt)
    stop_sequences = payload.stop if isinstance(payload.stop, list) else (
        [payload.stop] if payload.stop else []
    )
    if payload.stream:
        return _streaming_completion(payload, prompt, stop_sequences)
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
    except Exception as exc:  # pragma: no cover - bubble as OpenAI-style error
        raise openai_http_error(
            500,
            f"Generation error: {exc}",
            error_type="server_error",
            code="generation_error",
        )
    choices: List[CompletionChoice] = []
    total_completion_tokens = 0
    for index, item in enumerate(result.completions):
        total_completion_tokens += item.tokens
        choices.append(
            CompletionChoice(
                text=item.text,
                index=index,
                logprobs=None,
                finish_reason=item.finish_reason,
            )
        )
    usage = UsageInfo(
        prompt_tokens=result.prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=result.prompt_tokens + total_completion_tokens,
    )
    return CompletionResponse(model=payload.model, choices=choices, usage=usage)


def _streaming_completion(
    payload: CompletionRequest,
    prompt: str,
    stop_sequences: List[str],
) -> StreamingResponse:
    completion_id = f"cmpl-{uuid.uuid4().hex}"

    def event_stream() -> Generator[bytes, None, None]:
        total_completion_tokens = 0
        prompt_tokens = None
        for index in range(payload.n):
            stream = engine.create_stream(
                payload.model,
                prompt,
                temperature=payload.temperature,
                top_p=payload.top_p,
                max_tokens=payload.max_tokens,
                stop=stop_sequences,
            )
            if prompt_tokens is None:
                prompt_tokens = stream.prompt_tokens
            collected_text = ""
            for token in stream.iter_tokens():
                collected_text += token
                chunk = CompletionChunk(
                    id=completion_id,
                    created=int(time.time()),
                    model=payload.model,
                    choices=[
                        CompletionChunkChoice(
                            text=token,
                            index=index,
                        )
                    ],
                )
                yield _sse_payload(chunk)
            total_completion_tokens += stream.completion_tokens
            final_chunk = CompletionChunk(
                id=completion_id,
                created=int(time.time()),
                model=payload.model,
                choices=[
                    CompletionChunkChoice(
                        text="",
                        index=index,
                        finish_reason=stream.finish_reason,
                    )
                ],
            )
            yield _sse_payload(final_chunk)
        usage = UsageInfo(
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=total_completion_tokens,
            total_tokens=(prompt_tokens or 0) + total_completion_tokens,
        )
        tail = CompletionChunk(
            id=completion_id,
            created=int(time.time()),
            model=payload.model,
            choices=[],
        ).model_dump()
        tail["usage"] = usage.model_dump()
        yield _sse_payload(tail)
        yield b"data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


def _sse_payload(data: CompletionChunk | dict) -> bytes:
    if isinstance(data, CompletionChunk):
        payload = data.model_dump(exclude_none=True)
    else:
        payload = data
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")
