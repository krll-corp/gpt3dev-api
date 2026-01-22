"""Responses API endpoint."""
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
from ..schemas.responses import (
    ResponseInputMessage,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponsePayload,
    ResponseRequest,
    ResponseUsage,
)

router = APIRouter(prefix="/v1", tags=["responses"])


def _render_input_text(message: ResponseInputMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(part.text for part in message.content if part.type == "input_text")


def _normalize_messages(input_payload: List[ResponseInputMessage]) -> List[dict]:
    return [{"role": message.role, "content": _render_input_text(message)} for message in input_payload]


def _stop_sequences(stop: List[str] | str | None) -> List[str]:
    if isinstance(stop, list):
        return stop
    return [stop] if stop else []


def _build_output(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id=f"msg_{uuid.uuid4().hex}",
        content=[ResponseOutputText(text=text)],
    )


@router.post("/responses", response_model=ResponsePayload)
async def create_response(payload: ResponseRequest) -> ResponsePayload | StreamingResponse:
    """Generate a response using OpenAI's Responses API format."""
    try:
        spec = get_model_spec(payload.model)
    except KeyError:
        raise model_not_found(payload.model)

    stop_sequences = _stop_sequences(payload.stop)

    if isinstance(payload.input, str):
        if spec.is_instruct:
            messages = [{"role": "user", "content": payload.input}]
            prompt = engine.apply_chat_template(payload.model, messages)
        else:
            prompt = payload.input
    else:
        if not spec.is_instruct:
            raise openai_http_error(
                400,
                f"Model '{payload.model}' is not an instruct model and cannot accept structured input. "
                "Provide a plain string input or use /v1/chat/completions for chat-formatted prompts.",
                error_type="invalid_request_error",
                param="model",
            )
        messages = _normalize_messages(payload.input)
        prompt = engine.apply_chat_template(payload.model, messages)

    if payload.stream:
        return _streaming_response(payload, prompt, stop_sequences)

    try:
        result = await asyncio.to_thread(
            engine.generate,
            payload.model,
            prompt,
            temperature=payload.temperature,
            top_p=payload.top_p,
            max_tokens=payload.max_output_tokens,
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

    output: List[ResponseOutputMessage] = []
    total_completion_tokens = 0
    for item in result.completions:
        total_completion_tokens += item.tokens
        output.append(_build_output(item.text.strip()))

    usage = ResponseUsage(
        input_tokens=result.prompt_tokens,
        output_tokens=total_completion_tokens,
        total_tokens=result.prompt_tokens + total_completion_tokens,
    )
    return ResponsePayload(
        id=f"resp_{uuid.uuid4().hex}",
        model=payload.model,
        output=output,
        usage=usage,
    )


def _streaming_response(
    payload: ResponseRequest,
    prompt: str,
    stop_sequences: List[str],
) -> StreamingResponse:
    response_id = f"resp_{uuid.uuid4().hex}"
    message_id = f"msg_{uuid.uuid4().hex}"
    created = int(time.time())

    def event_stream() -> Generator[bytes, None, None]:
        stream = engine.create_stream(
            payload.model,
            prompt,
            temperature=payload.temperature,
            top_p=payload.top_p,
            max_tokens=payload.max_output_tokens,
            stop=stop_sequences,
        )
        base_payload = ResponsePayload(
            id=response_id,
            created=created,
            model=payload.model,
            output=[],
            usage=ResponseUsage(input_tokens=0, output_tokens=0, total_tokens=0),
        )
        created_payload = {
            "type": "response.created",
            "response": base_payload.model_dump(),
        }
        yield f"data: {json.dumps(created_payload)}\n\n".encode()

        collected = ""
        for token in stream.iter_tokens():
            collected += token
            delta_payload = {
                "type": "response.output_text.delta",
                "response_id": response_id,
                "item_id": message_id,
                "output_index": 0,
                "content_index": 0,
                "delta": token,
            }
            yield f"data: {json.dumps(delta_payload)}\n\n".encode()

        usage = ResponseUsage(
            input_tokens=stream.prompt_tokens,
            output_tokens=stream.completion_tokens,
            total_tokens=stream.prompt_tokens + stream.completion_tokens,
        )
        final_payload = ResponsePayload(
            id=response_id,
            created=created,
            model=payload.model,
            output=[ResponseOutputMessage(id=message_id, content=[ResponseOutputText(text=collected)])],
            usage=usage,
        )
        completed_payload = {
            "type": "response.completed",
            "response": final_payload.model_dump(),
        }
        yield f"data: {json.dumps(completed_payload)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )
