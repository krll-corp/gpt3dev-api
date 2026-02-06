"""Streaming contract tests for OpenAI-compatible SSE endpoints."""
from __future__ import annotations

import asyncio
import json
from collections import deque

import pytest
from fastapi.responses import StreamingResponse

from app.core.model_registry import ModelSpec
from app.routers import chat, completions, responses
from app.schemas.chat import ChatCompletionRequest
from app.schemas.completions import CompletionRequest
from app.schemas.responses import ResponseRequest


class DummyStream:
    def __init__(
        self,
        *,
        tokens: list[str],
        prompt_tokens: int,
        completion_tokens: int,
        finish_reason: str = "stop",
    ) -> None:
        self._tokens = tokens
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.finish_reason = finish_reason

    def iter_tokens(self):
        for token in self._tokens:
            yield token


async def _read_stream_body(response: StreamingResponse) -> str:
    chunks: list[str] = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunks.append(chunk.decode("utf-8"))
        else:
            chunks.append(chunk)
    return "".join(chunks)


def _parse_sse_data_frames(raw_body: str) -> list[str]:
    frames = [frame.strip() for frame in raw_body.split("\n\n") if frame.strip()]
    data_frames: list[str] = []
    for frame in frames:
        assert frame.startswith("data: ")
        data_frames.append(frame[len("data: ") :])
    return data_frames


def test_completions_stream_emits_sse_chunks_usage_and_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.routers.completions.get_model_spec", lambda _: None)
    monkeypatch.setattr(
        "app.routers.completions.engine.create_stream",
        lambda *_, **__: DummyStream(
            tokens=["Hel", "lo"],
            prompt_tokens=3,
            completion_tokens=2,
            finish_reason="stop",
        ),
    )

    payload = CompletionRequest.model_validate(
        {
            "model": "GPT3-dev",
            "prompt": "Hello",
            "stream": True,
        }
    )
    response = asyncio.run(completions.create_completion(payload))
    assert isinstance(response, StreamingResponse)

    body = asyncio.run(_read_stream_body(response))
    data_frames = _parse_sse_data_frames(body)
    assert data_frames[-1] == "[DONE]"

    chunks = [json.loads(frame) for frame in data_frames[:-1]]
    assert chunks[0]["object"] == "text_completion.chunk"
    assert chunks[0]["choices"][0]["text"] == "Hel"
    assert chunks[1]["choices"][0]["text"] == "lo"
    assert chunks[2]["choices"][0]["finish_reason"] == "stop"

    tail = chunks[-1]
    assert tail["choices"] == []
    assert tail["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }


def test_chat_stream_emits_initial_role_delta_and_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.routers.chat.get_model_spec",
        lambda model: ModelSpec(name=model, hf_repo="dummy/instruct", is_instruct=True),
    )
    monkeypatch.setattr("app.routers.chat.engine.apply_chat_template", lambda *_: "formatted")
    monkeypatch.setattr(
        "app.routers.chat.engine.create_stream",
        lambda *_, **__: DummyStream(
            tokens=["Hi", " there"],
            prompt_tokens=4,
            completion_tokens=2,
            finish_reason="stop",
        ),
    )

    payload = ChatCompletionRequest.model_validate(
        {
            "model": "GPT4-dev-177M-1511-Instruct",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        }
    )
    response = asyncio.run(chat.create_chat_completion(payload))
    assert isinstance(response, StreamingResponse)

    body = asyncio.run(_read_stream_body(response))
    data_frames = _parse_sse_data_frames(body)
    assert data_frames[-1] == "[DONE]"

    chunks = [json.loads(frame) for frame in data_frames[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[1]["choices"][0]["delta"]["content"] == "Hi"
    assert chunks[2]["choices"][0]["delta"]["content"] == " there"
    assert chunks[3]["choices"][0]["finish_reason"] == "stop"


def test_responses_stream_emits_created_delta_completed_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.routers.responses.get_model_spec",
        lambda model: ModelSpec(name=model, hf_repo="dummy/base", is_instruct=False),
    )
    monkeypatch.setattr(
        "app.routers.responses.engine.create_stream",
        lambda *_, **__: DummyStream(
            tokens=["Hi", " there"],
            prompt_tokens=5,
            completion_tokens=2,
            finish_reason="stop",
        ),
    )

    payload = ResponseRequest.model_validate(
        {
            "model": "GPT3-dev",
            "input": "Say hi",
            "stream": True,
        }
    )
    response = asyncio.run(responses.create_response(payload))
    assert isinstance(response, StreamingResponse)

    body = asyncio.run(_read_stream_body(response))
    data_frames = _parse_sse_data_frames(body)
    assert data_frames[-1] == "[DONE]"

    events = [json.loads(frame) for frame in data_frames[:-1]]
    assert events[0]["type"] == "response.created"
    assert events[1]["type"] == "response.output_text.delta"
    assert events[1]["delta"] == "Hi"
    assert events[2]["type"] == "response.output_text.delta"
    assert events[2]["delta"] == " there"
    assert events[3]["type"] == "response.completed"
    assert events[3]["response"]["output"][0]["content"][0]["text"] == "Hi there"
    assert events[3]["response"]["usage"] == {
        "input_tokens": 5,
        "output_tokens": 2,
        "total_tokens": 7,
    }


def test_completions_stream_usage_aggregates_prompt_and_completion_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    streams = deque(
        [
            DummyStream(tokens=["a1"], prompt_tokens=10, completion_tokens=1),
            DummyStream(tokens=["a2"], prompt_tokens=999, completion_tokens=2),
            DummyStream(tokens=["b1"], prompt_tokens=20, completion_tokens=3),
            DummyStream(tokens=["b2"], prompt_tokens=888, completion_tokens=4),
        ]
    )

    def fake_create_stream(model: str, prompt: str, **_: object) -> DummyStream:
        calls.append(prompt)
        return streams.popleft()

    monkeypatch.setattr("app.routers.completions.get_model_spec", lambda _: None)
    monkeypatch.setattr("app.routers.completions.engine.create_stream", fake_create_stream)

    payload = CompletionRequest.model_validate(
        {
            "model": "GPT3-dev",
            "prompt": ["alpha", "beta"],
            "n": 2,
            "stream": True,
        }
    )
    response = asyncio.run(completions.create_completion(payload))
    body = asyncio.run(_read_stream_body(response))
    data_frames = _parse_sse_data_frames(body)
    assert data_frames[-1] == "[DONE]"

    chunks = [json.loads(frame) for frame in data_frames[:-1]]
    tail = chunks[-1]

    assert calls == ["alpha", "alpha", "beta", "beta"]
    assert tail["usage"] == {
        "prompt_tokens": 30,
        "completion_tokens": 10,
        "total_tokens": 40,
    }
