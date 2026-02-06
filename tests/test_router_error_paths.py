"""Router-level error path tests for OpenAI-compatible payloads."""
from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from app.core.model_registry import ModelSpec
from app.routers import chat, completions, embeddings, responses
from app.schemas.chat import ChatCompletionRequest
from app.schemas.completions import CompletionRequest
from app.schemas.responses import ResponseRequest


def _raise_key_error(_: str) -> None:
    raise KeyError("unknown")


def test_completions_unknown_model_returns_404_openai_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.routers.completions.get_model_spec", _raise_key_error)
    payload = CompletionRequest.model_validate({"model": "missing", "prompt": "Hi"})

    with pytest.raises(HTTPException) as exc:
        asyncio.run(completions.create_completion(payload))

    assert exc.value.status_code == 404
    assert exc.value.detail["type"] == "model_not_found"
    assert exc.value.detail["param"] == "model"
    assert exc.value.detail["code"] == "model_not_found"


def test_chat_unknown_model_returns_404_openai_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.routers.chat.get_model_spec", _raise_key_error)
    payload = ChatCompletionRequest.model_validate(
        {
            "model": "missing",
            "messages": [{"role": "user", "content": "Hi"}],
        }
    )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(chat.create_chat_completion(payload))

    assert exc.value.status_code == 404
    assert exc.value.detail["type"] == "model_not_found"
    assert exc.value.detail["param"] == "model"
    assert exc.value.detail["code"] == "model_not_found"


def test_responses_unknown_model_returns_404_openai_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.routers.responses.get_model_spec", _raise_key_error)
    payload = ResponseRequest.model_validate({"model": "missing", "input": "Hi"})

    with pytest.raises(HTTPException) as exc:
        asyncio.run(responses.create_response(payload))

    assert exc.value.status_code == 404
    assert exc.value.detail["type"] == "model_not_found"
    assert exc.value.detail["param"] == "model"
    assert exc.value.detail["code"] == "model_not_found"


def test_completions_generation_exception_returns_generation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_: object, **__: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("app.routers.completions.get_model_spec", lambda _: None)
    monkeypatch.setattr("app.routers.completions.engine.generate", boom)
    payload = CompletionRequest.model_validate({"model": "GPT3-dev", "prompt": "Hi"})

    with pytest.raises(HTTPException) as exc:
        asyncio.run(completions.create_completion(payload))

    assert exc.value.status_code == 500
    assert exc.value.detail["type"] == "server_error"
    assert exc.value.detail["code"] == "generation_error"
    assert "Generation error:" in exc.value.detail["message"]


def test_chat_generation_exception_returns_generation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_: object, **__: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "app.routers.chat.get_model_spec",
        lambda model: ModelSpec(name=model, hf_repo="dummy/instruct", is_instruct=True),
    )
    monkeypatch.setattr("app.routers.chat.engine.apply_chat_template", lambda *_: "prompt")
    monkeypatch.setattr("app.routers.chat.engine.generate", boom)

    payload = ChatCompletionRequest.model_validate(
        {
            "model": "GPT4-dev-177M-1511-Instruct",
            "messages": [{"role": "user", "content": "Hi"}],
        }
    )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(chat.create_chat_completion(payload))

    assert exc.value.status_code == 500
    assert exc.value.detail["type"] == "server_error"
    assert exc.value.detail["code"] == "generation_error"
    assert "Generation error:" in exc.value.detail["message"]


def test_responses_generation_exception_returns_generation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_: object, **__: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "app.routers.responses.get_model_spec",
        lambda model: ModelSpec(name=model, hf_repo="dummy/base", is_instruct=False),
    )
    monkeypatch.setattr("app.routers.responses.engine.generate", boom)
    payload = ResponseRequest.model_validate({"model": "GPT3-dev", "input": "Hi"})

    with pytest.raises(HTTPException) as exc:
        asyncio.run(responses.create_response(payload))

    assert exc.value.status_code == 500
    assert exc.value.detail["type"] == "server_error"
    assert exc.value.detail["code"] == "generation_error"
    assert "Generation error:" in exc.value.detail["message"]


def test_responses_structured_input_with_non_instruct_model_returns_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.routers.responses.get_model_spec",
        lambda model: ModelSpec(name=model, hf_repo="dummy/base", is_instruct=False),
    )

    payload = ResponseRequest.model_validate(
        {
            "model": "GPT3-dev",
            "input": [{"role": "user", "content": "Hi"}],
        }
    )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(responses.create_response(payload))

    assert exc.value.status_code == 400
    assert exc.value.detail["type"] == "invalid_request_error"
    assert exc.value.detail["param"] == "model"
    assert "not an instruct model" in exc.value.detail["message"]


def test_embeddings_enabled_backend_returns_pending_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummySettings:
        enable_embeddings_backend = True

    monkeypatch.setattr("app.routers.embeddings.get_settings", lambda: DummySettings())

    with pytest.raises(HTTPException) as exc:
        asyncio.run(embeddings.create_embeddings())

    assert exc.value.status_code == 501
    assert exc.value.detail["type"] == "not_implemented_error"
    assert exc.value.detail["code"] == "embeddings_backend_pending"
