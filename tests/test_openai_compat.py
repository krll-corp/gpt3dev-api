"""Integration tests for OpenAI-compatible endpoint logic."""
from __future__ import annotations

import sys
import types
from pathlib import Path
import asyncio

import pytest
from fastapi import HTTPException
import pydantic

fake_pydantic_settings = types.ModuleType("pydantic_settings")


class _FakeBaseSettings(pydantic.BaseModel):
    model_config: dict = {}

    def model_dump(self, *args, **kwargs):  # pragma: no cover - passthrough
        return super().model_dump(*args, **kwargs)


fake_pydantic_settings.BaseSettings = _FakeBaseSettings
fake_pydantic_settings.SettingsConfigDict = dict
sys.modules.setdefault("pydantic_settings", fake_pydantic_settings)

fake_torch = types.ModuleType("torch")
fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
fake_torch.backends = types.SimpleNamespace(
    mps=types.SimpleNamespace(is_available=lambda: False)
)


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _no_grad() -> _NoGrad:
    return _NoGrad()


def _dummy_to(self, *args, **kwargs):  # noqa: D401
    return self


fake_torch.no_grad = _no_grad
fake_torch.Tensor = type("Tensor", (), {"to": _dummy_to, "dim": lambda self: 2, "unsqueeze": lambda self, _: self})
sys.modules.setdefault("torch", fake_torch)

fake_transformers = types.ModuleType("transformers")


class _DummyTokenizer:
    pad_token_id = 0
    eos_token = ""  # noqa: D401

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # noqa: D401
        return cls()

    def __call__(self, prompt: str, return_tensors: str = "pt") -> dict:
        tensor = types.SimpleNamespace(
            shape=(1, max(len(prompt), 1)),
            __getitem__=lambda self, key: self,
            to=_dummy_to,
        )
        return {"input_ids": tensor, "attention_mask": tensor}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [0] * max(len(text), 1)

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return "".join(token_ids) if isinstance(token_ids, list) else ""


class _DummyModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # noqa: D401
        return cls()

    def to(self, device):  # noqa: D401
        return self

    def generate(self, *args, **kwargs):  # noqa: D401
        return types.SimpleNamespace(dim=lambda: 2, unsqueeze=lambda _: None)


class _DummyGenerationConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DummyStreamer:
    def __init__(self, *args, **kwargs):
        pass

    def __iter__(self):
        return iter([])


fake_transformers.AutoTokenizer = _DummyTokenizer
fake_transformers.AutoModelForCausalLM = _DummyModel
fake_transformers.GenerationConfig = _DummyGenerationConfig
fake_transformers.TextIteratorStreamer = _DummyStreamer
fake_transformers.PreTrainedTokenizerBase = object
sys.modules.setdefault("transformers", fake_transformers)

fake_yaml = types.ModuleType("yaml")
fake_yaml.safe_load = lambda data: []
sys.modules.setdefault("yaml", fake_yaml)

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.routers import chat, completions, embeddings, models
from app.schemas.chat import ChatCompletionRequest
from app.schemas.completions import CompletionRequest


def test_list_models() -> None:
    payload = models.list_available_models()
    assert payload["object"] == "list"
    assert payload["data"] == []


def test_completions_non_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyResult:
        prompt_tokens = 5
        completions = [type("C", (), {"text": "Hello", "tokens": 2, "finish_reason": "stop"})()]

    def fake_generate(*args, **kwargs):
        return DummyResult()

    monkeypatch.setattr("app.routers.completions.engine.generate", fake_generate)
    monkeypatch.setattr("app.routers.completions.get_model_spec", lambda model: None)
    payload = CompletionRequest.model_validate({
        "model": "GPT3-dev",
        "prompt": "Hello",
    })
    response = asyncio.run(completions.create_completion(payload))
    body = response.model_dump()
    assert body["model"] == "GPT3-dev"
    assert body["choices"][0]["text"] == "Hello"
    assert body["usage"]["total_tokens"] == 7


def test_chat_disabled() -> None:
    payload = ChatCompletionRequest.model_validate({
        "model": "GPT3-dev",
        "messages": [
            {"role": "user", "content": "Hi"},
        ],
    })

    with pytest.raises(HTTPException) as exc:
        asyncio.run(chat.create_chat_completion(payload))
    assert exc.value.status_code == 501
    assert exc.value.detail["code"] == "chat_completions_not_available"


def test_embeddings_not_implemented() -> None:
    with pytest.raises(HTTPException) as exc:
        asyncio.run(embeddings.create_embeddings())
    assert exc.value.status_code == 501
    assert exc.value.detail["code"] == "embeddings_backend_unavailable"
