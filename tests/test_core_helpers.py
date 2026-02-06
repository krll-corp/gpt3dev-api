"""Unit tests for prompt/token/engine helper utilities."""
from __future__ import annotations

import types

from app.core import engine, tokens
from app.core.prompting import DEFAULT_SYSTEM_PROMPT, render_chat_prompt
from app.schemas.chat import ChatMessage


class DummyTokenizer:
    def __init__(self) -> None:
        self.called_with: tuple[str, bool] | None = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.called_with = (text, add_special_tokens)
        return [1, 2, 3]


class DummyEncoding:
    def __init__(self, size: int) -> None:
        self._size = size

    def encode(self, _: str) -> list[int]:
        return list(range(self._size))


class DummyTikToken:
    def __init__(self, size: int) -> None:
        self._size = size

    def encoding_for_model(self, _: str) -> DummyEncoding:
        return DummyEncoding(self._size)


def test_render_chat_prompt_uses_default_system_prompt() -> None:
    prompt = render_chat_prompt([ChatMessage(role="user", content="Hello")])
    assert prompt.startswith(f"System: {DEFAULT_SYSTEM_PROMPT}\n\n")
    assert prompt.endswith("Assistant:")
    assert "User: Hello" in prompt


def test_render_chat_prompt_overrides_system_prompt_when_present() -> None:
    prompt = render_chat_prompt(
        [
            ChatMessage(role="system", content="Custom system"),
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi"),
        ]
    )
    assert prompt.startswith("System: Custom system\n\n")
    assert "User: Hello" in prompt
    assert "Assistant: Hi" in prompt


def test_count_tokens_returns_zero_for_empty_text() -> None:
    assert tokens.count_tokens("", "GPT3-dev") == 0


def test_count_tokens_uses_tiktoken_when_available(monkeypatch) -> None:
    monkeypatch.setattr(tokens, "tiktoken", DummyTikToken(size=4))
    assert tokens.count_tokens("hello", "GPT3-dev") == 4


def test_count_tokens_falls_back_to_tokenizer_encode(monkeypatch) -> None:
    monkeypatch.setattr(tokens, "tiktoken", None)
    tokenizer = DummyTokenizer()
    assert tokens.count_tokens("hello", "GPT3-dev", tokenizer=tokenizer) == 3
    assert tokenizer.called_with == ("hello", False)


def test_apply_stop_sequences_returns_earliest_stop_index() -> None:
    text, reason = engine._apply_stop_sequences(
        "abc<END>xyz<STOP>",
        ["<STOP>", "<END>"],
    )
    assert text == "abc"
    assert reason == "stop"


def test_normalize_stop_handles_none_string_and_iterable() -> None:
    assert engine._normalize_stop(None) == ()
    assert engine._normalize_stop("stop") == ("stop",)
    assert engine._normalize_stop(["a", "b"]) == ("a", "b")


def test_pad_token_id_prefers_pad_then_eos_then_zero() -> None:
    with_pad = types.SimpleNamespace(pad_token_id=9, eos_token_id=7)
    with_eos_only = types.SimpleNamespace(pad_token_id=None, eos_token_id=7)
    with_none = types.SimpleNamespace(pad_token_id=None, eos_token_id=None)

    assert engine._pad_token_id_or_default(with_pad) == 9
    assert engine._pad_token_id_or_default(with_eos_only) == 7
    assert engine._pad_token_id_or_default(with_none) == 0
