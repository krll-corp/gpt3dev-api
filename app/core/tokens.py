"""Utilities for counting tokens in prompts and completions."""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

try:  # pragma: no cover - optional dependency
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None  # type: ignore

if TYPE_CHECKING:  # Only for static type checking; avoid runtime import cost
    from transformers import PreTrainedTokenizerBase

from .model_registry import get_model_spec
from .settings import get_settings

_tokenizer_cache: dict[str, "PreTrainedTokenizerBase"] = {}


def _get_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]
    # Lazy import to keep app startup fast
    from transformers import AutoTokenizer  # noqa: WPS433 - local import by design

    spec = get_model_spec(model_name)
    settings = get_settings()
    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_repo,
        use_auth_token=settings.hf_token,
        trust_remote_code=True,
    )
    _tokenizer_cache[model_name] = tokenizer
    return tokenizer


def count_tokens(
    text: str,
    model_name: str,
    tokenizer: Optional["PreTrainedTokenizerBase"] = None,
) -> int:
    """Count tokens using tiktoken when available, otherwise HF tokenizers."""

    if not text:
        return 0
    if tiktoken is not None:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    tokenizer = tokenizer or _get_tokenizer(model_name)
    return len(tokenizer.encode(text, add_special_tokens=False))
