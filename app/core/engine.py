"""Lazy model loading and generation utilities."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock, Thread
from typing import Generator, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextIteratorStreamer,
)

from .errors import model_not_found, openai_http_error
from .model_registry import get_model_spec
from .settings import get_settings
from .tokens import count_tokens

logger = logging.getLogger(__name__)


@dataclass
class GeneratedText:
    """Container for a single completion."""

    text: str
    tokens: int
    finish_reason: str


@dataclass
class GenerationResult:
    """Result of a non-streaming generation call."""

    prompt_tokens: int
    completions: List[GeneratedText]


class StreamingGeneration:
    """Represents an in-flight streaming generation."""

    def __init__(
        self,
        *,
        model_name: str,
        handle: "_ModelHandle",
        prompt_tokens: int,
        streamer: TextIteratorStreamer,
        thread: Thread,
        stop_sequences: Sequence[str],
    ) -> None:
        self.model_name = model_name
        self._handle = handle
        self.prompt_tokens = prompt_tokens
        self._streamer = streamer
        self._thread = thread
        self._stop_sequences = stop_sequences
        self._buffer: str = ""
        self.finish_reason: str = "length"
        self.completion_tokens: int = 0

    def iter_tokens(self) -> Generator[str, None, None]:
        """Yield decoded tokens as they become available."""

        try:
            for token in self._streamer:
                if token is None:
                    continue
                previous_len = len(self._buffer)
                self._buffer += token
                emit = token
                if self._stop_sequences:
                    trimmed, finished = _apply_stop_sequences(self._buffer, self._stop_sequences)
                    if trimmed != self._buffer:
                        emit = trimmed[previous_len:]
                        self._buffer = trimmed
                        self.finish_reason = finished
                        if emit:
                            yield emit
                        break
                if emit:
                    yield emit
        finally:
            self._thread.join()
            if self.finish_reason != "stop":
                self.finish_reason = "length"
            self.completion_tokens = count_tokens(
                self._buffer, self.model_name, self._handle.tokenizer
            )

    @property
    def text(self) -> str:
        return self._buffer


class _ModelHandle:
    """Wrap a loaded model/tokenizer pair."""

    def __init__(self, model_name: str) -> None:
        spec = get_model_spec(model_name)
        settings = get_settings()
        token = settings.hf_token
        tokenizer = AutoTokenizer.from_pretrained(
            spec.hf_repo,
            use_auth_token=token,
            trust_remote_code=True,
        )
        model_kwargs = {
            "use_auth_token": token,
            "trust_remote_code": True,
        }
        if spec.dtype:
            torch_dtype = getattr(torch, spec.dtype, None)
            if torch_dtype is not None:
                model_kwargs["torch_dtype"] = torch_dtype
        device_pref = spec.device or settings.default_device
        if device_pref == "auto":
            if torch.cuda.is_available():
                device_pref = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # pragma: no cover - macOS
                device_pref = "mps"
            else:
                device_pref = "cpu"
        device_map = "auto" if device_pref == "cuda" else None
        model = AutoModelForCausalLM.from_pretrained(
            spec.hf_repo,
            device_map=device_map,
            **model_kwargs,
        )
        if device_map is None:
            model.to(device_pref)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        self.model = model
        self.spec = spec
        self.device = device_pref
        self.device_map = device_map


_model_cache: dict[str, _ModelHandle] = {}
_model_lock = Lock()


def _get_handle(model_name: str) -> _ModelHandle:
    if model_name in _model_cache:
        return _model_cache[model_name]
    with _model_lock:
        if model_name in _model_cache:
            return _model_cache[model_name]
        try:
            handle = _ModelHandle(model_name)
        except KeyError as exc:
            raise model_not_found(model_name) from exc
        _model_cache[model_name] = handle
        logger.info("Loaded model %s", model_name)
        return handle


def _normalize_stop(stop: Optional[Iterable[str] | str]) -> Tuple[str, ...]:
    if stop is None:
        return ()
    if isinstance(stop, str):
        return (stop,)
    return tuple(stop)


def _apply_stop_sequences(text: str, stop_sequences: Sequence[str]) -> tuple[str, str]:
    if not stop_sequences:
        return text, "length"
    earliest: Optional[int] = None
    for sequence in stop_sequences:
        idx = text.find(sequence)
        if idx != -1 and (earliest is None or idx < earliest):
            earliest = idx
    if earliest is not None:
        return text[:earliest], "stop"
    return text, "length"


def _prepare_inputs(
    handle: _ModelHandle,
    prompt: str,
    max_new_tokens: int,
) -> tuple[dict[str, torch.Tensor], int]:
    tokenizer = handle.tokenizer
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    max_context = handle.spec.max_context_tokens or get_settings().max_context_tokens
    if input_ids.shape[1] > max_context:
        input_ids = input_ids[:, -max_context:]
        attention_mask = attention_mask[:, -max_context:]
    prompt_tokens = input_ids.shape[1]
    if prompt_tokens + max_new_tokens > max_context:
        overflow = prompt_tokens + max_new_tokens - max_context
        if overflow >= prompt_tokens:
            raise openai_http_error(
                400,
                "Requested max_tokens exceeds context window for this model.",
                param="max_tokens",
            )
        input_ids = input_ids[:, overflow:]
        attention_mask = attention_mask[:, overflow:]
        prompt_tokens = input_ids.shape[1]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }, prompt_tokens


def generate(
    model_name: str,
    prompt: str,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: Optional[int] = None,
    stop: Optional[Iterable[str] | str] = None,
    n: int = 1,
) -> GenerationResult:
    handle = _get_handle(model_name)
    stop_sequences = _normalize_stop(stop)
    max_new_tokens = max_tokens or 256
    inputs, prompt_tokens = _prepare_inputs(handle, prompt, max_new_tokens)
    target_device = handle.device
    if handle.device_map is not None:
        try:
            target_device = next(handle.model.parameters()).device
        except StopIteration:  # pragma: no cover - defensive
            target_device = handle.device
    inputs = {k: v.to(target_device) for k, v in inputs.items()}
    generation_config = GenerationConfig(
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        num_return_sequences=n,
        pad_token_id=handle.tokenizer.pad_token_id,
    )
    with torch.no_grad():
        output_ids = handle.model.generate(**inputs, generation_config=generation_config)
    if output_ids.dim() == 1:
        sequences = output_ids.unsqueeze(0)
    else:
        sequences = output_ids
    completions: List[GeneratedText] = []
    prompt_length = inputs["input_ids"].shape[1]
    for seq in sequences:
        generated_ids = seq[prompt_length:]
        text = handle.tokenizer.decode(generated_ids, skip_special_tokens=True)
        text, finish_reason = _apply_stop_sequences(text, stop_sequences)
        completion_tokens = count_tokens(text, model_name, handle.tokenizer)
        completions.append(
            GeneratedText(text=text, tokens=completion_tokens, finish_reason=finish_reason)
        )
    return GenerationResult(prompt_tokens=prompt_tokens, completions=completions)


def create_stream(
    model_name: str,
    prompt: str,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: Optional[int] = None,
    stop: Optional[Iterable[str] | str] = None,
) -> StreamingGeneration:
    handle = _get_handle(model_name)
    stop_sequences = _normalize_stop(stop)
    max_new_tokens = max_tokens or 256
    inputs, prompt_tokens = _prepare_inputs(handle, prompt, max_new_tokens)
    target_device = handle.device
    if handle.device_map is not None:
        try:
            target_device = next(handle.model.parameters()).device
        except StopIteration:  # pragma: no cover - defensive
            target_device = handle.device
    inputs = {k: v.to(target_device) for k, v in inputs.items()}
    streamer = TextIteratorStreamer(
        handle.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    generation_config = GenerationConfig(
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        pad_token_id=handle.tokenizer.pad_token_id,
    )

    def _worker() -> None:
        with torch.no_grad():
            handle.model.generate(
                **inputs,
                generation_config=generation_config,
                streamer=streamer,
            )

    thread = Thread(target=_worker, daemon=True)
    thread.start()
    return StreamingGeneration(
        model_name=model_name,
        handle=handle,
        prompt_tokens=prompt_tokens,
        streamer=streamer,
        thread=thread,
        stop_sequences=stop_sequences,
    )
