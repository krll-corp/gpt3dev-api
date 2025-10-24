"""Schemas for the legacy completions endpoint."""
from __future__ import annotations

import time
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from .common import UsageInfo


class CompletionRequest(BaseModel):
    model: str
    prompt: str | List[str]
    suffix: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    logprobs: Optional[int] = None
    echo: bool = False
    stop: Optional[List[str] | str] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    best_of: Optional[int] = None
    user: Optional[str] = None


class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[dict] = None
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{int(time.time()*1000)}")
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo


class CompletionChunkChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[dict] = None
    finish_reason: Optional[str] = None


class CompletionChunk(BaseModel):
    id: str
    object: Literal["text_completion.chunk"] = "text_completion.chunk"
    created: int
    model: str
    choices: List[CompletionChunkChoice]
