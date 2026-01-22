"""Schemas for the Responses API endpoint."""
from __future__ import annotations

import time
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, AliasChoices


class ResponseInputContentPart(BaseModel):
    type: Literal["input_text"] = "input_text"
    text: str


class ResponseInputMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, List[ResponseInputContentPart]]


class ResponseRequest(BaseModel):
    model: str
    input: Union[str, List[ResponseInputMessage]]
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stop: Optional[List[str] | str] = None
    max_output_tokens: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("max_output_tokens", "max_tokens"),
    )
    stream: bool = False


class ResponseOutputText(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str


class ResponseOutputMessage(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List[ResponseOutputText]


class ResponseUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class ResponsePayload(BaseModel):
    id: str
    object: Literal["response"] = "response"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    output: List[ResponseOutputMessage]
    usage: ResponseUsage
