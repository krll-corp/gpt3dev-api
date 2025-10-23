"""Prompt construction utilities for chat-style inputs."""
from __future__ import annotations

from typing import Iterable, List

from ..schemas.chat import ChatMessage

DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."


def render_chat_prompt(messages: Iterable[ChatMessage]) -> str:
    """Render OpenAI-style chat messages into a single prompt string."""

    system_prompt = DEFAULT_SYSTEM_PROMPT
    conversation: List[str] = []
    for message in messages:
        if message.role == "system":
            system_prompt = message.content
            continue
        role = "User" if message.role == "user" else "Assistant"
        conversation.append(f"{role}: {message.content.strip()}".strip())
    header = f"System: {system_prompt.strip()}\n\n"
    transcript = "\n".join(conversation)
    if transcript:
        transcript += "\n"
    transcript += "Assistant:"
    return header + transcript
