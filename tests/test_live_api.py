"""Live API smoke tests hitting a running server.

Skipped by default; set RUN_LIVE_API_TESTS=1 to enable.
Configure API base via API_BASE_URL (default: http://localhost:5001).
"""
from __future__ import annotations

import os
from typing import List, Set

import pytest
import httpx


RUN_LIVE = os.environ.get("RUN_LIVE_API_TESTS") == "1"
BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:5001")
PROMPT = "he is a doctor. His main goal is"


def _get_models(timeout: float = 10.0) -> Set[str]:
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(f"{BASE_URL}/v1/models")
        resp.raise_for_status()
        data = resp.json()
        return {item["id"] for item in data.get("data", [])}


@pytest.mark.skipif(not RUN_LIVE, reason="set RUN_LIVE_API_TESTS=1 to run live API tests")
@pytest.mark.parametrize("model", ["GPT-2", "GPT3-dev-350m-2805"])  # adjust names as available
def test_completion_basic(model: str) -> None:
    available = _get_models()
    if model not in available:
        pytest.skip(f"model {model} not available on server; available={sorted(available)}")

    payload = {
        "model": model,
        "prompt": PROMPT,
        "max_tokens": 16,
        "temperature": 0.0,
    }
    # Allow generous timeout for first-run weight downloads
    timeout = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0)
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(f"{BASE_URL}/v1/completions", json=payload)
        resp.raise_for_status()
        body = resp.json()
    assert body.get("model") == model
    choices = body.get("choices") or []
    assert len(choices) >= 1
    assert isinstance(choices[0].get("text"), str)
    # The completion can be empty for some models with temperature=0, but should be a string
    usage = body.get("usage") or {}
    assert "total_tokens" in usage

