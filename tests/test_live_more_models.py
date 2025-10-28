"""Additional live API tests to exercise multiple models.

Skipped by default; set RUN_LIVE_API_TESTS=1 to enable.
This test is lenient: if a model is listed by /v1/models but is not
actually usable (e.g., missing HF auth or unavailable backend), the
test skips that case instead of failing the suite.
"""
from __future__ import annotations

import os
from typing import Set

import pytest
import httpx


RUN_LIVE = os.environ.get("RUN_LIVE_API_TESTS") == "1"
BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:5001")
PROMPT = "he is a doctor. His main goal is"

# Candidate models to probe. Each test checks /v1/models first and will
# skip if the model id is not listed by the server.
CANDIDATES = [
    "GPT3-dev-350m-2805",
    "GPT3-dev-125m-0104",
    "GPT3-dev-125m-1202",
    "GPT3-dev-125m-0612",
    "GPT3-dev",
    "GPT3-dev-125m",
    "GPT-2",
]


def _get_models(timeout: float = 10.0) -> Set[str]:
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(f"{BASE_URL}/v1/models")
        resp.raise_for_status()
        data = resp.json()
        return {item.get("id") for item in (data.get("data") or [])}


@pytest.mark.skipif(not RUN_LIVE, reason="set RUN_LIVE_API_TESTS=1 to run live API tests")
@pytest.mark.parametrize("model", CANDIDATES)
def test_completion_for_models(model: str) -> None:
    available = _get_models()
    if model not in available:
        pytest.skip(f"model {model} not listed by server; available={sorted(available)}")

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

    if resp.status_code != 200:
        pytest.skip(
            f"model {model} not usable on server (status={resp.status_code}): {resp.text[:500]}"
        )

    body = resp.json()
    assert body.get("model") == model
    choices = body.get("choices") or []
    assert len(choices) >= 1
    assert isinstance(choices[0].get("text"), str)
    usage = body.get("usage") or {}
    assert "total_tokens" in usage

