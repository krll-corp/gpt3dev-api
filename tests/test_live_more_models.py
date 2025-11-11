"""Additional live API tests to exercise multiple models.

Skipped by default; set RUN_LIVE_API_TESTS=1 to enable.
The suite skips candidates that are missing from /v1/models but now
fails whenever the live API returns an error so issues surface in CI.
"""
from __future__ import annotations

import os
import warnings
from functools import lru_cache
from typing import Set

import pytest
import httpx


RUN_LIVE = os.environ.get("RUN_LIVE_API_TESTS") == "1"
BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:5001")
VERIFY_SSL = os.environ.get("API_VERIFY_SSL", "1") != "0"
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


@lru_cache(maxsize=1)
def _get_models(timeout: float = 10.0) -> Set[str]:
    with httpx.Client(timeout=timeout, verify=VERIFY_SSL) as client:
        resp = client.get(f"{BASE_URL}/v1/models")
        resp.raise_for_status()
        data = resp.json()
        models = {item.get("id") for item in (data.get("data") or [])}

    if not models:
        pytest.fail(f"/v1/models returned no data from {BASE_URL}")

    available_candidates = sorted(models & set(CANDIDATES))
    if not available_candidates:
        pytest.fail(
            "None of the candidate models are exposed by the API. "
            f"Available={sorted(models)} candidates={CANDIDATES}"
        )

    print(
        f"[live-more-models] candidates under test: {', '.join(available_candidates)}"
    )
    return models


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
    with httpx.Client(timeout=timeout, verify=VERIFY_SSL) as client:
        resp = client.post(f"{BASE_URL}/v1/completions", json=payload)

    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:  # pragma: no cover - only hit when live API misbehaves
        pytest.fail(
            f"model {model} returned an error response: "
            f"{exc.response.status_code} {exc.response.text[:500]}"
        )

    body = resp.json()
    assert body.get("model") == model
    choices = body.get("choices") or []
    assert len(choices) >= 1
    text = choices[0].get("text", "")
    assert isinstance(text, str)
    assert text.strip(), "Model returned empty completion text"
    message = f"[live-more-models] {model} generated: {text[:120]!r}"
    print(message)
    warnings.warn(message, stacklevel=1)
    usage = body.get("usage") or {}
    assert "total_tokens" in usage

