"""Behavioral tests for app.main helpers and handlers."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from fastapi import HTTPException
from starlette.requests import Request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import main


def _dummy_request() -> Request:
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "path": "/",
        "headers": [],
        "query_string": b"",
    }
    return Request(scope)


def test_root_returns_ok_when_no_failures(monkeypatch) -> None:
    monkeypatch.setattr(main, "_endpoint_status", {"failures": {}, "last_checked": None})
    payload = asyncio.run(main.root())
    assert payload == {"status": "ok", "message": "GPT3dev API is running"}


def test_root_returns_degraded_with_sorted_issues_and_last_checked(monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "_endpoint_status",
        {
            "failures": {
                "/v1/zeta": {"status_code": 503},
                "/v1/alpha": {"status_code": 500, "detail": "boom"},
            },
            "last_checked": "2026-02-05T12:00:00+00:00",
        },
    )
    payload = asyncio.run(main.root())

    assert payload["status"] == "degraded"
    assert payload["message"] == "GPT3dev API is running"
    assert payload["issues"][0]["endpoint"] == "/v1/alpha"
    assert payload["issues"][1]["endpoint"] == "/v1/zeta"
    assert payload["last_checked"] == "2026-02-05T12:00:00+00:00"


def test_openai_exception_handler_wraps_error_payload() -> None:
    exc = HTTPException(
        status_code=400,
        detail={
            "message": "bad request",
            "type": "invalid_request_error",
            "param": "model",
            "code": "bad_model",
        },
    )
    response = asyncio.run(main.openai_http_exception_handler(_dummy_request(), exc))

    assert response.status_code == 400
    assert json.loads(response.body.decode("utf-8")) == {
        "error": {
            "message": "bad request",
            "type": "invalid_request_error",
            "param": "model",
            "code": "bad_model",
        }
    }


def test_openai_exception_handler_preserves_generic_detail() -> None:
    exc = HTTPException(status_code=422, detail="unprocessable")
    response = asyncio.run(main.openai_http_exception_handler(_dummy_request(), exc))

    assert response.status_code == 422
    assert json.loads(response.body.decode("utf-8")) == {"detail": "unprocessable"}
