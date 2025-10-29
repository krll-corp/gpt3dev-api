"""FastAPI application entrypoint."""
from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime, timezone
from logging.config import dictConfig
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from .core.settings import get_settings
from .routers import chat, completions, embeddings, models


def configure_logging(level: str) -> None:
    config: Dict[str, Any] = {
        "version": 1,
        "formatters": {
            "default": {
                "format": "%(levelname)s [%(name)s] %(message)s",
            }
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "default",
            }
        },
        "root": {"handlers": ["default"], "level": level},
    }
    dictConfig(config)


settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title="GPT3dev OpenAI-Compatible API", version="1.0.0")

CHECK_INTERVAL_SECONDS = 60
IGNORED_MONITOR_PATHS = {"/"}

EndpointStatus = Dict[str, Dict[str, Any]]

_endpoint_status: Dict[str, Any] = {"failures": {}, "last_checked": None}
_endpoint_monitor_task: Optional[asyncio.Task[None]] = None


def _monitored_endpoints() -> List[str]:
    endpoints: List[str] = []
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        if "GET" not in (route.methods or set()):
            continue
        if route.path in IGNORED_MONITOR_PATHS:
            continue
        if route.dependant.path_params:
            continue
        if not route.include_in_schema:
            continue
        endpoints.append(route.path)
    return sorted(set(endpoints))


async def _poll_endpoint_health() -> None:
    previous_failures: set[str] = set()
    async with httpx.AsyncClient(app=app, base_url="http://status-check", timeout=10.0) as client:
        while True:
            try:
                monitored_paths = _monitored_endpoints()
                failures: Dict[str, Dict[str, Any]] = {}
                for path in monitored_paths:
                    try:
                        response = await client.get(path)
                    except httpx.HTTPError as exc:
                        failures[path] = {"error": str(exc)}
                        continue
                    except Exception as exc:  # pragma: no cover - defensive
                        failures[path] = {"error": str(exc)}
                        continue
                    if not 200 <= response.status_code < 400:
                        failures[path] = {
                            "status_code": response.status_code,
                            "detail": response.text[:200],
                        }
                _endpoint_status["failures"] = failures
                _endpoint_status["last_checked"] = datetime.now(timezone.utc).isoformat()
                current_failures = set(failures.keys())
                if current_failures != previous_failures:
                    if current_failures:
                        logger.warning("Endpoint monitor detected failures: %s", sorted(current_failures))
                    elif previous_failures:
                        logger.info("All monitored endpoints restored")
                    previous_failures = current_failures
                await asyncio.sleep(CHECK_INTERVAL_SECONDS)
            except asyncio.CancelledError:  # pragma: no cover - shutdown handling
                raise
            except Exception:  # pragma: no cover - defensive logging only
                logger.exception("Unexpected error during endpoint monitoring")
                await asyncio.sleep(CHECK_INTERVAL_SECONDS)


def _ensure_monitor_task() -> None:
    global _endpoint_monitor_task
    if _endpoint_monitor_task is None or _endpoint_monitor_task.done():
        _endpoint_monitor_task = asyncio.create_task(_poll_endpoint_health())

if settings.cors_allow_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

app.include_router(models.router)
app.include_router(completions.router)
app.include_router(chat.router)
app.include_router(embeddings.router)


@app.get("/healthz")
async def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint used by platform health checks (e.g., HF Spaces)."""
    base_response: Dict[str, Any] = {"status": "ok", "message": "GPT3dev API is running"}
    failures: EndpointStatus = _endpoint_status.get("failures", {})
    if not failures:
        return base_response
    degraded_response = dict(base_response)
    degraded_response["status"] = "degraded"
    degraded_response["issues"] = [
        {"endpoint": path, **details} for path, details in sorted(failures.items())
    ]
    last_checked = _endpoint_status.get("last_checked")
    if last_checked:
        degraded_response["last_checked"] = last_checked
    return degraded_response


@app.on_event("startup")
async def on_startup() -> None:
    # Light-weight startup log to confirm the server is up
    try:
        from .core.model_registry import list_models

        models = ", ".join(spec.name for spec in list_models())
    except Exception:  # pragma: no cover - defensive logging only
        models = "(unavailable)"
    logger.info("API startup complete. Log level=%s. Models=[%s]", settings.log_level, models)
    _ensure_monitor_task()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    global _endpoint_monitor_task
    if _endpoint_monitor_task is not None:
        _endpoint_monitor_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _endpoint_monitor_task
        _endpoint_monitor_task = None


@app.exception_handler(HTTPException)
async def openai_http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict) and "message" in detail and "type" in detail:
        return JSONResponse(status_code=exc.status_code, content={"error": detail})
    return JSONResponse(status_code=exc.status_code, content={"detail": detail})
