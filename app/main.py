"""FastAPI application entrypoint."""
from __future__ import annotations

import logging
from logging.config import dictConfig
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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

app = FastAPI(title="GPT3dev OpenAI-Compatible API", version="1.0.0")

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


@app.exception_handler(HTTPException)
async def openai_http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict) and "message" in detail and "type" in detail:
        return JSONResponse(status_code=exc.status_code, content={"error": detail})
    return JSONResponse(status_code=exc.status_code, content={"detail": detail})
