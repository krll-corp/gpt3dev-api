"""Endpoints for listing available models."""
from __future__ import annotations

from fastapi import APIRouter

from ..core.model_registry import list_models

router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models")
def list_available_models() -> dict:
    data = []
    for spec in list_models():
        data.append(
            {
                "id": spec.name,
                "object": "model",
                "owned_by": "owner",
                "permission": [],
            }
        )
    return {"object": "list", "data": data}
