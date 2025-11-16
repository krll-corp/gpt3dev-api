"""Endpoints for listing available models."""
from __future__ import annotations

from fastapi import APIRouter

from ..core.errors import model_not_found
from ..core.model_registry import ModelSpec, get_model_spec, list_models

router = APIRouter(prefix="/v1", tags=["models"])


def _serialize_model(spec: ModelSpec, include_metadata: bool = False) -> dict:
    payload = {
        "id": spec.name,
        "object": "model",
        "owned_by": "owner",
        "permission": [],
    }
    if include_metadata:
        metadata = spec.metadata.to_dict() if spec.metadata else {"description": "No additional details provided."}
        metadata.setdefault("huggingface_repo", spec.hf_repo)
        if spec.max_context_tokens is not None:
            metadata.setdefault("max_context_tokens", spec.max_context_tokens)
        if spec.dtype:
            metadata.setdefault("dtype", spec.dtype)
        if spec.device:
            metadata.setdefault("default_device", spec.device)
        payload["metadata"] = metadata
    return payload


@router.get("/models")
def list_available_models() -> dict:
    data = [_serialize_model(spec, include_metadata=False) for spec in list_models()]
    return {"object": "list", "data": data}


@router.get("/models/{model_id}")
def retrieve_model(model_id: str) -> dict:
    try:
        spec = get_model_spec(model_id)
    except KeyError:
        raise model_not_found(model_id)
    return _serialize_model(spec, include_metadata=True)
