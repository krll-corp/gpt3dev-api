"""Model registry providing metadata for available engines."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

from .settings import get_settings


@dataclass(frozen=True)
class ModelSpec:
    """Metadata describing a supported model."""

    name: str
    hf_repo: str
    dtype: Optional[str] = None
    device: Optional[str] = None
    max_context_tokens: Optional[int] = None


_DEFAULT_MODELS: List[ModelSpec] = [
    ModelSpec(
        name="GPT3-dev-350m-2805",
        hf_repo="k050506koch/GPT3-dev-350m-2805",  # TODO confirm
        dtype="float16",
        device="auto",
        max_context_tokens=4096,
    ),
    ModelSpec(
        name="GPT3-dev-125m-0104",
        hf_repo="k050506koch/GPT3-dev-125m-0104",  # TODO confirm
        dtype="float16",
        device="auto",
    ),
    ModelSpec(
        name="GPT3-dev-125m-1202",
        hf_repo="k050506koch/GPT3-dev-125m-1202",  # TODO confirm
        dtype="float16",
        device="auto",
    ),
    ModelSpec(
        name="GPT3-dev-125m-0612",
        hf_repo="k050506koch/GPT3-dev-125m-0612",  # TODO confirm
        dtype="float16",
        device="auto",
    ),
    ModelSpec(
        name="GPT3-dev",
        hf_repo="k050506koch/GPT3-dev",  # TODO confirm
        dtype="float16",
        device="auto",
    ),
    ModelSpec(
        name="GPT3-dev-125m",
        hf_repo="k050506koch/GPT3-dev-125m",  # TODO confirm
        dtype="float16",
        device="auto",
    ),
    ModelSpec(
        name="GPT-2",
        hf_repo="openai-community/gpt2",  # TODO confirm
        dtype="float32",
        device="auto",
    ),
]

_registry_lock = threading.Lock()
_registry: Dict[str, ModelSpec] = {}


def _load_registry_from_file(path: Path) -> Iterable[ModelSpec]:
    data = path.read_text()
    try:
        loaded = json.loads(data)
    except json.JSONDecodeError:
        loaded = yaml.safe_load(data)
    if not isinstance(loaded, list):
        raise ValueError("Model registry file must contain a list of model specs")
    specs: List[ModelSpec] = []
    for entry in loaded:
        if not isinstance(entry, dict):
            raise ValueError("Model registry entries must be objects")
        specs.append(
            ModelSpec(
                name=entry["name"],
                hf_repo=entry["hf_repo"],
                dtype=entry.get("dtype"),
                device=entry.get("device"),
                max_context_tokens=entry.get("max_context_tokens"),
            )
        )
    return specs


def _initialize_registry() -> None:
    if _registry:
        return
    settings = get_settings()
    specs: List[ModelSpec] = list(_DEFAULT_MODELS)
    registry_path_value = settings.model_dump().get("model_registry_path")
    if registry_path_value:
        registry_path = Path(registry_path_value)
        if registry_path.exists():
            specs = list(_load_registry_from_file(registry_path))
        else:
            raise FileNotFoundError(f"MODEL_REGISTRY_PATH not found: {registry_path}")
    allow_list = None
    if settings.model_allow_list:
        allow_list = {name for name in settings.model_allow_list}
    for spec in specs:
        if allow_list is not None and spec.name not in allow_list:
            continue
        _registry[spec.name] = spec
    if allow_list is not None:
        missing = allow_list.difference(_registry)
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise KeyError(
                f"MODEL_ALLOW_LIST references unknown models: {missing_str}"
            )


def _ensure_registry_loaded() -> None:
    if _registry:
        return
    with _registry_lock:
        if not _registry:
            _initialize_registry()


def list_models() -> List[ModelSpec]:
    """Return all known model specifications."""

    _ensure_registry_loaded()
    return list(_registry.values())


def get_model_spec(model_name: str) -> ModelSpec:
    """Return the specification for ``model_name`` or raise ``KeyError``."""

    _ensure_registry_loaded()
    try:
        return _registry[model_name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Unknown model '{model_name}'") from exc
