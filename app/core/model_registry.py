"""Model registry providing metadata for available engines."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

from .settings import get_settings
import os


@dataclass(frozen=True)
class ModelMetadata:
    """Descriptive metadata gathered from author-provided sources."""

    description: str
    parameter_count: Optional[str] = None
    training_datasets: Optional[str] = None
    training_tokens: Optional[str] = None
    training_steps: Optional[str] = None
    evaluation: Optional[str] = None
    notes: Optional[str] = None
    sources: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {"description": self.description}
        if self.parameter_count:
            data["parameter_count"] = self.parameter_count
        if self.training_datasets:
            data["training_datasets"] = self.training_datasets
        if self.training_tokens:
            data["training_tokens"] = self.training_tokens
        if self.training_steps:
            data["training_steps"] = self.training_steps
        if self.evaluation:
            data["evaluation"] = self.evaluation
        if self.notes:
            data["notes"] = self.notes
        if self.sources:
            data["sources"] = list(self.sources)
        return data


@dataclass(frozen=True)
class ModelSpec:
    """Metadata describing a supported model."""

    name: str
    hf_repo: str
    dtype: Optional[str] = None
    device: Optional[str] = None
    max_context_tokens: Optional[int] = None
    metadata: Optional[ModelMetadata] = None


_DEFAULT_MODELS: List[ModelSpec] = [
    ModelSpec(
        name="GPT4-dev-177M-1511",
        hf_repo="k050506koch/GPT4-dev-177M-1511",
        dtype="float16",
        device="auto",
        max_context_tokens=512,
        metadata=ModelMetadata(
            description="117M parameter GPT-4-inspired checkpoint released on 15-11-2025.",
            parameter_count="117M",
            training_datasets="HuggingFaceFW/fineweb",
            training_steps="78,000 steps · sequence length 512 · batch size 192 · Lion optimizer",
            evaluation="29.30% MMLU (author reported)",
            notes="Custom GPT-4-insopired architecture that requires trust_remote_code when loading.",
            sources=(
                "https://huggingface.co/k050506koch/GPT4-dev-177M-1511",
            ),
            sources=("https://huggingface.co/k050506koch/GPT4-dev-177M-1511",),
        ),
    ),
    ModelSpec(
        name="GPT3-dev-350m-2805",
        hf_repo="k050506koch/GPT3-dev-350m-2805",
        dtype="float16",
        device="auto",
        max_context_tokens=4096,
        metadata=ModelMetadata(
            description="350M parameter GPT-3-style checkpoint released on 2025-05-28.",
            parameter_count="350M",
            training_datasets="HuggingFaceFW/fineweb",
            training_steps="10,000 steps · sequence length 512 · batch size 192 · Lion optimizer",
            evaluation="28.55% MMLU (author reported)",
            notes="Custom GPT-3 architecture that requires trust_remote_code when loading.",
            sources=(
                "https://huggingface.co/k050506koch/GPT3-dev-350m-2805",
                "https://github.com/krll-corp/GPT3",
            ),
        ),
    ),
    ModelSpec(
        name="GPT3-dev-125m-0104",
        hf_repo="k050506koch/GPT3-dev-125m-0104",
        dtype="float16",
        device="auto",
        metadata=ModelMetadata(
            description="Fourth 125M checkpoint for the GPT3-dev series released 2025-01-04.",
            parameter_count="125M",
            training_datasets="HuggingFaceFW/fineweb",
            training_steps="65,000 steps · sequence length 512 · batch size 12 · grad_accum 4 · Lion optimizer",
            evaluation="28.65% MMLU (author reported)",
            notes="Shares custom GPT-3 style architecture and instruct template guidance in repository README.",
            sources=(
                "https://huggingface.co/k050506koch/GPT3-dev-125m-0104",
                "https://github.com/krll-corp/GPT3",
            ),
        ),
    ),
    ModelSpec(
        name="GPT3-dev-125m-1202",
        hf_repo="k050506koch/GPT3-dev-125m-1202",
        dtype="float16",
        device="auto",
        metadata=ModelMetadata(
            description="Third 125M checkpoint from December 2, 2025 with Lion optimizer fine-tuning.",
            parameter_count="125M",
            training_datasets="HuggingFaceFW/fineweb",
            training_steps="36,500 steps · sequence length 512 · batch size 12 · grad_accum 4 · Lion optimizer",
            evaluation="28.03% MMLU (author reported)",
            notes="Requires trust_remote_code; serves as base for later instruct-tuned releases.",
            sources=(
                "https://huggingface.co/k050506koch/GPT3-dev-125m-1202",
                "https://github.com/krll-corp/GPT3",
            ),
        ),
    ),
    ModelSpec(
        name="GPT3-dev-125m-0612",
        hf_repo="k050506koch/GPT3-dev-125m-0612",
        dtype="float16",
        device="auto",
        metadata=ModelMetadata(
            description="June 12, 2024 125M checkpoint trained longer with gradient accumulation.",
            parameter_count="125M",
            training_datasets="HuggingFaceFW/fineweb",
            training_steps="600,000 steps · sequence length 512 · batch size 12 · grad_accum 4",
            evaluation="27.65% MMLU (author reported)",
            notes="Custom GPT-3 style architecture distributed with gguf exports alongside Transformers weights.",
            sources=(
                "https://huggingface.co/k050506koch/GPT3-dev-125m-0612",
                "https://github.com/krll-corp/GPT3",
            ),
        ),
    ),
    ModelSpec(
        name="GPT3-dev",
        hf_repo="k050506koch/GPT3-dev",
        dtype="float16",
        device="auto",
        metadata=ModelMetadata(
            description="17M parameter architecture demonstrator for the GPT3-dev project.",
            parameter_count="17M",
            training_datasets="HuggingFaceFW/fineweb",
            notes="Early experimental checkpoint intended to showcase the custom GPT-3 style stack.",
            sources=(
                "https://huggingface.co/k050506koch/GPT3-dev",
                "https://github.com/krll-corp/GPT3",
            ),
        ),
    ),
    ModelSpec(
        name="GPT3-dev-125m",
        hf_repo="k050506koch/GPT3-dev-125m",
        dtype="float16",
        device="auto",
        metadata=ModelMetadata(
            description="Early 125M parameter GPT3-dev checkpoint trained on roughly 3.6B tokens.",
            parameter_count="125M",
            training_datasets="HuggingFaceFW/fineweb",
            training_tokens="≈3.6B tokens",
            notes="Technology demonstrator preceding the longer 2024-2025 training runs.",
            sources=(
                "https://huggingface.co/k050506koch/GPT3-dev-125m",
                "https://github.com/krll-corp/GPT3",
            ),
        ),
    ),
    ModelSpec(
        name="GPT-2",
        hf_repo="openai-community/gpt2",
        dtype="float32",
        device="auto",
        metadata=ModelMetadata(
            description="No additional details provided.",
        ),
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
        metadata_entry = entry.get("metadata")
        metadata = None
        if isinstance(metadata_entry, dict) and metadata_entry.get("description"):
            metadata = ModelMetadata(
                description=metadata_entry["description"],
                parameter_count=metadata_entry.get("parameter_count"),
                training_datasets=metadata_entry.get("training_datasets"),
                training_tokens=metadata_entry.get("training_tokens"),
                training_steps=metadata_entry.get("training_steps"),
                evaluation=metadata_entry.get("evaluation"),
                notes=metadata_entry.get("notes"),
                sources=tuple(metadata_entry.get("sources", ()) or ()),
            )
        specs.append(
            ModelSpec(
                name=entry["name"],
                hf_repo=entry["hf_repo"],
                dtype=entry.get("dtype"),
                device=entry.get("device"),
                max_context_tokens=entry.get("max_context_tokens"),
                metadata=metadata,
            )
        )
    return specs


def _initialize_registry() -> None:
    if _registry:
        return
    settings = get_settings()
    # Decide source of model specs based on configuration.
    specs: List[ModelSpec] = []
    model_dump = settings.model_dump() if hasattr(settings, "model_dump") else {}
    registry_path_value = model_dump.get("model_registry_path")
    raw_include = (
        model_dump.get("include_default_models")
        if "include_default_models" in model_dump
        else getattr(settings, "include_default_models", None)
    )
    if raw_include is None:
        # Auto mode: enable defaults unless running under pytest
        include_defaults = os.environ.get("PYTEST_CURRENT_TEST") is None
    else:
        include_defaults = bool(raw_include)
    file_specs: List[ModelSpec] = []
    if registry_path_value:
        registry_path = Path(registry_path_value)
        if registry_path.exists():
            file_specs = list(_load_registry_from_file(registry_path))
        else:
            raise FileNotFoundError(f"MODEL_REGISTRY_PATH not found: {registry_path}")
    if include_defaults:
        specs.extend(_DEFAULT_MODELS)
    specs.extend(file_specs)
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
