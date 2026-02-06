"""Tests for the dynamic model registry filtering."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pydantic
import pytest

fake_yaml = types.ModuleType("yaml")
fake_yaml.safe_load = lambda data: []
sys.modules.setdefault("yaml", fake_yaml)

fake_pydantic_settings = types.ModuleType("pydantic_settings")


class _FakeBaseSettings(pydantic.BaseModel):
    model_config: dict = {}

    def model_dump(self, *args, **kwargs):  # pragma: no cover - passthrough
        return super().model_dump(*args, **kwargs)


fake_pydantic_settings.BaseSettings = _FakeBaseSettings
fake_pydantic_settings.SettingsConfigDict = dict
sys.modules.setdefault("pydantic_settings", fake_pydantic_settings)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core import model_registry


class DummySettings:
    """Minimal settings stand-in for registry tests."""

    def __init__(
        self,
        *,
        allow_list: list[str] | None = None,
        registry_path: str | None = None,
        include_defaults: bool | None = None,
    ) -> None:
        self.model_allow_list = allow_list
        self._registry_path = registry_path
        self.include_default_models = include_defaults

    def model_dump(self) -> dict:
        data = {"model_registry_path": self._registry_path}
        if self.include_default_models is not None:
            data["include_default_models"] = self.include_default_models
        return data


@pytest.fixture(autouse=True)
def reset_registry(monkeypatch):
    def apply(
        *,
        allow_list: list[str] | None = None,
        registry_path: str | None = None,
        include_defaults: bool | None = None,
    ) -> None:
        dummy = DummySettings(
            allow_list=allow_list,
            registry_path=registry_path,
            include_defaults=include_defaults,
        )
        monkeypatch.setattr(model_registry, "get_settings", lambda: dummy, raising=False)
        model_registry._registry.clear()

    apply()
    yield apply
    apply()


def test_default_registry_is_empty(reset_registry):
    names = {spec.name for spec in model_registry.list_models()}
    assert names == set()


def test_model_allow_list_filters(reset_registry, tmp_path: Path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            [
                {"name": "GPT3-dev", "hf_repo": "dummy/dev"},
                {"name": "Tiny", "hf_repo": "dummy/tiny"},
            ]
        )
    )
    reset_registry(registry_path=str(registry_path))
    names = {spec.name for spec in model_registry.list_models()}
    assert names == {"GPT3-dev", "Tiny"}

    reset_registry(allow_list=["GPT3-dev"], registry_path=str(registry_path))
    names = {spec.name for spec in model_registry.list_models()}
    assert names == {"GPT3-dev"}


def test_model_allow_list_unknown_model(reset_registry):
    reset_registry(allow_list=["unknown"])
    with pytest.raises(KeyError):
        model_registry.list_models()


def test_custom_registry_can_extend_defaults(reset_registry, tmp_path: Path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps([{"name": "Tiny", "hf_repo": "dummy/tiny"}])
    )
    reset_registry(registry_path=str(registry_path), include_defaults=True)
    names = {spec.name for spec in model_registry.list_models()}
    assert "Tiny" in names
    assert "GPT3-dev" in names


def test_registry_loads_yaml_when_json_parse_fails(
    reset_registry,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text("- name: Tiny\n  hf_repo: dummy/tiny\n")

    def fake_safe_load(data: str) -> list[dict[str, str]]:
        assert "name: Tiny" in data
        return [{"name": "Tiny", "hf_repo": "dummy/tiny"}]

    monkeypatch.setattr(model_registry.yaml, "safe_load", fake_safe_load)

    reset_registry(registry_path=str(registry_path))
    names = {spec.name for spec in model_registry.list_models()}
    assert names == {"Tiny"}


def test_registry_rejects_non_list_file_payload(reset_registry, tmp_path: Path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"name": "Tiny", "hf_repo": "dummy/tiny"}))

    reset_registry(registry_path=str(registry_path))
    with pytest.raises(ValueError, match="must contain a list"):
        model_registry.list_models()


def test_registry_rejects_non_object_entries(reset_registry, tmp_path: Path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(["not-an-object"]))

    reset_registry(registry_path=str(registry_path))
    with pytest.raises(ValueError, match="entries must be objects"):
        model_registry.list_models()


def test_registry_path_missing_raises_file_not_found(reset_registry, tmp_path: Path):
    reset_registry(registry_path=str(tmp_path / "missing.json"))

    with pytest.raises(FileNotFoundError, match="MODEL_REGISTRY_PATH not found"):
        model_registry.list_models()


def test_file_registry_overrides_default_model_with_same_name(reset_registry, tmp_path: Path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps([{"name": "GPT3-dev", "hf_repo": "custom/override"}])
    )

    reset_registry(registry_path=str(registry_path), include_defaults=True)

    spec = model_registry.get_model_spec("GPT3-dev")
    names = {item.name for item in model_registry.list_models()}
    assert "GPT-2" in names
    assert spec.hf_repo == "custom/override"
