"""Tests for the dynamic model registry filtering."""
from __future__ import annotations

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

    def __init__(self, *, allow_list: list[str] | None = None, registry_path: str | None = None) -> None:
        self.model_allow_list = allow_list
        self._registry_path = registry_path

    def model_dump(self) -> dict:
        return {"model_registry_path": self._registry_path}


@pytest.fixture(autouse=True)
def reset_registry(monkeypatch):
    def apply(*, allow_list: list[str] | None = None, registry_path: str | None = None) -> None:
        dummy = DummySettings(allow_list=allow_list, registry_path=registry_path)
        monkeypatch.setattr(model_registry, "get_settings", lambda: dummy, raising=False)
        model_registry._registry.clear()

    apply()
    yield apply
    apply()


def test_default_registry_includes_all_models(reset_registry):
    names = {spec.name for spec in model_registry.list_models()}
    assert names == {"GPT3-dev"}


def test_model_allow_list_filters(reset_registry):
    reset_registry(allow_list=["GPT3-dev"])
    names = {spec.name for spec in model_registry.list_models()}
    assert names == {"GPT3-dev"}


def test_model_allow_list_unknown_model(reset_registry):
    reset_registry(allow_list=["unknown"])
    with pytest.raises(KeyError):
        model_registry.list_models()
