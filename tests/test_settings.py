"""Tests for environment-driven settings parsing validators."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.settings import Settings


def test_cors_allow_origins_parses_comma_separated_values() -> None:
    settings = Settings.model_validate(
        {"CORS_ALLOW_ORIGINS": "https://a.example, https://b.example"}
    )
    assert settings.cors_allow_origins == ["https://a.example", "https://b.example"]


def test_cors_allow_origins_rejects_invalid_type() -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({"CORS_ALLOW_ORIGINS": 123})


def test_model_allow_list_parses_comma_separated_values() -> None:
    settings = Settings.model_validate({"MODEL_ALLOW_LIST": "GPT3-dev, GPT-2"})
    assert settings.model_allow_list == ["GPT3-dev", "GPT-2"]


def test_model_allow_list_rejects_invalid_type() -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({"MODEL_ALLOW_LIST": 123})
