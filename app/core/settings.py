"""Application settings loaded from environment variables."""
from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the API service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    hf_token: Optional[str] = Field(default=None, validation_alias="HF_TOKEN")
    default_device: str = Field(default="auto", validation_alias="DEFAULT_DEVICE")
    max_model_workers: int = Field(
        default=1, validation_alias="MAX_MODEL_WORKERS"
    )
    max_context_tokens: int = Field(
        default=2048, validation_alias="MAX_CONTEXT_TOKENS"
    )
    enable_embeddings_backend: bool = Field(
        default=False, validation_alias="ENABLE_EMBEDDINGS_BACKEND"
    )
    cors_allow_origins: List[str] = Field(
        default_factory=list, validation_alias="CORS_ALLOW_ORIGINS"
    )
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    model_registry_path: Optional[str] = Field(
        default=None, validation_alias="MODEL_REGISTRY_PATH"
    )

    @field_validator("cors_allow_origins", mode="before")
    def _split_origins(cls, value: object) -> List[str]:  # noqa: D401, N805
        """Allow comma separated CORS origins from the environment."""
        if not value:
            return []
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        if isinstance(value, list):
            return value
        raise ValueError("Invalid CORS origins configuration")


@lru_cache()
def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()
