"""Runtime configuration via pydantic-settings (reads env / .env)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://orderflow:orderflow@localhost:5432/orderflow"
    temporal_target: str = "localhost:7233"
    task_queue: str = "order-processing"
    api_host: str = "0.0.0.0"
    api_port: int = 8000


def get_settings() -> Settings:
    return Settings()
