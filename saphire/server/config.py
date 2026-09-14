from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SAPHIRE_", env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./data/saphire.db"
    api_key: str = "dev-key"
    artifacts_dir: str = "./artifacts"
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    inline_jobs: bool = False  # run jobs in a background thread of the API process (dev / tests)
    worker_poll_s: float = 1.0
    default_project: str = "default"
    log_level: str = "info"

    @property
    def artifacts_path(self) -> Path:
        p = Path(self.artifacts_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
