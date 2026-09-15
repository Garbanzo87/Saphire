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
    # --- auth / SSO ---
    jwt_secret: str = "change-me-in-production"
    jwt_ttl_s: int = 12 * 3600
    dev_login: bool = False  # enable POST /v1/auth/dev-login (mint a JWT for any email) for local dev & tests
    superadmin_emails: list[str] = []
    oidc_issuer: str = ""  # e.g. https://accounts.google.com or https://login.microsoftonline.com/<tenant>/v2.0 or Okta
    oidc_client_id: str = ""
    oidc_client_secret: str = ""
    oidc_redirect_url: str = "http://localhost:8000/v1/auth/callback"
    oidc_scopes: str = "openid email profile"
    frontend_url: str = "http://localhost:3000"
    # --- limits / retention ---
    rate_limit_rpm: int = 0  # 0 = disabled; org.quotas.requests_per_minute overrides
    trust_proxy: bool = False  # honour X-Forwarded-For (first hop) for API-key IP allowlists when behind a load balancer
    retention_days: int = 0  # 0 = keep forever; `saphire retention` / retention job deletes older traces & rollouts
    audit_request_bodies: bool = False  # store (redacted, truncated) request bodies in the audit log

    @property
    def artifacts_path(self) -> Path:
        p = Path(self.artifacts_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
