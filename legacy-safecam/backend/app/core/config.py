from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "SafeCam AI Assistant"
    app_env: str = "local"
    database_url: str = Field(
        default="postgresql+psycopg://safecam:safecam@localhost:5432/safecam",
        alias="DATABASE_URL",
    )
    media_root: str = Field(default="./media", alias="MEDIA_ROOT")
    cors_origins: str = Field(
        default="http://localhost:5173,http://localhost:3000,http://localhost:8080",
        alias="CORS_ORIGINS",
    )
    clip_seconds: int = Field(default=10, alias="CLIP_SECONDS")
    embedding_dim: int = Field(default=384, alias="EMBEDDING_DIM")
    max_upload_mb: int = Field(default=512, alias="MAX_UPLOAD_MB")

    @property
    def media_path(self) -> Path:
        return Path(self.media_root).expanduser().resolve()

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

