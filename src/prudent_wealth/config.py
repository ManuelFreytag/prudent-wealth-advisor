"""Application configuration loaded from environment variables."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra environment variables
    )

    # Gemini API
    google_api_key: str = Field(validation_alias="GEMINI_API_KEY")
    gemini_model: str = "gemini-3-pro-preview"

    # Database (optional for dev, required for prod)
    database_url: str | None = None

    # API Authentication
    api_token: str

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False


settings = Settings()
