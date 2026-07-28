"""Environment-based configuration. Single source of truth for all settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FINRL_", env_file=".env", extra="ignore")

    # Which trained model the API serves
    algorithm: str = "ppo"  # ppo | dqn
    reward_mode: str = "return_minus_drawdown"  # return_minus_drawdown | risk_adjusted
    model_dir: str = "./models"

    # Observability
    log_level: str = "INFO"
    service_name: str = "finrl-portfolio"


def get_settings() -> Settings:
    return Settings()
