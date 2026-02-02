"""Configuration settings for the application."""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings."""

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_title: str = "Prêt à Dépenser - Scoring API"
    api_version: str = "1.0.0"

    # Model Configuration
    model_path: Path = Path("models")
    model_name: str = "scoring_model.pkl"
    mlflow_tracking_uri: str = "mlruns"

    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "logs/api.log"

    # Data Drift Configuration
    reference_data_path: str = "data/reference_data.csv"
    production_data_path: str = "logs/predictions.csv"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()