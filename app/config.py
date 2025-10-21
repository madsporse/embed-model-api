from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """
    Application configuration settings.
    
    Loads settings from environment variables with 'EMB_' prefix.
    Falls back to default values if environment variables are not set.
    """
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    MODEL_ID: str = "intfloat/multilingual-e5-large"
    MODEL_PATH: str = "models/multilingual-e5-large"
    USE_LOCAL_MODEL: bool = False
    MAX_BATCH: int = 128
    MAX_CHARS_PER_ITEM: int = 8000
    DEFAULT_BATCH_SIZE: int = 32
    CORS_ALLOW_ALL: bool = True

    model_config = SettingsConfigDict(env_prefix="EMB_", case_sensitive=False)


settings = Settings()
