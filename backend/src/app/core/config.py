from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    MODEL_PATH: str = "models/model.joblib"
    PIPELINE_PATH: str = "models/pipeline.joblib"
    MODEL_VERSION: str = "v0.0.1"
    THRESHOLD: float = 0.8
    DB_URL: str | None = None
    API_KEY: str | None = None


settings = Settings()
