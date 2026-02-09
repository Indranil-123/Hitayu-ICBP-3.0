from pydantic import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Hitayu AI"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    

    #API
    API_V1_PREFIX: str = "/api/v1"

    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
