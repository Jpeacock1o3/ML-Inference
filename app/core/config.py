from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    db_host: str = "localhost"
    db_port: int = 3306
    db_name: str = "ml_inference"
    db_user: str = "ml_user"
    db_password: str = "ml_password"
    db_pool_size: int = 20
    db_max_overflow: int = 10

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_ttl: int = 300

    model_path: str = "./models/classifier.joblib"

    rate_limit_requests: int = 100
    rate_limit_window: int = 60

    @property
    def database_url(self) -> str:
        return (
            f"mysql+aiomysql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
