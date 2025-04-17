from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    API_HOST: str = os.getenv("API_HOST", "localhost")
    API_PORT: int = int(os.getenv("API_PORT", 8000))

    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./app.db")

    # Redis Configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB: int = int(os.getenv("REDIS_DB", 0))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Model Configuration
    MODEL_PATH: str = os.getenv("MODEL_PATH", "app/models/best.pt")
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", 0.25))
    MAX_DETECTIONS: int = int(os.getenv("MAX_DETECTIONS", 10))
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "app/static/output")

    # Media Configuration
    MEDIA_ROOT: str = os.getenv("MEDIA_ROOT", "app/static/media")

    # Upload Limits
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", 100 * 1024 * 1024))  # 100MB by default

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

    class Config:
        env_file = ".env"
        extra = "allow"  # Allow extra fields in the environment

settings = Settings()