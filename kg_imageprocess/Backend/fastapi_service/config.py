import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # تنظیمات عمومی پروژه
    PROJECT_NAME: str = "Live Motion AI - Pose Processing"
    DEBUG: bool = True

    # تنظیمات اتصال به Neo4j
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password123")
    NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "kgimageprocessdb")
    CYPHER_DEFAULT_LANGUAGE_VERSION: str = "CYPHER_2025"
    
    # تنظیمات Ollama (تنظیم روی آدرس پایه و تغییر مدل به qwen2.5:7b)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")  # 👈 به‌روزرسانی شده به qwen2.5:7b

    # آستانه‌های بیومکانیکی (Biomechanical Thresholds)
    SQUAT_KNEE_ANGLE_THRESHOLD: float = 90.0

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
