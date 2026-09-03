import os
import requests
from config import settings


class OllamaService:
    def __init__(self):
        # پاک‌سازی و استانداردسازی آدرس پایه (حذف /v1 و اسلش پایانی)
        raw_url = getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")
        self.base_url = raw_url.rstrip('/').replace('/v1', '')
        
        # خواندن مدل اصلی از settings (که اکنون qwen2.5:7b است)
        self.model = getattr(settings, "OLLAMA_MODEL", "qwen2.5:7b")

    def check_connection(self) -> bool:
        """بررسی برقرار بودن سرویس محلی Ollama"""
        try:
            response = requests.get(self.base_url, timeout=3)
            return response.status_code == 200
        except Exception:
            return False

    def set_model(self, model_name: str):
        """امکان تغییر دینامیک مدل در صورت نیاز"""
        self.model = model_name

    def setup_environment(self):
        """
        تنظیم تمام متغیرهای محیطی برای اجبار Graphiti، OpenAI Client و Embedder
        به استفاده از Ollama محلی.
        """
        # همگام‌سازی آخرین مقدار مدل از settings
        self.model = getattr(settings, "OLLAMA_MODEL", self.model)
        
        # آدرس استاندارد OpenAI API برای Ollama همراه با v1
        openai_compatible_url = f"{self.base_url}/v1"

        os.environ["OPENAI_API_KEY"] = "ollama"
        os.environ["OPENAI_BASE_URL"] = openai_compatible_url
        os.environ["OPENAI_API_BASE"] = openai_compatible_url

        # تنظیمات مدل اصلی LLM
        os.environ["OPENAI_MODEL_NAME"] = self.model
        os.environ["MODEL_NAME"] = self.model
        os.environ["LLM_MODEL"] = self.model

        # تنظیمات مدل Embedder جهت جلوگیری از فراخوانی اشتباه مدل‌های آنلاین
        os.environ["EMBEDDING_MODEL_NAME"] = "nomic-embed-text"
        os.environ["EMBEDDING_MODEL"] = "nomic-embed-text"
        
        print(f"🔗 اتصال Ollama با مدل '{self.model}' روی '{openai_compatible_url}' تنظیم شد.")


ollama_service = OllamaService()