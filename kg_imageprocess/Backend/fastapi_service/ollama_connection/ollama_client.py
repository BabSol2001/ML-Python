import os
import requests
import httpx
from typing import List, Dict, Any, Optional
from config import settings


class OllamaService:
    def __init__(self):
        raw_url = getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")
        self.base_url = raw_url.rstrip('/').replace('/v1', '')
        self.model = getattr(settings, "OLLAMA_MODEL", "qwen2.5:7b")

    def check_connection(self) -> bool:
        """بررسی برقرار بودن سرویس محلی Ollama"""
        try:
            response = requests.get(self.base_url, timeout=3)
            return response.status_code == 200
        except Exception:
            return False

    def set_model(self, model_name: str):
        self.model = model_name

    def setup_environment(self):
        """تنظیم تمام متغیرهای محیطی برای اجبار Graphiti و OpenAI Client به Ollama محلی"""
        self.model = getattr(settings, "OLLAMA_MODEL", self.model)
        openai_compatible_url = f"{self.base_url}/v1"

        os.environ["OPENAI_API_KEY"] = "ollama"
        os.environ["OPENAI_BASE_URL"] = openai_compatible_url
        os.environ["OPENAI_API_BASE"] = openai_compatible_url

        os.environ["OPENAI_MODEL_NAME"] = self.model
        os.environ["MODEL_NAME"] = self.model
        os.environ["LLM_MODEL"] = self.model

        os.environ["EMBEDDING_MODEL_NAME"] = "nomic-embed-text"
        os.environ["EMBEDDING_MODEL"] = "nomic-embed-text"
        
        print(f"🔗 اتصال Ollama با مدل '{self.model}' روی '{openai_compatible_url}' تنظیم شد.")

    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: float = 300.0
    ) -> Optional[str]:
        """ارسال درخواست غیرهمگام Chat Completion به API سازگار با OpenAI در Ollama"""
        endpoint = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,  # برای جلوگیری از قطع شدن پاسخ در لایه OpenAI-compatible
            "options": {
                "num_predict": max_tokens  # برای اطمینان در ساختار بومی Ollama
            }
        }

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(endpoint, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    print(f"⚠️ Ollama Error Code: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"⚠️ خطای ارتباط با سرویس Ollama: {e}")
        
        return None


ollama_service = OllamaService()