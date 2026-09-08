import logging
from typing import Dict, List, Union, Optional
from openai import AsyncOpenAI
from .config_translator import TranslationConfig

logger = logging.getLogger("StandaloneTranslator")


class StandaloneTranslator:
    """
    موتور ترجمه مستقل و هوشمند با استفاده از مدل‌های زبانی (LLM).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.base_url = base_url or TranslationConfig.OLLAMA_BASE_URL
        self.model_name = model_name or TranslationConfig.OLLAMA_MODEL
        self.api_key = api_key or TranslationConfig.API_KEY

        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

    async def translate(
        self,
        text_or_list: Union[str, List[str]],
        target_languages: List[str] = ["fa"],
        domain_context: str = "Biomechanics & Sports Science"
    ) -> Dict[str, str]:
        """
        ترجمه یک متن یا لیستی از متون به زبان‌های درخواستی.
        """
        if isinstance(text_or_list, list):
            input_text = "\n".join([f"- {item}" for item in text_or_list])
        else:
            input_text = text_or_list.strip()

        if not input_text:
            return {lang: "متن ورودی خالی است." for lang in target_languages}

        # پرومپت ساده‌تر و مستقیم‌تر برای مدل‌های لوکال
        prompt = f"""
You are a professional multi-lingual translator specializing in {domain_context}.
Translate the following input text into Persian (FA).

Term Mapping Rules:
- Translate 'User' as 'ورزشکار'
- Use accurate sports biomechanics terminology.

{TranslationConfig.BIOMECHANICS_GLOSSARY}

Input Text:
{input_text}

Output ONLY the Persian translation below under the heading ---PERSIAN---.
---PERSIAN---
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )

            raw_output = response.choices[0].message.content or ""
            return self._parse_output(raw_output, target_languages)

        except Exception as e:
            logger.error(f"خطا در ارتباط با موتور ترجمه: {e}")
            return {lang: f"Error in translation: {str(e)}" for lang in target_languages}

    def _parse_output(self, raw_text: Optional[str], target_languages: List[str]) -> Dict[str, str]:
        results = {}
        text = (raw_text or "").strip()

        if "---PERSIAN---" in text:
            fa_part = text.split("---PERSIAN---")[-1]
            if "---GERMAN---" in fa_part:
                fa_part = fa_part.split("---GERMAN---")[0]
            results["fa"] = fa_part.strip()
        else:
            # Fallback: اگر دلیمیتر تولید نشد، کل متن خروجی را استفاده کن
            results["fa"] = text.strip()

        return results