import logging
from typing import Dict, List, Union, Optional
from openai import AsyncOpenAI
# تلاش برای امپورت پروژه (یک سطح بالاتر یا امپورت مستقیم)
from .config_translator import TranslationConfig  # امپورت مستقیم از Root

logger = logging.getLogger("StandaloneTranslator")

class StandaloneTranslator:
    """
    موتور ترجمه مستقل و هوشمند با استفاده از مدل‌های زبانی (LLM).
    قابل استفاده برای انواع متون با پشتیبانی از واژه‌نامه تخصصی.
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
        target_languages: List[str] = ["fa", "de"],
        domain_context: str = "Biomechanics & Sports Science"
    ) -> Dict[str, str]:
        """
        ترجمه یک متن یا لیستی از متون به زبان‌های درخواستی.
        """
        # آماده‌سازی متن ورودی
        if isinstance(text_or_list, list):
            input_text = "\n".join([f"- {item}" for item in text_or_list])
        else:
            input_text = text_or_list.strip()

        if not input_text:
            return {lang: "متن ورودی خالی است." for lang in target_languages}

        # ساخت پرومپت پویا بر اساس زبان‌های درخواستی
        lang_instructions = []
        output_format_blocks = []

        if "fa" in target_languages:
            lang_instructions.append("- Persian (FA): Accurate, fluent, and using standard technical terms.")
            output_format_blocks.append("---PERSIAN---\n[Persian translation here]")

        if "de" in target_languages:
            lang_instructions.append("- German (DE): Professional, grammatically sound, using precise technical terms.")
            output_format_blocks.append("---GERMAN---\n[German translation here]")

        prompt = f"""
You are a professional multi-lingual translator specializing in {domain_context}.
Translate the input text into the specified target languages below.

{TranslationConfig.BIOMECHANICS_GLOSSARY}

Target Languages:
{chr(10).join(lang_instructions)}

Input Text:
{input_text}

Strictly use the following output delimiters for parsing:
{chr(10).join(output_format_blocks)}
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1  # دمای پایین برای ترجمه دقیق
            )

            # اصلاح ۱: تبدیل None احتمالی به رشته خالی یا اطمینان از رشته بودن
            raw_output = response.choices[0].message.content or ""
            return self._parse_output(raw_output, target_languages)

        except Exception as e:
            logger.error(f"خطا در ارتباط با موتور ترجمه: {e}")
            return {lang: f"Error in translation: {str(e)}" for lang in target_languages}

    # اصلاح ۲: پشتیبانی از Optional[str] برای متغیر ورودی جهت ایمنی کامل
    def _parse_output(self, raw_text: Optional[str], target_languages: List[str]) -> Dict[str, str]:
        """تجزیه متن خروجی مدل براساس تفکیک‌کننده‌ها"""
        results = {}
        
        # اگر raw_text مقدار None باشد، آن را رشته خالی فرض می‌کنیم
        text = raw_text or ""

        if "fa" in target_languages:
            if "---PERSIAN---" in text:
                fa_part = text.split("---PERSIAN---")[-1]
                if "---GERMAN---" in fa_part:
                    fa_part = fa_part.split("---GERMAN---")[0]
                results["fa"] = fa_part.strip()
            else:
                results["fa"] = text.strip()

        if "de" in target_languages:
            if "---GERMAN---" in text:
                de_part = text.split("---GERMAN---")[-1]
                results["de"] = de_part.strip()
            else:
                results["de"] = text.strip()

        return results