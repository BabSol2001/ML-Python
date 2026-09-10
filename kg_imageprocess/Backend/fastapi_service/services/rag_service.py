import logging
from typing import List, Tuple, Optional

from db.graphiti_client import graphiti_client
from ollama_connection.ollama_client import ollama_service
from schemas.feedback_schema import MemoryFact
from translation_standalone.config_translator import TranslationConfig

logger = logging.getLogger("RAGService")


class RAGService:
    """
    سرویس تحلیل عمیق پس از تمرین بر اساس MS-RAG، بازنویسی کوئری، 
    واژه‌نامه بیومکانیک و حافظه زمان‌مند Graphiti
    """

    async def _rewrite_query(self, user_message: str, chat_history: Optional[List[dict]] = None) -> str:
        """
        تبدیل کوئری کاربر به یک عبارت مستقل و توصیفی جهت جستجوی دقیق در گراف
        """
        if len(user_message.split()) > 6 and not chat_history:
            return user_message

        system_prompt = (
            "تو یک دستیار بازنویسی متن هستی. وظیفه تو این است که سوال یا پیام کاربر را "
            "با توجه به سابقه گفتگوی قبلی به یک سوال مستقل، واضح و توصیفی تبدیل کنی تا برای جستجو در دیتابیس مناسب باشد. "
            "فقط و فقط سوال بازنویسی‌شده را به زبان فارسی بنویس و هیچ توضیح اضافه‌ای نده."
        )

        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            messages.extend(chat_history[-4:])
        messages.append({"role": "user", "content": f"پیام کاربر: {user_message}"})

        rewritten = await ollama_service.chat_completion(messages=messages, temperature=0.2, timeout=30.0)
        if rewritten:
            logger.info(f"🔄 کوئری بازنویسی شد: '{user_message}' -> '{rewritten}'")
            return rewritten

        return user_message

    def _build_system_prompt(self, retrieved_facts: List[MemoryFact], user_message: str) -> str:
        """
        ساخت پرامپت تخصصی مربی بیومکانیک با تزریق حافظه گرافی و قانون واژه‌نامه ترجمه
        """
        facts_text = "\n".join([f"- {fact.fact} (زمان: {fact.valid_at or 'نامشخص'})" for fact in retrieved_facts])

        prompt = f"""شما یک مربی حرفه‌ای فیزیوتراپی و بیومکانیک ورزشی هستید.
وظیفه شما تحلیل وضعیت کاربر و پاسخ‌دهی کاملاً روان، مشوقانه و علم‌محور به زبان فارسی است.

قوانین واژه‌شناسی تخصصی (حتماً از این برگردان‌ها استفاده کنید):
{TranslationConfig.BIOMECHANICS_GLOSSARY}

تاریخچه بیومکانیکی و سوابق استخراج‌شده کاربر از گراف دانش (Graphiti):
{facts_text if facts_text else "هنوز سابقه قبلی برای این کاربر ثبت نشده است."}

سوال/پیام کاربر:
"{user_message}"

دستورالعمل پاسخ‌دهی:
۱. پاسخ را کوتاه، روان، صمیمی و کلامی بدهید (مناسب برای خوانده شدن توسط سیستم تبدیل متن به صوت TTS).
۲. اگر کاربر خطای تکراری در جلسات قبل داشته، حتماً به پیشرفت یا تکرار آن اشاره کنید.
۳. راهکار بیومکانیکی اصلاحی ساده ارائه دهید.
۴. از اصطلاحات خیلی پیچیده پزشکی بدون توضیح ساده استفاده نکنید."""
        return prompt.strip()

    async def generate_coaching_feedback(
        self, 
        user_id: str, 
        user_message: str,
        chat_history: Optional[List[dict]] = None
    ) -> Tuple[str, List[MemoryFact]]:
        """
        ۱. بازنویسی کوئری کاربر
        ۲. استخراج فکت‌های مرتبط از Graphiti
        ۳. ارسال به Ollama و دریافت پاسخ مربی
        """
        # ۱. بازنویسی کوئری
        search_query = await self._rewrite_query(user_message, chat_history)

        # ۲. استخراج ۵ حقیقت زمان‌مند از گراف
        raw_facts = await graphiti_client.search_user_memory(
            user_id=user_id,
            query=search_query,
            num_results=5
        )

        memory_facts = [
            MemoryFact(fact=f.get("fact", ""), valid_at=f.get("valid_at"))
            for f in raw_facts
            if f.get("fact")
        ]

        # ۳. ساخت پرامپت با تزریق قوانین ترجمه
        system_prompt = self._build_system_prompt(memory_facts, user_message)

        # ۴. فراخوانی Ollama برای تولید پاسخ مربی
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        ai_reply = await ollama_service.chat_completion(messages=messages, temperature=0.7)

        if not ai_reply:
            ai_reply = (
                "سلام! تحلیل بیومکانیکی شما انجام شد، اما در حال حاضر پاسخ‌دهی هوشمند موقتاً در دسترس نیست. "
                "لطفاً دوباره تلاش کنید."
            )

        return ai_reply, memory_facts


rag_service = RAGService()