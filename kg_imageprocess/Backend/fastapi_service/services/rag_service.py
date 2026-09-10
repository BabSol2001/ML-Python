import logging
from typing import List, Tuple, Optional, Any, Dict

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
        system_prompt = (
            "تو یک متخصص بازنویسی پرس‌وجو در سیستم‌های بازیابی اطلاعات (RAG) بیومکانیک ورزشی هستی. "
            "وظیفه تو این است که آخرین پیام کاربر را با توجه به تاریخچه گفتگو، به یک عبارت کلیدی و کوتاه برای جستجو در دیتابیس تبدیل کنی. "
            "قوانین سخت‌گیرانه:\n"
            "1. فقط و فقط کلمات کلیدی اصلی مانند نام حرکت، نوع خطا یا واژه بیومکانیکی مربوطه را خروجی بده (مثلاً: اسکوات خطا عمق زانو).\n"
            "2. هیچ مقدمه، مؤخره، توضیحات اضافی یا علامت کوتیشن اضافه نکن.\n"
            "3. زبان خروجی باید فارسی باشد."
        )

        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            messages.extend(chat_history[-4:])
        messages.append({"role": "user", "content": f"پیام کاربر: {user_message}"})

        rewritten = await ollama_service.chat_completion(messages=messages, temperature=0.2, timeout=30.0)
        if rewritten:
            clean_query = rewritten.strip().replace('"', '').replace("'", "")
            logger.info(f"🔄 کوئری بازنویسی شد: '{user_message}' -> '{clean_query}'")
            return clean_query

        return user_message

    def _build_system_prompt(self, retrieved_facts: List[MemoryFact], user_message: str) -> str:
        facts_text = "\n".join([f"- {fact.fact}" for fact in retrieved_facts])

        prompt = f"""شما یک مربی حرفه‌ای، صمیمی و متخصص بیومکانیک ورزشی هستید.
وظیفه شما تحلیل داده‌های زیر و پاسخ به کاربر به زبان فارسی روان، روان‌شناختی و کاملاً طبیعی است.

قوانین واژه‌شناسی تخصصی:
{TranslationConfig.BIOMECHANICS_GLOSSARY}

داده‌های بیومکانیکی استخراج‌شده از تمرین کاربر:
{facts_text if facts_text else "سابقه‌ای ثبت نشده است."}

سوال کاربر:
"{user_message}"

قوانین پاسخ‌دهی (بسیار مهم):
۱. به هیچ وجه از کلمات نامأنوس یا ترجمه‌های نامفهوم (مانند اقلیم، پای پا و...) استفاده نکنید.
۲. کد NOT_DEEP_ENOUGH را مستقیماً "عمق ناکافی اسکات" یا "پایین نرفتن به اندازه کافی" معنا کنید.
۳. توضیح دهید که زاویه ۱۶۸ درجه یعنی زانو خیلی باز مانده و برای اسکات استاندارد (موازی شدن ران با زمین) باید زاویه زانو به حدود ۹۰ درجه برسد.
۴. پاسخ باید کاملاً مختصر (حداکثر ۳ یا ۴ جمله کوتاه)، صمیمی و اصلاحی باشد.
"""
        return prompt.strip()
    
    async def generate_coaching_feedback(
        self, 
        user_id: str, 
        user_message: str,
        session_id: Optional[str] = None,
        chat_history: Optional[List[dict]] = None
    ) -> Tuple[str, List[MemoryFact]]:
        """
        ۱. بازنویسی کوئری کاربر
        ۲. استخراج فکت‌های مرتبط از Graphiti با شناسه کاربر و جلسه تمرینی
        ۳. ارسال به Ollama و دریافت پاسخ مربی
        """
        # ۱. بازنویسی کوئری
        search_query = await self._rewrite_query(user_message, chat_history)

        # ترکیب شناسه جلسه با کوئری جستجو در صورت وجود
        full_query = f"Session: {session_id} {search_query}" if session_id else search_query

        # ۲. استخراج ۵ حقیقت زمان‌مند از گراف
        raw_facts = await graphiti_client.search_user_memory(
            user_id=user_id,
            query=full_query,
            num_results=5
        )

        logger.info(f"📊 خروجی خام دریافت شده از Graphiti: {raw_facts}")

        # اگر خروجی خالی بود، اجرای Fallback بر اساس session_id یا حرکت اسکوات
        if not raw_facts:
            fallback_query = f"Session: {session_id}" if session_id else "squat"
            logger.info(f"🔍 فکتی با کوئری '{full_query}' یافت نشد؛ اجرای جستجوی Fallback با '{fallback_query}'...")
            raw_facts = await graphiti_client.search_user_memory(
                user_id=user_id,
                query=fallback_query,
                num_results=5
            )
            logger.info(f"📊 خروجی Fallback از Graphiti: {raw_facts}")

        # تبدیل به ساختار Pydantic با پشتیبانی از کلیدهای مختلف (fact, content, text, episode)
        memory_facts = []
        for f in raw_facts:
            if isinstance(f, dict):
                fact_text = f.get("fact") or f.get("content") or f.get("text") or f.get("episode")
                valid_at = f.get("valid_at") or f.get("created_at")
                if fact_text:
                    memory_facts.append(MemoryFact(fact=str(fact_text), valid_at=valid_at))
            elif isinstance(f, str):
                memory_facts.append(MemoryFact(fact=f, valid_at=None))

        logger.info(f"✅ فکت‌های نهایی پردازش شده برای پرامپت: {memory_facts}")

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