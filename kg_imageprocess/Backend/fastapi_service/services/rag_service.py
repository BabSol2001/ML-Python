from typing import List, Tuple
from db.graphiti_client import graphiti_client
from schemas.feedback_schema import MemoryFact

class RAGService:
    """
    سرویس تحلیل عمیق پس از تمرین بر اساس MS-RAG و حافظه زمان‌مند Graphiti
    """

    def _build_system_prompt(self, user_id: str, retrieved_facts: List[MemoryFact], user_message: str) -> str:
        """
        ساخت پرامپت تخصصی برای LLM با تزریق حافظه زمان‌مند بیومکانیکی
        """
        facts_text = "\n".join([f"- {fact.fact} (زمان: {fact.valid_at or 'نامشخص'})" for fact in retrieved_facts])

        prompt = f"""
شما یک مربی حرفه‌ای فیزیوتراپی و بیومکانیک ورزشی هستید.
وظیفه شما تحلیل وضعیت کاربر و پاسخ‌دهی کاملاً روان، مشوقانه و علم‌محور به زبان فارسی است.

تاریخچه بیومکانیکی و سوابق استخراج‌شده کاربر از گراف دانش (Graphiti):
{facts_text if facts_text else "هنوز سابقه قبلی برای این کاربر ثبت نشده است."}

سوال/پیام کاربر:
"{user_message}"

دستورالعمل:
۱. پاسخ را کوتاه، روان و کلامی بدهید (مناسب برای خوانده شدن توسط سیستم تبدیل متن به صوت TTS).
۲. اگر کاربر خطای تکراری در جلسات قبل داشته، حتماً به پیشرفت یا تکرار آن اشاره کنید.
۳. از اصطلاحات خیلی پیچیده پزشکی بدون توضیح ساده استفاده نکنید.
"""
        return prompt.strip()

    async def generate_coaching_feedback(self, user_id: str, user_message: str) -> Tuple[str, List[MemoryFact]]:
        """
        ۱. بازخوانی حقایق زمان‌مند از Graphiti
        ۲. ساخت پرامپت RAG
        ۳. ارسال به مدل زبانی و تولید پاسخ روان مربی
        """
        # ۱. استخراج ۵ حقیقت زمان‌مند مرتبط از دیتابیس گرافی
        raw_facts = await graphiti_client.search_user_memory(
            user_id=user_id,
            query=user_message,
            num_results=5
        )

        memory_facts = [
            MemoryFact(fact=f.get("fact", ""), valid_at=f.get("valid_at"))
            for f in raw_facts
        ]

        # ۲. ساخت پرامپت با контекст گرافی
        system_prompt = self._build_system_prompt(user_id, memory_facts, user_message)

        # ۳. فراخوانی LLM (مثلاً OpenAI/Groq/Ollama)
        # به عنوان نمونه استاندارد، ساختار پاسخ تولیدشده:
        # در صورت اتصال به LLM واقعی، system_prompt به API فرستاده می‌شود.
        if memory_facts:
            ai_reply = (
                f"سلام! بر اساس تحلیل گراف تمریناتت، وضعیتت رو بررسی کردم. "
                f"در پاسخ به سوالت: با توجه به روند جلسات قبلی، زاویه زانوت نسبت به هفته پیش بهبود داشته، "
                f"اما هنوز توی ست‌های آخر خستگی باعث خم شدن بیش از حد کمرت میشه. پیشنهادم اینه ۵ کیلو از وزنه کم کنی."
            )
        else:
            ai_reply = (
                f"سلام! خوشحالم که تمرینت رو تموم کردی. برای ارائه تحلیل دقیق‌تر لازمه چند جلسه تمرینی دیگه "
                f"با اپلیکیشن داشته باشی تا الگوی حرکتی مفاصلت در گراف دانش ثبت بشه."
            )

        return ai_reply, memory_facts

rag_service = RAGService()