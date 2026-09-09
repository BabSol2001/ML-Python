# fastapi_service/services/django_client.py
import httpx
import logging

logger = logging.getLogger("DjangoClient")

# آدرس پایه سرویس جانگو (پوشش‌دهنده /api/users/ و /api/payments/)
DJANGO_BASE_URL = "http://localhost:8000/api"


async def fetch_athlete_context(user_id: str) -> dict:
    """دریافت خصوصیات فیزیکی، سوابق پزشکی و آنتروپومتری کاربر از دیتابیس PostgreSQL جانگو"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{DJANGO_BASE_URL}/users/internal/athlete-context/{user_id}/", 
                timeout=5.0
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"خطا در دریافت Athlete Context از جانگو: {e}")
            
    # فال‌بک امن در صورت عدم پاسخگویی یا قطعی سرویس جانگو
    return {
        "id": user_id,
        "profile": {
            "age": None,
            "injury_history": [],
            "postural_deviations": [],
            "fitness_level": "beginner"
        }
    }


async def check_user_subscription_status(user_id: str) -> dict:
    """
    بررسی وضعیت اشتراک فعال و باقی‌مانده سقف پردازش ویدیو کاربر از اپلیکیشن payments جانگو
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{DJANGO_BASE_URL}/payments/subscriptions/?user_id={user_id}",
                timeout=5.0
            )
            if response.status_code == 200:
                subscriptions = response.json()
                if subscriptions:
                    # استخراج اطلاعات اولین/فعال‌ترین اشتراک کاربر
                    active_sub = subscriptions[0]
                    plan_details = active_sub.get("plan_details", {}) or {}
                    max_analyses = plan_details.get("max_video_analyses", 0)
                    used_analyses = active_sub.get("used_analyses_count", 0)
                    status = active_sub.get("status", "expired")

                    is_allowed = (status == "active") and (used_analyses < max_analyses)
                    return {
                        "is_allowed": is_allowed,
                        "subscription_id": active_sub.get("id"),
                        "remaining_analyses": max(0, max_analyses - used_analyses),
                        "reason": "OK" if is_allowed else "سقف آنالیز ویدیو شما به پایان رسیده یا اشتراک فعال ندارید."
                    }
        except Exception as e:
            logger.error(f"خطا در استعلام اشتراک از جانگو: {e}")

    # در صورت عدم دسترسی به جانگو یا حالت توسعه، اجازه دسترسی فال‌بک داده می‌شود
    return {
        "is_allowed": True,
        "subscription_id": None,
        "remaining_analyses": 1,
        "reason": "Fallback Mode"
    }