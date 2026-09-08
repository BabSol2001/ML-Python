# fastapi_service/services/django_client.py
import httpx

DJANGO_BASE_URL = "http://localhost:8000/api/users"  # آدرس سرویس جانگو

async def fetch_athlete_context(user_id: str) -> dict:
    """دریافت خصوصیات فیزیکی و سوابق کاربر از دیتابیس PostgreSQL جانگو"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{DJANGO_BASE_URL}/internal/athlete-context/{user_id}/", timeout=5.0)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error connecting to Django service: {e}")
            
    # در صورت عدم دسترس بودن جانگو، یک فال‌بک پیش‌فرض برمی‌گرداند
    return {
        "id": user_id,
        "profile": {"age": None, "injury_history": [], "fitness_level": "beginner"}
    }