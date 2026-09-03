import sys
import os
import asyncio
from fastapi.testclient import TestClient

# اضافه کردن مسیر پروژه به سیستم ایمپورت
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import settings
from main import app


def test_config_loading():
    """تست ۱: صحت بارگذاری تنظیمات از فایل config.py"""
    print("\n--- [Test 1] تست بارگذاری پیکربندی (Config) ---")
    print(f"📌 نام پروژه: {settings.PROJECT_NAME}")
    print(f"📌 حالت برنامه‌نویسی (Debug): {settings.DEBUG}")
    print(f"📌 آدرس Neo4j: {settings.NEO4J_URI}")
    print(f"📌 حد آستانه زوایای اسکوات: {settings.SQUAT_KNEE_ANGLE_THRESHOLD} درجه")
    
    assert settings.PROJECT_NAME is not None, "نام پروژه خالی است!"
    assert settings.SQUAT_KNEE_ANGLE_THRESHOLD == 90.0, "آستانه زانو اشتباه خوانده شده است!"
    print("✅ تست پیکربندی با موفقیت انجام شد.")


def test_fastapi_health_endpoint():
    """تست ۲: تست اندپوینت Health و اتصال به دیتابیس‌ها با استفاده از TestClient"""
    print("\n--- [Test 2] تست اندپوینت /health سرویس FastAPI ---")
    
    # استفاده از TestClient با مدیریت چرخه حیات (Lifespan Context Manager)
    with TestClient(app) as client:
        response = client.get("/health")
        
        print(f"📌 کد وضعیت پاسخ (Status Code): {response.status_code}")
        print(f"📌 خروجی دریافت شده: {response.json()}")
        
        assert response.status_code == 200, "پاسخ سرور 200 نبود!"
        data = response.json()
        assert data["status"] == "online", "وضعیت سیستم آنلاین نیست!"
        assert data["service"] == settings.PROJECT_NAME, "نام سرویس در پاسخ ناهمخوان است!"
        
        if data["neo4j_connected"]:
            print("🟢 دیتابیس Neo4j متصل و فعال است.")
        else:
            print("⚠️ دیتابیس Neo4j متصل نیست (بررسی کنید که سرویس Neo4j اجرا شده باشد).")

        if data["graphiti_initialized"]:
            print("🟢 موتور Graphiti با موفقیت مقداردهی شده است.")
        else:
            print("⚠️ موتور Graphiti مقداردهی نشده است.")

    print("✅ تست API با موفقیت به پایان رسید.")


if __name__ == "__main__":
    print("🚀 در حال اجرای تست‌های ترکیبی main.py و config.py...\n")
    try:
        test_config_loading()
        test_fastapi_health_endpoint()
        print("\n🎉 تمامی تست‌ها با موفقیت سپری شدند!")
    except Exception as e:
        print(f"\n❌ تست با خطا مواجه شد: {e}")