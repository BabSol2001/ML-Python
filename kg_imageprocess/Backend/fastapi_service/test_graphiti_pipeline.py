import asyncio
import re
import time
from datetime import datetime, timezone

from config import settings
from db.graphiti_client import graphiti_client
from ollama_connection.ollama_client import ollama_service
from translation_standalone.translator import StandaloneTranslator


def is_clean_human_text(text: str) -> bool:
    """
    بررسی معتبر بودن متن جهت جلوگیری از ارسال UUIDها و کدهای داخلی به مترجم
    """
    if not text or len(text.strip()) < 10:
        return False

    # الگوی تشخیص UUID
    uuid_pattern = r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}"
    if re.search(uuid_pattern, text, re.IGNORECASE):
        return False

    # اگر فقط لیستی از کلمات کلیدی کوتاه یا شناسه بود
    words = text.split()
    if len(words) < 3:
        return False

    return True


async def test_add_pose_episode():
    print("🚀 در حال بررسی و آماده‌سازی اتصال Ollama...")

    # ۱. بررسی و تنظیم متغیرهای محیطی جهت هدایت Graphiti به Ollama
    if not ollama_service.check_connection():
        print("❌ سرویس Ollama روی سیستم فعال نیست! مطمئن شوید Ollama در حال اجراست.")
        return

    ollama_service.setup_environment()
    print("✅ متغیرهای محیطی Ollama با موفقیت ست شدند.")

    print("\n🚀 در حال مقداردهی اولیه پایپ‌لاین Graphiti...")
    # ۲. فراخوانی initialize پس از تنظیم متغیرهای محیطی
    await graphiti_client.initialize()
    print("✅ اتصالات Neo4j و ایندکس‌های Graphiti با موفقیت آماده شدند.")

    # ۳. داده‌های نمونه حرکتی (Pose Estimator Data)
    user_id = "user_athlete_01"
    session_id = "squat_session_20260829"

    sample_text_data = (
        "User user_athlete_01 performed Squat. "
        "Knee depth: 82 degrees. "
        "Biomechanical observation: Knee Valgus occurred during concentric phase."
    )

    print(f"\n📝 در حال ثبت اپیزود تمرین برای کاربر `{user_id}`...")
    print(f"متن ورودی:\n\"{sample_text_data}\"\n")

    start_time = time.time()
    try:
        # ۴. ثبت اپیزود در گراف دانش
        await graphiti_client.add_workout_episode(
            user_id=user_id,
            session_id=session_id,
            episode_body=sample_text_data,
            source_description="MediaPipe Pose Analysis",
            timestamp=datetime.now(timezone.utc),
        )

        elapsed = time.time() - start_time
        print(f"✅ اپیزود با موفقیت در گراف دانش ذخیره شد! (زمان پردازش: {elapsed:.2f} ثانیه)")

        # ۵. جستجو و استخراج حقایق تمیز و انسانی از Neo4j
        search_query = "What biomechanical issue occurred with the athlete's knee during squat?"
        print(f"\n🔍 در حال جستجو در گراف حافظه برای: '{search_query}'...")

        memory_facts = []

        # الف) سعی در جستجوی مستقیم از طریق خود Graphiti
        if hasattr(graphiti_client, "_graphiti") and graphiti_client._graphiti:
            try:
                search_res = await graphiti_client._graphiti.search(query=search_query, group_ids=[user_id])
                edges = getattr(search_res, "edges", None) or getattr(search_res, "results", None) or (search_res if isinstance(search_res, list) else [])
                for item in edges[:5]:
                    fact_val = getattr(item, "fact", None) or getattr(item, "content", None) or getattr(item, "name", None) or str(item)
                    if fact_val and is_clean_human_text(str(fact_val)) and str(fact_val) not in memory_facts:
                        memory_facts.append(str(fact_val))
            except Exception as e:
                print(f"⚠️ خطای جستجوی گراف Graphiti: {e}")

        # ب) استخراج هوشمند خلاصه‌ها و متون تحلیلی از Neo4j (در صورت عدم دریافت خروجی تمیز از API)
        if not memory_facts and hasattr(graphiti_client, "_driver") and graphiti_client._driver:
            target_db = getattr(settings, "NEO4J_DATABASE", "kgimageprocessdb")

            cypher_query = """
            MATCH (n)
            UNWIND ['summary', 'content', 'fact'] AS prop
            WITH n, prop, n[prop] AS val
            WHERE val IS NOT NULL AND size(toString(val)) > 15
            RETURN DISTINCT toString(val) AS clean_val
            LIMIT 10
            """

            try:
                records, _, _ = await graphiti_client._driver.execute_query(
                    cypher_query, 
                    database_=target_db
                )
                for rec in records:
                    val = rec.get("clean_val")
                    if val and is_clean_human_text(val) and val not in memory_facts:
                        memory_facts.append(val)
            except Exception as e:
                print(f"⚠️ خطای اجرای Cypher روی دیتابیس `{target_db}`: {e}")

        # ۶. نمایش حقایق انگلیسی پالایش‌شده
        print("\n📊 حقایق (Facts) استخراج‌شده از دیتابیس (فیلترشده و تمیز):")
        print("=" * 60)
        
        if not memory_facts:
            print("⚠️ هیچ متن تحلیلی خالصی یافت نشد. استفاده از متن ورودی اولیه برای ترجمه.")
            texts_to_translate = [sample_text_data]
        else:
            # انتخاب حداکثر ۳ حقیقت برتر جهت افزایش سرعت مدل ترجمه
            texts_to_translate = memory_facts[:3]
            for idx, fact_text in enumerate(texts_to_translate, 1):
                print(f"[{idx}] 📌 {fact_text}")
                print("-" * 40)
        print("=" * 60)

        # ۷. ترجمه خودکار اتوماتیک (فارسی و آلمانی)
        print("\n🌐 در حال ترجمه سریع خروجی‌ها با ابزار مستقل (StandaloneTranslator)...")

        translator = StandaloneTranslator()
        translations = await translator.translate(
            text_or_list=texts_to_translate,
            target_languages=["fa", "de"],
            domain_context="Sports Biomechanics"
        )

        print("\n" + "=" * 60)
        print("🇮🇷 **گزارش تحلیل بیومکانیک (فارسی):**")
        print("=" * 60)
        print(translations.get("fa", "ترجمه‌ای دریافت نشد."))

        print("\n" + "=" * 60)
        print("🇩🇪 **Biomechanischer Analysebericht (Deutsch):**")
        print("=" * 60)
        print(translations.get("de", "Keine Übersetzung empfangen."))
        print("=" * 60)

    except Exception as e:
        print(f"❌ خطایی در فرایند تست رخ داد: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # ۸. بستن اتصالات
        await graphiti_client.close()
        print("\n🛑 اتصالات Neo4j و Graphiti بسته شدند.")


if __name__ == "__main__":
    asyncio.run(test_add_pose_episode())