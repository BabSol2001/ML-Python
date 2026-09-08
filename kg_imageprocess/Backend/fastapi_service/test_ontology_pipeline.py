# test_ontology_pipeline.py
import asyncio
import re
import time
from datetime import datetime, timezone

from db.graphiti_client import graphiti_client
from ollama_connection.ollama_client import ollama_service
from translation_standalone.translator import StandaloneTranslator


def is_clean_human_text(text: str) -> bool:
    if not text or len(str(text).strip()) < 5:
        return False
    uuid_pattern = r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}"
    if re.search(uuid_pattern, str(text), re.IGNORECASE):
        return False
    return True


async def test_exercise_episode(user_id: str, session_id: str, exercise_name: str, sample_text: str):
    print(f"\n==================================================")
    print(f"🏃 تست آنتولوژی حرکت: {exercise_name}")
    print(f"==================================================")

    start_time = time.time()
    await graphiti_client.add_workout_episode(
        user_id=user_id,
        session_id=session_id,
        episode_body=sample_text,
        source_description=f"MediaPipe Analysis - {exercise_name}",
        timestamp=datetime.now(timezone.utc),
    )
    print(f"✅ اپیزود با موفقیت ثبت شد (زمان: {time.time() - start_time:.2f} ثانیه)")

    # جستجو در حافظه با استفاده از متد اصلاح‌شده graphiti_client
    search_query = f"What biomechanical issues occurred during {exercise_name}?"
    print(f"🔍 در حال جستجوی حافظه برای: '{search_query}'...")
    
    memory_facts = []
    try:
        search_res = await graphiti_client.search_user_memory(
            user_id=user_id,
            query=search_query,
            num_results=3
        )
        for item in search_res:
            fact_val = item.get("fact", "")
            if fact_val and is_clean_human_text(fact_val):
                memory_facts.append(str(fact_val))
    except Exception as e:
        print(f"⚠️ خطای جستجو: {e}")

    # Fallback به متن ورودی اولیه در صورت نبود نتایج جستجو
    if not memory_facts:
        print("💡 نتایج مستقیم جستجو یافت نشد؛ استفاده از متن ورودی به عنوان Fallback...")
        memory_facts = [sample_text]

    print(f"📌 متن ورودی به مترجم: {memory_facts}")

    # فراخوانی درست متد async اصلی در StandaloneTranslator
    translator = StandaloneTranslator()
    translations = await translator.translate(
        text_or_list=memory_facts,
        target_languages=["fa"],
        domain_context="Biomechanics & Sports Science"
    )

    print("\n🇮🇷 **گزارش تحلیل بیومکانیک (فارسی):**")
    fa_result = translations.get("fa", "ترجمه‌ای یافت نشد.")
    print(f"   {fa_result}")


async def main():
    if not ollama_service.check_connection():
        print("❌ سرویس Ollama در دسترس نیست.")
        return

    ollama_service.setup_environment()
    await graphiti_client.initialize()

    test_cases = [
        {
            "user_id": "athlete_female_01",
            "session_id": "squat_sess_01",
            "exercise": "Squat",
            "text": "User athlete_female_01 performed Squat. Knee depth: 82 degrees. Observation: Knee Valgus occurred during concentric phase on Left Knee."
        },
        {
            "user_id": "athlete_male_02",
            "session_id": "rdl_sess_01",
            "exercise": "Romanian Deadlift",
            "text": "User athlete_male_02 performed Romanian Deadlift. Observation: Spine Flexion (Rounded Back) detected at Lumbar Spine during bottom phase."
        },
        {
            "user_id": "athlete_male_01",
            "session_id": "bench_sess_01",
            "exercise": "Bench Press",
            "text": "User athlete_male_01 performed Bench Press. Observation: Elbow Flaring detected on Right Elbow during concentric phase."
        }
    ]

    for tc in test_cases:
        await test_exercise_episode(tc["user_id"], tc["session_id"], tc["exercise"], tc["text"])

    await graphiti_client.close()
    print("\n🛑 تمام تست‌ها با موفقیت اجرا شدند.")


if __name__ == "__main__":
    asyncio.run(main())