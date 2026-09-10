import asyncio
from ollama_connection.ollama_client import ollama_service
from db.graphiti_client import graphiti_client
from services.rag_service import rag_service

async def main():
    print("--- ۱. بررسی اتصال Ollama ---")
    ollama_service.setup_environment()
    if not ollama_service.check_connection():
        print("❌ سرویس Ollama در دسترس نیست! مطمئن شوید Ollama در حال اجرا است.")
        return
    print("✅ اتصال Ollama برقرار است.")

    print("\n--- ۲. مقداردهی موتور گراف ---")
    await graphiti_client.initialize()

    user_id = "user_step3_test"
    session_id = "session_rag_1"

    print("\n--- ۳. ثبت یک فکت تست شامل اصطلاح بیومکانیکی (Knee Valgus) ---")
    await graphiti_client.log_biomechanical_fact(
        user_id=user_id,
        session_id=session_id,
        error_code="KNEE_VALGUS",
        details="در ست سوم حرکت اسکوات، Knee Valgus مشهود بود."
    )

    print("\n--- ۴. اجرای کامل RAG و بازنویسی کوئری ---")
    user_query = "امروز فرم اسکات من چطور بود؟"
    
    reply, facts = await rag_service.generate_coaching_feedback(
        user_id=user_id, 
        user_message=user_query
    )

    print("\n================== پاسخ مربی (Ollama + Translator) ==================")
    print(reply)
    print("=====================================================================")
    
    print("\nفکت‌های بازیابی شده از حافظه:")
    for f in facts:
        print(f"• {f.fact}")

    await graphiti_client.close()

if __name__ == "__main__":
    asyncio.run(main())