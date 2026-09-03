import time
from ollama_connection.ollama_client import ollama_service
from openai import OpenAI


def test_ollama_direct_response():
    print("🚀 در حال بررسی و تست مستقیم ماژول Ollama...")

    # ۱. بررسی سلامت سرویس
    if not ollama_service.check_connection():
        print("❌ سرویس Ollama روی سیستم فعال نیست! لطفا دستور 'ollama run llama3' را اجرا کنید.")
        return

    print("✅ اتصال اولیه به Ollama برقرار است.")

    # ۲. آماده‌سازی متغیرهای محیطی
    ollama_service.setup_environment()

    # ۳. ارسال یک پرامپت تست با استفاده از OpenAI SDK (که به Ollama متصل شده)
    client = OpenAI(
        base_url=ollama_service.base_url,
        api_key="ollama",  # کلید فرضی
    )

    prompt = "پاسخ کوتاه بده: وظیفه اصلی دیتابیس گرافی در پردازش حرکات ورزشی چیست؟"
    print(f"\n💬 ارسال پرسش نمونه: '{prompt}'")

    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=ollama_service.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        elapsed_time = time.time() - start_time

        reply_content = response.choices[0].message.content
        print("\n🤖 پاسخ دریافت شده از مدل Ollama:")
        print("--------------------------------------------------")
        print(reply_content)
        print("--------------------------------------------------")
        print(f"⏱️ زمان پاسخ‌دهی: {elapsed_time:.2f} ثانیه")
        print("🎉 تست ماژول ollama_connection با موفقیت به پایان رسید!")

    except Exception as e:
        print(f"❌ خطایی در دریافت پاسخ از مدل رخ داد: {e}")


if __name__ == "__main__":
    test_ollama_direct_response()