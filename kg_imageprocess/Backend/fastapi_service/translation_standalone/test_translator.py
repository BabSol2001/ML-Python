import asyncio
from .translator import StandaloneTranslator

async def main():
    print("🚀 در حال فراخوانی ابزار ترجمه مستقل...")
    
    translator = StandaloneTranslator()

    # تست ۱: ترجمه یک لیست از ورودی‌های بیومکانیک (مشابه خروجی دیتابیس)
    sample_facts = [
        "User user_athlete_01 performed Squat.",
        "Knee depth: 82 degrees.",
        "Knee Valgus occurred during concentric phase."
    ]

    print("\n--------------------------------------------------")
    print("📌 نمونه ۱: ترجمه لیست داده‌های ورزشی/بیومکانیک")
    print("--------------------------------------------------")
    
    results = await translator.translate(
        text_or_list=sample_facts,
        target_languages=["fa", "de"],
        domain_context="Sports Biomechanics"
    )

    print("🇮🇷 **ترجمه فارسی:**")
    print(results.get("fa"))
    print("\n🇩🇪 **ترجمه آلمانی:**")
    print(results.get("de"))

    # تست ۲: ترجمه یک متن عمومی یا پزشکی (نشان‌دهنده عمومی بودن ابزار)
    sample_general_text = "The athlete shows slight muscle fatigue in the hamstring group after 5 sets."

    print("\n--------------------------------------------------")
    print("📌 نمونه ۲: ترجمه یک متن متفاوته مجزا")
    print("--------------------------------------------------")
    
    results_general = await translator.translate(
        text_or_list=sample_general_text,
        target_languages=["fa", "de"],
        domain_context="General Sports Medicine"
    )

    print("🇮🇷 **ترجمه فارسی:**")
    print(results_general.get("fa"))
    print("\n🇩🇪 **ترجمه آلمانی:**")
    print(results_general.get("de"))

if __name__ == "__main__":
    asyncio.run(main())