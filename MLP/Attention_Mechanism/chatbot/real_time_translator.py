"""
ماژول اختصاصی ترجمه هم‌زمان (Real-Time Translator)
این فایل فقط وظیفه ارتباط با موتور ترجمه را بر عهده دارد.
"""

from deep_translator import GoogleTranslator

class RealTimeTranslator:
    def __init__(self):
        # تعریف دو مترجم مستقل برای دو مسیر ترجمه
        self.fa_to_en_translator = GoogleTranslator(source='fa', target='en')
        self.en_to_fa_translator = GoogleTranslator(source='en', target='fa')

    def fa_to_en(self, text: str) -> str:
        """ترجمه ورودی فارسی کاربر به انگلیسی"""
        if not text or not text.strip():
            return ""
        try:
            return self.fa_to_en_translator.translate(text)
        except Exception as e:
            print(f"هشدار در ترجمه فارسی به انگلیسی: {e}")
            return text  # در صورت بروز خطا، اصل متن بازگردانده می‌شود

    def en_to_fa(self, text: str) -> str:
        """ترجمه خروجی انگلیسی مدل به فارسی"""
        if not text or not text.strip():
            return ""
        try:
            return self.en_to_fa_translator.translate(text)
        except Exception as e:
            print(f"هشدار در ترجمه انگلیسی به فارسی: {e}")
            return text  # در صورت بروز خطا، اصل متن بازگردانده می‌شود