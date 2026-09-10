import os
from dotenv import load_dotenv

load_dotenv()


class TranslationConfig:
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    API_KEY: str = os.getenv("OPENAI_API_KEY", "ollama")

    BIOMECHANICS_GLOSSARY = """
    STRICT TRANSLATION RULES:
    1. Do NOT transliterate proper exercise names or anatomical parts into nonsense Persian words.
    2. Always map terms EXACTLY according to the table below.

    --- EXERCISE NAMES ---
    - "Squat" -> FA: "اسکات" | DE: "Kniebeuge"
    - "Romanian Deadlift" / "RDL" -> FA: "ددلیفت رومانیایی" | DE: "Rumänisches Kreuzheben"
    - "Bench Press" -> FA: "پرس سینه" | DE: "Bankdrücken"
    - "Overhead Press" -> FA: "پرس بالای سر (اورهد)" | DE: "Schulterdrücken"
    - "Lunge" -> FA: "لانج (لانگز)" | DE: "Ausfallschritt"

    --- BIOMECHANICS & PHASES ---
    - "Concentric phase" -> FA: "فاز درون‌گرا (صعودی/غلبه)" | DE: "Konzentrische Phase"
    - "Eccentric phase" -> FA: "فاز برون‌گرا (فرود/کنترل)" | DE: "Exzentrische Phase"
    - "Bottom phase" -> FA: "پایین‌ترین نقطه حرکت" | DE: "Untere Umkehrphase"
    - "Lockout phase" -> FA: "فاز قفل کردن مفصل (انتها)" | DE: "Endposition / Lockout"
    - "Knee depth" -> FA: "عمق خم شدن زانو" | DE: "Kniewinkel / Beugungstiefe"
    - "Range of Motion (ROM)" -> FA: "دامنه حرکتی" | DE: "Bewegungsausmaß (ROM)"

    --- DEVIATIONS & BIOMECHANICAL ISSUES ---
    - "NOT_DEEP_ENOUGH" / "Knee depth insufficient" -> FA: "عمق ناکافی اسکات (عدم پایین رفتن کامل)" | DE: "Unzureichende Tiefe"
    - "Knee Valgus" -> FA: "انحراف زانو به داخل (والگوس زانو)" | DE: "Knee-Valgus (X-Bein-Tendenz)"
    - "Knee Varus" -> FA: "انحراف زانو به بیرون (واروس زانو)" | DE: "Knee-Varus (O-Bein-Tendenz)"
    - "Spine Flexion" / "Rounded Back" -> FA: "خم شدن/گرد شدن ستون فقرات" | DE: "Wirbelsäulenflexion (Runder Rücken)"
    - "Spine Extension" -> FA: "باز شدن/گودی بیش از حد ستون فقرات" | DE: "Wirbelsäulenextension"
    - "Elbow Flaring" -> FA: "باز شدن بیش از حد آرنج‌ها به طرفین (فلار شدن آرنج)" | DE: "Abspreizen der Ellenbogen (Elbow Flaring)"
    - "Asymmetric Loading" -> FA: "عدم تقارن در اعمال نیرو" | DE: "Asymmetrische Belastung"

    --- ANATOMY & JOINTS ---
    - "Lumbar Spine" -> FA: "ستون فقرات کمری (بخش لومبار)" | DE: "Lendenwirbelsäule (LWS)"
    - "Thoracic Spine" -> FA: "ستون فقرات پشتی/سینه" | DE: "Brustwirbelsäule (BWS)"
    - "Left Knee" -> FA: "زانوی چپ" | DE: "Linkes Knie"
    - "Right Knee" -> FA: "زانوی راست" | DE: "Rechtes Knie"
    - "Left Elbow" -> FA: "آرنج چپ" | DE: "Linker Ellenbogen"
    - "Right Elbow" -> FA: "آرنج راست" | DE: "Rechter Ellenbogen"
    - "User" / "Athlete" -> FA: "ورزشکار" | DE: "Athlet"
    """