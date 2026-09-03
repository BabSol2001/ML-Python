import os
from dotenv import load_dotenv

load_dotenv()

class TranslationConfig:
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    API_KEY: str = os.getenv("OPENAI_API_KEY", "ollama")

    BIOMECHANICS_GLOSSARY = """
    Terminology Guidance:
    - "Knee Valgus" -> FA: "انحراف زانو به داخل (والگوس زانو)" | DE: "Knie-Valgus (X-Bein-Tendenz)"
    - "Concentric phase" -> FA: "فاز درون‌گرا (صعودی/غلبه)" | DE: "Konzentrische Phase"
    - "Eccentric phase" -> FA: "فاز برون‌گرا (فرود/کشش)" | DE: "Exzentrische Phase"
    - "Knee depth" -> FA: "عمق خم شدن زانو" | DE: "Kniewinkel / Beugungstiefe"
    - "Range of Motion (ROM)" -> FA: "دامنه حرکتی" | DE: "Bewegungsausmaß (ROM)"
    """