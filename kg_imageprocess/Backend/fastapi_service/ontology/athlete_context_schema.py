# athlete_context_schema.py
"""
ماژول داده‌های زمینه‌ای ورزشکار (Athlete Context):
فاز ۱: سن، جنسیت، سطح حرفه‌ای، سوابق پزشکی و آسیب‌دیدگی.
فاز ۲: داده‌های ساعت‌های هوشمند (ECG, Heart Rate, Sleep, SpO2) و تغذیه.
"""

CONTEXT_NODE_LABELS = {
    "MEDICAL_CONDITION": "MedicalCondition",
    "INJURY_HISTORY": "InjuryHistory",
    "DAILY_PHYSIO": "DailyPhysioMetrics",
    "SLEEP_LOG": "SleepLog",
    "NUTRITION_LOG": "NutritionLog"
}

CONTEXT_RELATIONSHIPS = [
    "HAS_MEDICAL_HISTORY",
    "HAS_PRIOR_INJURY",
    "RECORDED_DAILY_METRIC",
    "LOGGED_SLEEP",
    "CONSUMED_MEAL"
]

ATHLETE_PROFILE_TEMPLATE = {
    "gender": "male / female",
    "age": "int",
    "experience_level": "beginner / intermediate / advanced",
    "height_cm": "float",
    "weight_kg": "float"
}