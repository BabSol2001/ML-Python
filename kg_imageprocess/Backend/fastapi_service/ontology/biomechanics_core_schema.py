# biomechanics_core_schema.py
"""
ماژول هسته بیومکانیک: مدیریت هندسه حرکت، مفاصل و خطاهای زوایه‌ای پایه.
"""

CORE_NODE_LABELS = {
    "ATHLETE": "Athlete",
    "SESSION": "WorkoutSession",
    "EXERCISE": "Exercise",
    "OBSERVATION": "BiomechanicalObservation",
    "JOINT": "BodyJoint"
}

CORE_RELATIONSHIPS = [
    "PERFORMED",           # (Athlete) -[PERFORMED]-> (WorkoutSession)
    "INCLUDES_EXERCISE",  # (WorkoutSession) -[INCLUDES_EXERCISE]-> (Exercise)
    "EXHIBITED",          # (WorkoutSession) -[EXHIBITED]-> (BiomechanicalObservation)
    "AFFECTS_JOINT"       # (BiomechanicalObservation) -[AFFECTS_JOINT]-> (BodyJoint)
]

# داده‌های مرجع ۳ حرکت پایه
MASTER_EXERCISES = {
    "Squat": {
        "category": "Lower Body",
        "primary_joints": ["Left Knee", "Right Knee", "Hip", "Ankle"],
        "common_faults": ["Knee Valgus", "Insufficient Depth", "Butt Wink"]
    },
    "Romanian Deadlift": {
        "category": "Posterior Chain",
        "primary_joints": ["Lumbar Spine", "Hip", "Knee"],
        "common_faults": ["Spine Flexion (Rounded Back)", "Excessive Knee Flexion"]
    },
    "Bench Press": {
        "category": "Upper Body Push",
        "primary_joints": ["Shoulder", "Elbow", "Wrist"],
        "common_faults": ["Elbow Flaring", "Asymmetric Bar Path", "Wrist Extension"]
    }
}

CORE_SYSTEM_PROMPT = """
You are a Sports Biomechanics Knowledge Graph Extractor.
Extract entities and relationships from the provided text adhering STRICTLY to this schema:

Node Types:
- Athlete (properties: user_id)
- WorkoutSession (properties: session_id, timestamp)
- Exercise (properties: name)
- BiomechanicalObservation (properties: fault_type, phase, degree_value, severity)
- BodyJoint (properties: joint_name, side)

Relationships:
- PERFORMED, INCLUDES_EXERCISE, EXHIBITED, AFFECTS_JOINT

Rules:
1. Do NOT invent new relationship names outside the allowed list.
2. Standardize joint names (e.g., 'Left Knee', 'Right Knee', 'Lumbar Spine').
"""