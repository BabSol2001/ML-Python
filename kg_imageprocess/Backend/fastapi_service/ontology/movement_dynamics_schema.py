# movement_dynamics_schema.py
"""
ماژول تحلیل پویایی حرکت (Dynamics): ریتم زمان‌بندی (Tempo)، تقارن (Asymmetry)، 
مسیر بار/مرکز ثقل (Bar Path) و شاخص‌های خستگی عصبی-عضلانی.
"""

DYNAMICS_NODE_LABELS = {
    "MOVEMENT_TEMPO": "MovementTempo",
    "BODY_ASYMMETRY": "BodyAsymmetry",
    "FATIGUE_LOG": "FatigueMetrics",
    "BAR_PATH": "BarPathAnalysis"
}

DYNAMICS_RELATIONSHIPS = [
    "HAS_TEMPO",
    "EXHIBITED_ASYMMETRY",
    "TRACKED_FATIGUE",
    "DEVIATED_BAR_PATH"
]

DYNAMICS_METRICS = {
    "tempo": {
        "eccentric_duration_sec": "float",
        "concentric_duration_sec": "float",
        "pause_duration_sec": "float",
        "tempo_ratio": "string"
    },
    "asymmetry": {
        "weight_distribution_left_pct": "float",
        "weight_distribution_right_pct": "float",
        "joint_angle_delta_degrees": "float"
    },
    "fatigue": {
        "set_number": "int",
        "rep_number": "int",
        "velocity_loss_pct": "float",
        "neuromuscular_fatigue_flag": "bool"
    },
    "bar_path": {
        "center_of_gravity_deviation_cm": "float",
        "horizontal_bar_drift_cm": "float"
    }
}