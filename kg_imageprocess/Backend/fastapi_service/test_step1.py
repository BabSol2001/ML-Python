# test_step1.py
from schemas.pose_schema import PoseFramePayload, Keypoint
from core.rule_engine import check_squat_rules

# ساخت فریم نمونه با استفاده از نام دقیق کلاس Keypoint
dummy_keypoints = [
    Keypoint(id=23, name="left_hip", x=0.5, y=0.4, score=0.9),
    Keypoint(id=25, name="left_knee", x=0.5, y=0.6, score=0.9),
    Keypoint(id=27, name="left_ankle", x=0.4, y=0.8, score=0.9),
]

payload = PoseFramePayload(
    frame_id=1,
    timestamp=1690000000.0,
    exercise_name="squat",
    keypoints=dummy_keypoints
)

# ۱. تست با کاربر معمولی
context_healthy = {"profile": {"injury_history": []}}
res_healthy = check_squat_rules(payload, athlete_context=context_healthy)
print("--- کاربر سالم ---")
print(f"Is Valid: {res_healthy.is_valid} | Message: {res_healthy.feedback_message}")

# ۲. تست با کاربر دارای سابقه آسیب زانو
context_injured = {"profile": {"injury_history": ["knee_acl_reconstruction"]}}
res_injured = check_squat_rules(payload, athlete_context=context_injured)
print("\n--- کاربر با سابقه آسیب زانو ---")
print(f"Is Valid: {res_injured.is_valid} | Message: {res_injured.feedback_message}")