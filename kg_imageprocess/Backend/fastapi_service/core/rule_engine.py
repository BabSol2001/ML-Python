from typing import Dict, Tuple
from schemas.pose_schema import PoseFramePayload, PoseFeedbackResponse
from core.pose_calculator import calculate_angle_2d
from config import settings

# شناسه مفاصل بر اساس استاندارد MediaPipe Pose
LEFT_HIP = 23
LEFT_KNEE = 25
LEFT_ANKLE = 27

def check_squat_rules(payload: PoseFramePayload) -> PoseFeedbackResponse:
    """
    بررسی قوانین حرکت اسکوات بر اساس زوایای مفاصل
    """
    # تبدیل لیست مفصل‌ها به دیکشنری برای دسترسی سریع O(1)
    keypoints_dict: Dict[int, Tuple[float, float]] = {
        kp.id: (kp.x, kp.y) for kp in payload.keypoints if kp.score > 0.5
    }

    # مطمئن شویم سه مفصل اصلی مربوط به پای چپ موجود هستند
    if not all(k in keypoints_dict for k in [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE]):
        return PoseFeedbackResponse(
            frame_id=payload.frame_id,
            is_valid=False,
            exercise_name=payload.exercise_name,
            feedback_message="مفاصل به طور کامل در تصویر دیده نمی‌شوند",
            error_code="JOINTS_NOT_VISIBLE",
            highlight_joints=[]
        )

    # استخراج مختصات (لگن - زانو - مچ پا)
    hip = keypoints_dict[LEFT_HIP]
    knee = keypoints_dict[LEFT_KNEE]
    ankle = keypoints_dict[LEFT_ANKLE]

    # محاسبه زاویه زانو
    knee_angle = calculate_angle_2d(hip, knee, ankle)

    # ارزیابی بیومکانیکی
    if knee_angle > settings.SQUAT_KNEE_ANGLE_THRESHOLD:
        return PoseFeedbackResponse(
            frame_id=payload.frame_id,
            is_valid=False,
            exercise_name=payload.exercise_name,
            feedback_message=f"بیشتر پایین برو! زاویه فعلی: {int(knee_angle)} درجه",
            error_code="NOT_DEEP_ENOUGH",
            highlight_joints=[LEFT_KNEE]
        )
    else:
        return PoseFeedbackResponse(
            frame_id=payload.frame_id,
            is_valid=True,
            exercise_name=payload.exercise_name,
            feedback_message="حرکت عالی و استاندارد است",
            error_code=None,
            highlight_joints=[]
        )