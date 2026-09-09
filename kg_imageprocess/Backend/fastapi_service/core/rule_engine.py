from typing import Dict, Tuple, Any, Optional
from schemas.pose_schema import PoseFramePayload, PoseFeedbackResponse
from core.pose_calculator import calculate_angle_2d
from config import settings

# شناسه مفاصل بر اساس استاندارد MediaPipe Pose
LEFT_HIP = 23
LEFT_KNEE = 25
LEFT_ANKLE = 27

RIGHT_HIP = 24
RIGHT_KNEE = 26
RIGHT_ANKLE = 28


def get_dynamic_thresholds(athlete_context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    تنظیم پویای حد آستانه زوایا بر اساس سوابق آسیب‌دیدگی و وضعیت کاربر در جانگو
    """
    # حد آستانه استاندارد پیش‌فرض از تنظیمات سیستم
    knee_threshold = settings.SQUAT_KNEE_ANGLE_THRESHOLD

    if not athlete_context:
        return {"knee_angle_threshold": knee_threshold}

    profile = athlete_context.get("profile", {}) or {}
    injury_history = [str(inj).lower() for inj in profile.get("injury_history", [])]
    postural_deviations = [str(dev).lower() for dev in profile.get("postural_deviations", [])]

    # اگر کاربر سابقه آسیب زانو، رباط صلیبی (ACL) یا منیسک داشته باشد:
    if any("knee" in inj or "acl" in inj or "meniscus" in inj for inj in injury_history):
        knee_threshold = max(knee_threshold, 100.0)  # محدود کردن عمق حرکت برای ایمنی زانو

    # اگر کاربر آسیب/ناهنجاری خاصی در ستون فقرات داشته باشد:
    if any("back" in inj or "spine" in inj or "lumbar" in inj for inj in injury_history) or \
       any("kyphosis" in dev or "lordosis" in dev for dev in postural_deviations):
        knee_threshold = max(knee_threshold, 95.0)

    return {"knee_angle_threshold": knee_threshold}


def check_squat_rules(
    payload: PoseFramePayload, 
    athlete_context: Optional[Dict[str, Any]] = None
) -> PoseFeedbackResponse:
    """
    بررسی قوانین حرکت اسکوات بر اساس زوایای مفاصل و پرونده پزشکی/بیومکانیکی کاربر
    """
    # تبدیل لیست مفصل‌ها به دیکشنری برای دسترسی سریع O(1)
    keypoints_dict: Dict[int, Tuple[float, float]] = {
        kp.id: (kp.x, kp.y) for kp in payload.keypoints if kp.score > 0.5
    }

    # محاسبه آستانه پویا متناسب با شرایط کاربر
    thresholds = get_dynamic_thresholds(athlete_context)
    applied_knee_threshold = thresholds["knee_angle_threshold"]

    # مطمئن شویم سه مفصل اصلی مربوط به پای چپ موجود هستند
    if not all(k in keypoints_dict for k in [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE]):
        return PoseFeedbackResponse(
            frame_id=payload.frame_id,
            is_valid=False,
            exercise_name=payload.exercise_name,
            feedback_message="مفاصل پای چپ به طور کامل در تصویر دیده نمی‌شوند",
            error_code="JOINTS_NOT_VISIBLE",
            highlight_joints=[]
        )

    # استخراج مختصات (لگن - زانو - مچ پا)
    hip = keypoints_dict[LEFT_HIP]
    knee = keypoints_dict[LEFT_KNEE]
    ankle = keypoints_dict[LEFT_ANKLE]

    # محاسبه زاویه زانو
    knee_angle = calculate_angle_2d(hip, knee, ankle)

    # ارزیابی بیومکانیکی بر اساس آستانه شخصی‌سازی‌شده
    if knee_angle > applied_knee_threshold:
        return PoseFeedbackResponse(
            frame_id=payload.frame_id,
            is_valid=False,
            exercise_name=payload.exercise_name,
            feedback_message=f"بیشتر پایین برو! زاویه فعلی: {int(knee_angle)} درجه (حداقل مجاز برای شما: {int(applied_knee_threshold)} درجه)",
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