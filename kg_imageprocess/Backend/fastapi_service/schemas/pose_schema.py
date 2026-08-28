from pydantic import BaseModel, Field
from typing import List, Optional

class Keypoint(BaseModel):
    """مدل مربوط به هر نقطه مفصلی بدن (مثلاً مچ پا، زانو و ...)"""
    id: int = Field(..., description="شناسه مفصل بر اساس استاندارد Pose Detection")
    name: str = Field(..., description="نام مفصل (مثل left_knee)")
    x: float = Field(..., description="مختصات x نرمال‌شده بین 0 تا 1")
    y: float = Field(..., description="مختصات y نرمال‌شده بین 0 تا 1")
    z: Optional[float] = Field(default=0.0, description="عمق ۳ بعدی در صورت وجود")
    score: float = Field(..., description="میزان اطمینان مدل از وجود مفصل")

class PoseFramePayload(BaseModel):
    """ساختار داده کل فریم دریافتی از طریق وب‌سوکت"""
    frame_id: int
    timestamp: float
    exercise_name: str = Field(default="squat", description="نام حرکت ورزشی در حال انجام")
    keypoints: List[Keypoint]

class PoseFeedbackResponse(BaseModel):
    """ساختار بازخورد خروجی سریع برای فرستادن به فلاتر"""
    frame_id: int
    is_valid: bool
    exercise_name: str
    feedback_message: str
    error_code: Optional[str] = None
    highlight_joints: List[int] = Field(default_factory=list, description="لیست آیدی مفصل‌هایی که باید قرمز شوند")