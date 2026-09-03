import math
import cv2
import numpy as np
from typing import Tuple, Dict, Any, List, Optional


def calculate_angle_2d(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    """
    محاسبه زاویه بین سه نقطه (p1 - p2 - p3) که p2 رأس زاویه است.
    خروجی زاویه به درجه (بین 0 تا 180) است.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # بردار p2 -> p1 و p2 -> p3
    v1 = (x1 - x2, y1 - y2)
    v2 = (x3 - x2, y3 - y2)

    # ضرب داخلی و قدر مطلق بردارها
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)

    # جلوگیری از تقسیم بر صفر
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0

    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
    cosine_angle = max(-1.0, min(1.0, cosine_angle))

    angle = math.acos(cosine_angle)
    return math.degrees(angle)


def draw_angle_annotation(
    frame: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    p3: Tuple[int, int],
    label: str = ""
) -> float:
    """
    ترسیم مفاصل، خطوط اتصال، زاویه و متن روی فریم تصویر جهت ارزیابی بصری
    """
    # 1. محاسبه زاویه
    angle = calculate_angle_2d(p1, p2, p3)

    # 2. رسم خطوط بین مفاصل (p1 -> p2 و p2 -> p3)
    cv2.line(frame, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)  # خط زرد
    cv2.line(frame, p2, p3, (0, 255, 255), 2, cv2.LINE_AA)  # خط زرد

    # 3. رسم نقاط کلیدی (Keypoints)
    cv2.circle(frame, p1, 4, (0, 0, 255), -1, cv2.LINE_AA)  # نقطه ابتدا (قرمز)
    cv2.circle(frame, p2, 6, (0, 255, 0), -1, cv2.LINE_AA)  # رأس زاویه (سبز)
    cv2.circle(frame, p3, 4, (0, 0, 255), -1, cv2.LINE_AA)  # نقطه انتها (قرمز)

    # 4. آماده‌سازی متن زاویه
    text = f"{label}: {int(angle)}deg" if label else f"{int(angle)}deg"

    # 5. محاسبه موقعیت قرارگیری متن روی تصویر (کنار رأس زاویه p2)
    text_pos = (p2[0] + 10, p2[1] - 10)

    # رسم پس‌زمینه مشکی کوچک پشت متن برای خوانایی بیشتر
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(
        frame, 
        (text_pos[0] - 3, text_pos[1] - text_h - 3), 
        (text_pos[0] + text_w + 3, text_pos[1] + 3), 
        (0, 0, 0), 
        -1
    )

    # 6. نوشتن میزان زاویه به درجه
    cv2.putText(
        frame, 
        text, 
        text_pos, 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.5, 
        (255, 255, 255), 
        1, 
        cv2.LINE_AA
    )

    return angle


def annotate_frame_angles(frame: np.ndarray, keypoints: np.ndarray, visibility_threshold: float = 0.3) -> Dict[str, float]:
    """
    محاسبه و رسم خودکار زوایای اصلی بدن بر روی آرایه‌ای از keypoint‌های COCO-17
    keypoints: آرایه‌ای با ابعاد (17, 3) به‌صورت [x, y, visibility]
    """
    h, w, _ = frame.shape
    pixel_kpts = {}

    for idx, pt in enumerate(keypoints):
        x, y, vis = pt[0], pt[1], pt[2]
        px, py = int(x * w), int(y * h)
        pixel_kpts[idx] = (px, py, vis)

    # تعریف زوایا براساس استاندارد COCO-17:
    # آرنج چپ: شانه(5) -> آرنج(7) -> مچ(9)
    # آرنج راست: شانه(6) -> آرنج(8) -> مچ(10)
    # زانوی چپ: لگن(11) -> زانو(13) -> مچ پا(15)
    # زانوی راست: لگن(12) -> زانو(14) -> مچ پا(16)
    angle_definitions = [
        (5, 7, 9, "L-Elbow"),
        (6, 8, 10, "R-Elbow"),
        (11, 13, 15, "L-Knee"),
        (12, 14, 16, "R-Knee"),
    ]

    calculated_angles = {}

    for i1, i2, i3, label in angle_definitions:
        p1, p2, p3 = pixel_kpts[i1], pixel_kpts[i2], pixel_kpts[i3]
        if p1[2] >= visibility_threshold and p2[2] >= visibility_threshold and p3[2] >= visibility_threshold:
            ang = draw_angle_annotation(
                frame,
                (p1[0], p1[1]),
                (p2[0], p2[1]),
                (p3[0], p3[1]),
                label=label
            )
            calculated_angles[label] = ang

    return calculated_angles


if __name__ == "__main__":
    # تست سریع روی یک فریم نمونه
    test_img = np.zeros((600, 800, 3), dtype=np.uint8)
    
    shoulder = (200, 200)
    elbow = (350, 300)
    wrist = (500, 250)

    angle = draw_angle_annotation(test_img, shoulder, elbow, wrist, label="Elbow Test")
    print(f"[Info] Test Angle Calculated: {angle:.2f} Degrees")

    cv2.imshow("Pose Calculator Test", test_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()