import math
from typing import Tuple

def calculate_angle_2d(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    """
    محاسبه زاویه بین سه نقطه (p1 - p2 - p3) که p2 راس زاویه است.
    خروجی زاویه به درجه (بین 0 تا 180) است.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # بردار p2 -> p1 و p2 -> p3
    v1 = (x1 - x2, y1 - y2)
    v2 = (x3 - x2, y3 - y2)

    # ضرب داخلی و قدر مطلق برداره‌ها
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)

    # جلوگیری از تقسیم بر صفر
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0

    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
    # محدود کردن مقدار بین -1 تا 1 برای جلوگیری از خطای acos
    cosine_angle = max(-1.0, min(1.0, cosine_angle))

    angle = math.acos(cosine_angle)
    return math.degrees(angle)