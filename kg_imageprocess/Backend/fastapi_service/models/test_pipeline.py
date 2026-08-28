import cv2
import numpy as np
import torch
from typing import List, Optional
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose # type: ignore # ایمپورت مستقیم ساب‌ماژول

# بارگذاری ماژول STGCNLoader که در مراحل قبلی توسعه داده شد
from models.st_gcn_loader import STGCNLoader


class MediaPipePoseExtractor:
    """
    استخراج مفاصل از ویدیو با استفاده از MediaPipe و نگاشت آن به فرمت استاندارد COCO-17
    """
    def __init__(self):
        # استفاده مستقیم از mp_pose
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # نگاشت اندیس‌های MediaPipe (33 نقطه) به COCO-17 (17 نقطه)
        # MediaPipe indices: 0:Nose, 2:L_Eye, 5:R_Eye, 7:L_Ear, 8:R_Ear, 11:L_Shoulder, 12:R_Shoulder,
        # 13:L_Elbow, 14:R_Elbow, 15:L_Wrist, 16:R_Wrist, 23:L_Hip, 24:R_Hip, 25:L_Knee, 26:R_Knee, 27:L_Ankle, 28:R_Ankle
        self.coco_indices = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    def extract_frame_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        دریافت یک فریم BGR و خروجی دادن آرایه‌ای با ابعاد (17, 3) به فرمت [X, Y, Confidence]
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark
        keypoints = []

        for idx in self.coco_indices:
            lm = landmarks[idx]
            # ذخیره مختصات نرمال‌شده و میزان اطمینان (Visibility)
            keypoints.append([lm.x, lm.y, lm.visibility])

        return np.array(keypoints, dtype=np.float32)  # Shape: (17, 3)

    def close(self):
        self.pose.close()


def process_video_and_predict(
    video_source: str | int, 
    loader: STGCNLoader, 
    sequence_length: int = 60
):
    """
    خوانش ویدیو/وب‌کم، استخراج مفاصل و ارسال پنجره‌های زمانی به STGCNLoader
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[Error] امکان باز کردن منبع ویدیویی '{video_source}' وجود ندارد.")
        return

    extractor = MediaPipePoseExtractor()
    pose_buffer: List[np.ndarray] = []

    print("[Info] پردازش ویدیو شروع شد... برای خروج کلید 'q' را فشار دهید.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # استخراج مفاصل فریم جاری
        kpts = extractor.extract_frame_keypoints(frame)

        if kpts is not None:
            pose_buffer.append(kpts)
        else:
            # در صورت عدم تشخیص انسان، فریم خالی (صفر) اضافه می‌شود
            pose_buffer.append(np.zeros((17, 3), dtype=np.float32))

        # نگه داشتن فقط sequence_length فریم اخیر
        if len(pose_buffer) > sequence_length:
            pose_buffer.pop(0)

        # انجام استنتاج زمانی که بافر به اندازه مورد نظر رسید
        display_text = "Extracting pose..."
        if len(pose_buffer) == sequence_length:
            # ساخت تنسور با شکل (T, V, C) -> ( sequence_length, 17, 3 )
            sequence_array = np.array(pose_buffer, dtype=np.float32)

            # پیش‌بینی با STGCNLoader
            results = loader.predict(sequence_array, return_softmax=True)

            pred_class = results['class_id'][0] if isinstance(results['class_id'], list) else results['class_id']
            confidence = results['confidence'][0] if isinstance(results['confidence'], list) else results['confidence']

            display_text = f"Action: Class {pred_class} ({confidence * 100:.1f}%)"

        # نمایش نتیجه روی فریم
        cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("ST-GCN Pose Pipeline Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    extractor.close()
    cv2.destroyAllWindows()


# ============================================================================
# بخش اصلی اجرا
# ============================================================================
if __name__ == "__main__":
    # ۱. راه‌اندازی ومدل ST-GCN (با 10 کلاس نمونه)
    print("[Info] در حال راه‌اندازی مدل ST-GCN...")
    stgcn_loader = STGCNLoader(
        checkpoint_path=None,  # در صورت داشتن فایل .pth، مسیر آن را اینجا وارد کنید
        in_channels=3,        # (X, Y, Visibility)
        num_class=10,
        layout='coco_17',
        device='cpu'
    )

    # ۲. تعیین منبع ویدیو: 
    # برای تست با وب‌کم: 0
    # برای تست با فایل ویدیو: "path/to/video.mp4"
    VIDEO_PATH = 0  

    process_video_and_predict(video_source=VIDEO_PATH, loader=stgcn_loader, sequence_length=60)