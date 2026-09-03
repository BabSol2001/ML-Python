import os
import sys
import urllib.request
import cv2
import numpy as np
import torch
from typing import List, Optional, Union

# اضافه کردن مسیر جاری جهت تضمین ایمپورت‌ها
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# بارگذاری مدل‌های ST-GCN و Graph
from models.st_gcn_loader import STGCNLoader, Graph

# بارگذاری تابع محاسبه زوایا
try:
    from ..core.pose_calculator import annotate_frame_angles
except ImportError:
    try:
        from core.pose_calculator import annotate_frame_angles
    except ImportError:
        def annotate_frame_angles(frame, keypoints, visibility_threshold=0.3):
            return frame


def download_model_if_not_exists(model_path: str = "pose_landmarker.task"):
    if not os.path.exists(model_path):
        print(f"[Info] فایل مدل '{model_path}' یافت نشد. در حال دانلود...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        try:
            urllib.request.urlretrieve(url, model_path)
            print("[Info] دانلود با موفقیت انجام شد.")
        except Exception as e:
            print(f"[Error] خطا در دانلود مدل: {e}")
            raise e


class SkeletonVisualizer:
    """
    کلاس اختصاصی برای رسم خطوط اسکلتی، نقاط مفاصل و لایه نمایش اطلاعات روی فریم
    """
    def __init__(self, layout: str = 'coco_17'):
        graph = Graph(layout=layout)
        self.edges = [(i, j) for i, j in graph.edge if i != j]
        
        self.joint_color = (0, 255, 255)    # زرد
        self.edge_color = (255, 0, 128)     # بنفش / صورتی
        self.bbox_bg_color = (20, 20, 20)   # خاکستری تیره

    def draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray, visibility_threshold: float = 0.3) -> np.ndarray:
        h, w, _ = frame.shape
        pixel_kpts = []

        for pt in keypoints:
            x, y, vis = pt[0], pt[1], pt[2]
            px, py = int(x * w), int(y * h)
            pixel_kpts.append((px, py, vis))

        # 1. رسم استخوان‌ها
        for i, j in self.edges:
            p1 = pixel_kpts[i]
            p2 = pixel_kpts[j]
            if p1[2] >= visibility_threshold and p2[2] >= visibility_threshold:
                cv2.line(frame, (p1[0], p1[1]), (p2[0], p2[1]), self.edge_color, 3, cv2.LINE_AA)

        # 2. رسم نقاط مفاصل
        for px, py, vis in pixel_kpts:
            if vis >= visibility_threshold:
                cv2.circle(frame, (px, py), 5, self.joint_color, -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 6, (0, 0, 0), 1, cv2.LINE_AA)

        return frame

    def draw_info_box(self, frame: np.ndarray, text: str, confidence_str: str) -> np.ndarray:
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (450, 95), self.bbox_bg_color, -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        cv2.rectangle(frame, (20, 20), (450, 95), (0, 255, 0), 2)
        cv2.putText(frame, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, confidence_str, (35, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        return frame


class MediaPipePoseExtractor:
    def __init__(self, model_path: str = "pose_landmarker.task"):
        download_model_if_not_exists(model_path)
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.coco_indices = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    def extract_frame_keypoints(self, frame: np.ndarray, timestamp_ms: int) -> Optional[np.ndarray]:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        results = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return None

        landmarks = results.pose_landmarks[0]
        keypoints = []
        for idx in self.coco_indices:
            lm = landmarks[idx]
            keypoints.append([lm.x, lm.y, getattr(lm, 'visibility', 1.0)])

        return np.array(keypoints, dtype=np.float32)

    def close(self):
        self.landmarker.close()


def process_video_and_save_visualization(
    input_video_path: str,
    output_video_path: str,
    loader: STGCNLoader,
    sequence_length: int = 60
):
    if not os.path.exists(input_video_path):
        print(f"[Error] ویدیو ورودی یافت نشد: {os.path.abspath(input_video_path)}")
        return

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"[Error] OpenCV نتوانست فایل ویدیو را باز کند.")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 25.0

    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    extractor = MediaPipePoseExtractor()
    visualizer = SkeletonVisualizer(layout='coco_17')
    pose_buffer: List[np.ndarray] = []

    print(f"[Info] پردازش ویدیو و رندر کردن خروجی روی '{output_video_path}' شروع شد...")

    action_text = "Buffering skeleton..."
    conf_text = f"Frames: 0/{sequence_length}"
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_count += 1
        timestamp_ms = int((frame_count / fps) * 1000)

        # 1. استخراج mfaasel
        kpts = extractor.extract_frame_keypoints(frame, timestamp_ms)

        if kpts is not None:
            pose_buffer.append(kpts)
            # 2. رسم اسکلت
            frame = visualizer.draw_skeleton(frame, kpts)
            # 3. محاسبه و رسم زوایا
            try:
                annotated = annotate_frame_angles(frame, kpts, visibility_threshold=0.3)
                if annotated is not None and isinstance(annotated, np.ndarray):
                    frame = annotated
            except Exception:
                pass
        else:
            pose_buffer.append(np.zeros((17, 3), dtype=np.float32))

        if len(pose_buffer) > sequence_length:
            pose_buffer.pop(0)

        # 4. پیش‌بینی ST-GCN
        if len(pose_buffer) == sequence_length:
            sequence_array = np.array(pose_buffer, dtype=np.float32)
            results = loader.predict(sequence_array, return_softmax=True)

            pred_class = results['class_id'][0] if isinstance(results['class_id'], list) else results['class_id']
            confidence = results['confidence'][0] if isinstance(results['confidence'], list) else results['confidence']

            action_text = f"Action: Class {pred_class}"
            conf_text = f"Confidence: {confidence * 100:.1f}%"
        else:
            conf_text = f"Buffering: {len(pose_buffer)}/{sequence_length}"

        # 5. رسم داشبورد اطلاعات
        frame = visualizer.draw_info_box(frame, action_text, conf_text)

        # 6. ذخیره‌سازی فریم
        out.write(frame)

        if frame_count % 30 == 0:
            print(f" -> {frame_count} فریم پردازش شد...")

    cap.release()
    out.release()
    extractor.close()
    
    print(f"[Info] پردازش تمام شد. ویدیوی جدید در مسیر زیر ذخیره شد:\n {os.path.abspath(output_video_path)}")


# ============================================================================
# بخش اصلی اجرا
# ============================================================================
if __name__ == "__main__":
    stgcn_loader = STGCNLoader(
        checkpoint_path=None,
        in_channels=3,
        num_class=10,
        layout='coco_17',
        device='cpu'
    )

    INPUT_VIDEO = "test_video.mp4"
    OUTPUT_VIDEO = "output_processed.mp4"

    process_video_and_save_visualization(
        input_video_path=INPUT_VIDEO,
        output_video_path=OUTPUT_VIDEO,
        loader=stgcn_loader,
        sequence_length=60
    )