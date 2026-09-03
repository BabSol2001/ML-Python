import os
import sys
import cv2
import numpy as np
from typing import List, Union

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.st_gcn_loader import STGCNLoader
# تصحیح مسیر ایمپورت از فایل هم‌مسیر visualize_pipeline
from models.visualize_pipeline import MediaPipePoseExtractor, process_video_and_save_visualization


def process_video_and_predict(
    video_source: Union[str, int], 
    loader: STGCNLoader, 
    sequence_length: int = 60,
    output_video_path: str = "output_processed.mp4"
):
    if isinstance(video_source, str) and not os.path.exists(video_source):
        print(f"[Error] فایل ویدیویی یافت نشد: '{os.path.abspath(video_source)}'")
        return

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[Error] امکان باز کردن منبع ویدیویی '{video_source}' وجود ندارد.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 25.0

    extractor = MediaPipePoseExtractor(model_path="pose_landmarker.task")
    pose_buffer: List[np.ndarray] = []

    print("[Info] تست و ارزیابی پایپ‌لاین ST-GCN شروع شد...")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_count += 1
        timestamp_ms = int((frame_count / fps) * 1000)

        # استخراج مفاصل فریم جاری
        kpts = extractor.extract_frame_keypoints(frame, timestamp_ms)

        if kpts is not None:
            pose_buffer.append(kpts)
        else:
            pose_buffer.append(np.zeros((17, 3), dtype=np.float32))

        if len(pose_buffer) > sequence_length:
            pose_buffer.pop(0)

        if len(pose_buffer) == sequence_length:
            sequence_array = np.array(pose_buffer, dtype=np.float32)
            results = loader.predict(sequence_array, return_softmax=True)

            pred_class = results['class_id'][0] if isinstance(results['class_id'], list) else results['class_id']
            confidence = results['confidence'][0] if isinstance(results['confidence'], list) else results['confidence']

            if frame_count % 15 == 0:
                print(f"[Frame {frame_count}] Action: Class {pred_class} | Confidence: {confidence * 100:.1f}%")

    cap.release()
    extractor.close()
    print(f"[Success] ارزیابی اولیه با موفقیت برای {frame_count} فریم به پایان رسید.\n")

    # فراخوانی خودکار رندر و ساخت فایل ویدیویی
    if isinstance(video_source, str):
        print("[Info] فراخوانی خودکار visualize_pipeline جهت رندر و ساخت ویدیو...")
        process_video_and_save_visualization(
            input_video_path=video_source,
            output_video_path=output_video_path,
            loader=loader,
            sequence_length=sequence_length
        )


if __name__ == "__main__":
    print("[Info] در حال راه‌اندازی مدل ST-GCN...")
    stgcn_loader = STGCNLoader(
        checkpoint_path=None,
        in_channels=3,
        num_class=10,
        layout='coco_17',
        device='cpu'
    )

    VIDEO_PATH = "test_video.mp4"
    OUTPUT_PATH = "output_processed.mp4"
    
    process_video_and_predict(
        video_source=VIDEO_PATH, 
        loader=stgcn_loader, 
        sequence_length=60,
        output_video_path=OUTPUT_PATH
    )