import os
import torch
import joblib
import numpy as np
from torchvision import models, transforms
from PIL import Image

def evaluate_svm_system():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ۱. پیدا کردن مسیر فایل pkl به صورت مطلق
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pkl_path = os.path.join(current_dir, "incremental_svm.pkl")
    
    if not os.path.exists(pkl_path):
        print(f"❌ فایل مدل یافت نشد: {pkl_path}")
        print("💡 لطفاً ابتدا اسکریپت feature_extractor.py را اجرا کنید تا مدل آموزش دیده و فایل pkl ساخته شود.")
        return

    # ۲. بارگذاری مدل آنلاین ذخیره‌شده
    print("📦 در حال بارگذاری مدل کلاسیفایر (incremental_svm.pkl)...")
    model_data = joblib.load(pkl_path)
    clf = model_data['classifier']
    classes = model_data['classes']

    # ۳. آماده‌سازی Feature Extractor (ResNet18 بدون لایه FC)
    print("⚙️ در حال بارگذاری مدل استخراج ویژگی (ResNet18)...")
    base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    feature_extractor = torch.nn.Sequential(*list(base_model.children())[:-1])
    feature_extractor.eval().to(device)

    # ۴. تنظیمات پیش‌پردازش تصویر
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("\n✅ سیستم تست آماده است!")
    print("=" * 60)

    # ۵. حلقه تعاملی دریافت عکس و پیش‌بینی
    while True:
        image_path = input("\n🖼️ مسیر عکس برای تست را وارد کنید (یا عبارت exit برای خروج): ").strip()
        
        # پاک‌سازی کتیشن‌های احتمالی دور مسیر فایل در ویندوز
        image_path = image_path.strip("'\"")

        if image_path.lower() in ['exit', 'quit', 'q']:
            print("👋 خروج از برنامه.")
            break

        if not os.path.exists(image_path):
            print(f"❌ تصویر یافت نشد: {image_path}")
            continue

        try:
            # تبدیل تصویر و استخراج ویژگی
            image = Image.open(image_path).convert('RGB')
            tensor = transform(image)
            input_tensor = torch.unsqueeze(tensor, 0).to(device) # type: ignore

            with torch.no_grad():
                feats = feature_extractor(input_tensor)
                feats = torch.flatten(feats, 1).cpu().numpy()

            # محاسبه درصد احتمالات توسط SGDClassifier
            probabilities = clf.predict_proba(feats)[0]
            
            # مرتب‌سازی کلاس‌ها از بیشترین درصد اطمینان به کمترین
            sorted_indices = np.argsort(probabilities)[::-1]

            print(f"\n🔍 نتیجه پیش‌بینی برای تصویر: {os.path.basename(image_path)}")
            print("-" * 60)
            for idx in sorted_indices:
                class_name = classes[idx]
                prob = probabilities[idx] * 100
                print(f"کلاس: {class_name:<30} | میزان اطمینان: {prob:.2f}%")
            print("-" * 60)

        except Exception as e:
            print(f"❌ خطا در پردازش تصویر: {e}")

if __name__ == "__main__":
    evaluate_svm_system()