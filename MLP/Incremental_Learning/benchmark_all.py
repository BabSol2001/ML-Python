import os
import warnings
import joblib
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# خاموش کردن هشدارهای غیرضروری PIL
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

def benchmark_all_methods(test_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.exists(test_dir):
        print(f"❌ پوشه داده‌های تست یافت نشد: {os.path.abspath(test_dir)}")
        return

    # پیش‌پردازش همراه با تبدیل تصاویر به RGB جهت رفع هشدار شفافیت
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB') if isinstance(img, Image.Image) else img),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    results = {}

    print("📊 شروع ارزیابی مقایسه‌ای مدل‌ها روی داده‌های تست...")
    print("=" * 65)

    # ----------------------------------------------------
    # ۱. ارزیابی روش اول: Class Expansion
    # ----------------------------------------------------
    path_m1 = os.path.join(current_dir, "class_expansion.pth")
    if os.path.exists(path_m1):
        try:
            checkpoint = torch.load(path_m1, map_location=device)
            m1 = models.resnet18()
            # استخراج خودکار تعداد کلاس‌ها از چک‌پوینت
            saved_classes = checkpoint['model_state_dict']['fc.weight'].shape[0]
            m1.fc = nn.Linear(m1.fc.in_features, saved_classes)
            m1.load_state_dict(checkpoint['model_state_dict'])
            m1 = m1.to(device).eval()
            
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    shifted_labels = labels.to(device) + 1000
                    outputs = m1(inputs)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == shifted_labels).sum().item()
                    total += labels.size(0)
            results["1. Class Expansion"] = (correct / total) * 100
        except Exception as e:
            results["1. Class Expansion"] = f"خطا: {e}"
    else:
        results["1. Class Expansion"] = "فایل یافت نشد"

    # ----------------------------------------------------
    # ۲. ارزیابی روش دوم: Feature Extraction + SGD
    # ----------------------------------------------------
    path_m2 = os.path.join(current_dir, "incremental_svm.pkl")
    if os.path.exists(path_m2):
        try:
            model_data = joblib.load(path_m2)
            clf = model_data['classifier']
            base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            fe = nn.Sequential(*list(base_model.children())[:-1]).to(device).eval()
            
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    feats = fe(inputs)
                    feats = torch.flatten(feats, 1).cpu().numpy()
                    preds = clf.predict(feats)
                    correct += np.sum(preds == labels.numpy())
                    total += labels.size(0)
            results["2. Feature Extractor + SGD"] = (correct / total) * 100
        except Exception as e:
            results["2. Feature Extractor + SGD"] = f"خطا: {e}"
    else:
        results["2. Feature Extractor + SGD"] = "فایل یافت نشد"

    # ----------------------------------------------------
    # ۳. ارزیابی روش سوم: LwF (Distillation)
    # ----------------------------------------------------
    path_m3 = os.path.join(current_dir, "lwf_model.pth")
    if os.path.exists(path_m3):
        try:
            checkpoint = torch.load(path_m3, map_location=device)
            m3 = models.resnet18()
            # استخراج خودکار تعداد کلاس‌ها از چک‌پوینت
            saved_classes = checkpoint['model_state_dict']['fc.weight'].shape[0]
            m3.fc = nn.Linear(m3.fc.in_features, saved_classes)
            m3.load_state_dict(checkpoint['model_state_dict'])
            m3 = m3.to(device).eval()
            
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    shifted_labels = labels.to(device) + 1000
                    outputs = m3(inputs)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == shifted_labels).sum().item()
                    total += labels.size(0)
            results["3. LwF (Distillation)"] = (correct / total) * 100
        except Exception as e:
            results["3. LwF (Distillation)"] = f"خطا: {e}"
    else:
        results["3. LwF (Distillation)"] = "فایل یافت نشد"

    # ----------------------------------------------------
    # ۴. ارزیابی روش چهارم: Experience Replay
    # ----------------------------------------------------
    path_m4 = os.path.join(current_dir, "replay_model.pth")
    if os.path.exists(path_m4):
        try:
            checkpoint = torch.load(path_m4, map_location=device)
            m4 = models.resnet18()
            # استخراج خودکار تعداد کلاس‌ها از چک‌پوینت
            saved_classes = checkpoint['model_state_dict']['fc.weight'].shape[0]
            m4.fc = nn.Linear(m4.fc.in_features, saved_classes)
            m4.load_state_dict(checkpoint['model_state_dict'])
            m4 = m4.to(device).eval()
            
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    shifted_labels = labels.to(device) + 1000
                    outputs = m4(inputs)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == shifted_labels).sum().item()
                    total += labels.size(0)
            results["4. Experience Replay"] = (correct / total) * 100
        except Exception as e:
            results["4. Experience Replay"] = f"خطا: {e}"
    else:
        results["4. Experience Replay"] = "فایل یافت نشد"

    # ----------------------------------------------------
    # نمایش خروجی نهایی به‌صورت جدول
    # ----------------------------------------------------
    print("\n🏆 جدول خروجی بنچمارک دقت روی کلاس‌های جدید (New Class Accuracy):")
    print("-" * 65)
    print(f"{'نام روش':<35} | {'دقت (Accuracy)':<20}")
    print("-" * 65)
    for method, acc in results.items():
        if isinstance(acc, float):
            print(f"{method:<35} | {acc:.2f}%")
        else:
            print(f"{method:<35} | {acc}")
    print("-" * 65)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.abspath(os.path.join(current_dir, "..", "dataset"))
    benchmark_all_methods(dataset_path)