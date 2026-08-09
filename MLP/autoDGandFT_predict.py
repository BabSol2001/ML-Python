import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
import os

def load_trained_model(model_path="auto_trained_model.pth"):
    if not os.path.exists(model_path):
        print(f"❌ فایل مدل یافت نشد: {model_path}")
        sys.exit(1)

    # ۱. بارگذاری فایل چک‌پوینت
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    class_names = checkpoint['class_names']

    # ۲. بازسازی ساختار مدل ResNet18
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    # ۳. بارگذاری وزن‌های آموزش‌دیده
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # قرار دادن مدل در حالت ارزیابی (Inference)

    return model, class_names

def predict_image(image_path, model, class_names):
    if not os.path.exists(image_path):
        print(f"❌ فایل تصویر یافت نشد: {image_path}")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ۱. باز کردن تصویر
    image = Image.open(image_path).convert('RGB')
    
    # ۲. اعمال تبدیل
    tensor = transform(image)
    
    # ۳. استفاده از تابع torch.unsqueeze جهت رفع خطای Pylance/Pyright
    input_tensor = torch.unsqueeze(tensor, 0) # type: ignore

    # پیش‌بینی
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    confidence, predicted_idx = torch.max(probabilities, 0)
    predicted_class = class_names[predicted_idx.item()]

    print("\n" + "="*40)
    print(f"🖼️  تصویر ورودی: {image_path}")
    print(f"🎯 حدس هوش مصنوعی: {predicted_class}")
    print(f"📊 میزان اطمینان: {confidence.item() * 100:.2f}%")
    print("="*40)

    print("\nجزئیات درصدها برای همه کلاس‌ها:")
    for idx, name in enumerate(class_names):
        print(f"  • {name}: {probabilities[idx].item() * 100:.2f}%")

if __name__ == "__main__":
    model_file = "auto_trained_model.pth"
    model, class_names = load_trained_model(model_file)

    img_path = input("\nمسیر عکس جدید برای تست را وارد کنید (مثلا test.jpg): ").strip()
    predict_image(img_path, model, class_names)