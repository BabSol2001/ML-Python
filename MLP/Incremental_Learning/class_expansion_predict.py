import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def predict_image(image_path: str, checkpoint_path: str = "expanded_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(image_path):
        print(f"❌ تصویر یافت نشد: {image_path}")
        return

    # ۱. بارگذاری چک‌پوینت ذخیره‌شده
    checkpoint = torch.load(checkpoint_path, map_location=device)
    all_classes = checkpoint['all_classes']
    
    # ۲. ساخت معماری اولیه ResNet18
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(all_classes))
    
    # ۳. بارگذاری وزن‌های آموزش‌دیده
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # ۴. پیش‌پردازش تصویر ورودی
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    
    # ۵. تبدیل به تنسور و افزودن بعد Batch (حل خطای Pylance)
    tensor = transform(image)
    input_tensor = torch.unsqueeze(tensor, 0).to(device) # type: ignore

    # ۶. پیش‌بینی
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # ۳ پیش‌بینی برتر (Top-3 Predictions)
        top3_prob, top3_catid = torch.topk(probabilities, 3)

    print(f"\n🔍 نتیجه تحلیل برای تصویر: {os.path.basename(image_path)}")
    print("-" * 50)
    for i in range(top3_prob.size(0)):
        class_idx = top3_catid[i].item()
        class_name = all_classes[class_idx]
        confidence = top3_prob[i].item() * 100
        print(f"{i+1}. کلاس: {class_name:<30} | درصد اطمینان: {confidence:.2f}%")

if __name__ == "__main__":
    # دریافت مسیر عکس از کاربر
    img_path = input("\nمسیر عکس برای تست را وارد کنید (مثلا test.jpg): ").strip()
    predict_image(img_path)