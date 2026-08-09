import os
import json
import urllib.request
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def get_imagenet_classes():
    url = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagenet_class_index.json")
    
    if not os.path.exists(file_path):
        try:
            print("📥 در حال دریافت فایل اسامی کلاس‌های ImageNet...")
            urllib.request.urlretrieve(url, file_path)
        except Exception:
            return [f"imagenet_{i}" for i in range(1000)]
    
    with open(file_path) as f:
        class_idx = json.load(f)
    return [class_idx[str(i)][1] for i in range(1000)]

def test_replay_system():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, "replay_model.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ فایل مدل یافت نشد: {checkpoint_path}")
        print("💡 لطفاً ابتدا rehearsal_buffer.py را اجرا کنید تا مدل ساخته شود.")
        return

    # بارگذاری اسامی کلاس‌ها
    imagenet_classes = get_imagenet_classes()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    new_classes = checkpoint.get('new_classes', [])
    all_classes = imagenet_classes + new_classes

    # بازسازی مدل
    print("📦 در حال بارگذاری مدل Experience Replay...")
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(all_classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("\n✅ سیستم تست Replay آماده است!")
    print("=" * 65)

    while True:
        image_path = input("\n🖼️ مسیر عکس برای تست را وارد کنید (یا عبارت exit برای خروج): ").strip()
        image_path = image_path.strip("'\"")

        if image_path.lower() in ['exit', 'quit', 'q']:
            print("👋 خروج از برنامه.")
            break

        if not os.path.exists(image_path):
            print(f"❌ تصویر یافت نشد: {image_path}")
            continue

        try:
            image = Image.open(image_path).convert('RGB')
            tensor = transform(image)
            input_tensor = torch.unsqueeze(tensor, 0).to(device) # type: ignore

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                top5_prob, top5_catid = torch.topk(probabilities, k=min(5, len(all_classes)))

            print(f"\n🔍 ۵ حدس برتر مدل Replay برای: {os.path.basename(image_path)}")
            print("-" * 65)
            for i in range(top5_prob.size(0)):
                class_idx = top5_catid[i].item()
                class_name = all_classes[class_idx]
                confidence = top5_prob[i].item() * 100
                print(f"{i+1}. کلاس: {class_name:<35} | میزان اطمینان: {confidence:.2f}%")
            print("-" * 65)

        except Exception as e:
            print(f"❌ خطا در پردازش تصویر: {e}")

if __name__ == "__main__":
    test_replay_system()