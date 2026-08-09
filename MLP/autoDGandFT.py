import os
import time
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
from ddgs import DDGS

SEARCH_PROMPTS = {
    "leopard": "wild leopard animal panthera pardus nature photo",
    "پلنگ": "wild leopard animal panthera pardus nature photo",
    "lion": "wild lion animal panthera leo photo",
    "شیر": "wild lion animal panthera leo photo",
    "cat": "domestic cat animal pet photo",
    "گربه": "domestic cat animal pet photo",
    "dog": "dog animal pet photo",
    "سگ": "dog animal pet photo"
}

# -------------------------------------------------------------
# ۱. دانلود با DuckDuckGo (با مدیریت Rate Limit)
# -------------------------------------------------------------
def download_images_ddg(query: str, save_dir: str, limit: int = 40):
    os.makedirs(save_dir, exist_ok=True)
    print(f"🔍 در حال جستجو برای: '{query}'...")
    
    urls = []
    try:
        with DDGS() as ddgs:
            # دریافت نتایج جستجو
            results = list(ddgs.images(query, max_results=limit))
            for r in results:
                urls.append(r["image"])
    except Exception as e:
        print(f"⚠️ خطای دریافت از DuckDuckGo: {e}")
        print("💡 در حال تلاش مجدد...")
        return

    print(f"📥 {len(urls)} آدرس پیدا شد. در حال دانلود...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    count = 0
    for idx, url in enumerate(urls):
        try:
            resp = requests.get(url, headers=headers, timeout=6)
            if resp.status_code == 200:
                ext = url.split('.')[-1].split('?')[0].lower()
                if ext not in ['jpg', 'jpeg', 'png']:
                    ext = 'jpg'
                
                file_path = os.path.join(save_dir, f"img_{idx+1}.{ext}")
                with open(file_path, 'wb') as f:
                    f.write(resp.content)
                count += 1
                time.sleep(0.2)  # ایجاد وقفه کوتاه برای عدم مسدودی
        except Exception:
            continue
            
    print(f"✅ تعداد {count} تصویر در '{save_dir}' ذخیره شد.")

def download_dataset(target_label: str, limit: int = 40):
    clean_label = target_label.strip().lower()
    search_query = SEARCH_PROMPTS.get(clean_label, f"{target_label} animal photo nature")
    
    target_dir = os.path.join('dataset', target_label)
    other_dir = os.path.join('dataset', 'other_objects')
    
    download_images_ddg(query=search_query, save_dir=target_dir, limit=limit)
    time.sleep(2)  # وقفه بین دو جستجو جهت جلوگیری از ارور ۴۰۳

    if not os.path.exists(other_dir) or len(os.listdir(other_dir)) < 10:
        download_images_ddg(query="outdoor landscape furniture scenery photo", save_dir=other_dir, limit=limit)

# -------------------------------------------------------------
# ۲. پاک‌سازی تصاویر خراب
# -------------------------------------------------------------
def clean_corrupted_images(data_dir: str):
    print("🧹 در حال بررسی تصاویر...")
    removed_count = 0
    for root, _, files in os.walk(data_dir):
        for file in files:
            path = os.path.join(root, file)
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception:
                os.remove(path)
                removed_count += 1
    print(f"🧹 تعداد {removed_count} فایل غیرقابل استفاده حذف شد.")

# -------------------------------------------------------------
# ۳. آموزش مدل PyTorch
# -------------------------------------------------------------
def train_model(data_dir: str, epochs: int = 6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ پردازش روی: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    class_names = dataset.classes
    print(f"📊 کلاس‌های مدل: {class_names}")
    
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    for param in model.parameters():
        param.requires_grad = False
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    model.train()
    print("🚀 شروع فرآیند آموزش...")
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
        
    print("✅ آموزش مدل تکمیل شد!")
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }
    torch.save(checkpoint, "auto_trained_model.pth")
    print("💾 مدل در auto_trained_model.pth ذخیره شد.")

if __name__ == "__main__":
    target = input("نام سوژه‌ای که می‌خواهید مدل یاد بگیرد را وارد کنید (مثلا پلنگ یا Leopard): ")
    
    download_dataset(target_label=target, limit=40)
    clean_corrupted_images("dataset")
    train_model("dataset", epochs=6)