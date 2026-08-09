"""
سناریو و کد روش اول: توسعه معماری (Class Expansion)
سناریو: سیستم باید ۱۰۰۰ کلاس پیش‌فرض ImageNet را کاملاً دست‌نخورده نگه دارد و ۳ کلاس جدید ما را به انتهای لایه خروجی پیوند بزند.
"""

import os
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

def train_incremental_architecture(data_dir: str, epochs: int = 3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # بررسی وجود پوشه داده‌ها
    if not os.path.exists(data_dir):
        print(f"❌ پوشه داده‌ها پیدا نشد: {os.path.abspath(data_dir)}")
        print("💡 لطفاً از وجود پوشه dataset در مسیر مطمئن شوید.")
        return

    # ۱. بارگذاری مدل پیش‌فرض ImageNet (۱۰۰۰ کلاسه)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # ۲. فریز کردن تمامی لایه‌های استخراج ویژگی
    for param in model.parameters():
        param.requires_grad = False
        
    # ۳. استخراج لایه خطی قدیمی
    old_fc = model.fc
    in_features = old_fc.in_features          # ۵۱۲
    num_old_classes = old_fc.out_features      # ۱۰۰۰
    
    # ۴. آماده‌سازی داده‌های جدید
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    new_classes = dataset.classes # کلاس‌های جدید
    num_new_classes = len(new_classes)
    total_classes = num_old_classes + num_new_classes # ۱۰۰۳ کلاس
    
    # ۵. ساخت لایه FC جدید با ابعاد ۱۰-۳
    new_fc = nn.Linear(in_features, total_classes)
    
    # ۶. انتقال وزن‌های ۱۰۰۰ کلاس قدیمی به لایه جدید
    with torch.no_grad():
        new_fc.weight[:num_old_classes] = old_fc.weight
        new_fc.bias[:num_old_classes] = old_fc.bias
        
    model.fc = new_fc
    model = model.to(device)
    
    # ۷. بهینه‌ساز برای لایه خروجی جدید
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    print(f"🚀 شروع آموزش توسعه معماری (مجموع کلاس‌ها: {total_classes})...")
    
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            # نگاشت لیبل‌ها: 0,1,2 -> 1000, 1001, 1002
            shifted_labels = (labels + num_old_classes).to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, shifted_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(dataset) if len(dataset) > 0 else 0.0
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

    # ذخیره چک‌پوینت
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'all_classes': [f"imagenet_{i}" for i in range(1000)] + new_classes
    }
    torch.save(checkpoint, "expanded_model.pth")
    print("💾 مدل ۱۰-۳ کلاسه در expanded_model.pth ذخیره شد.")

if __name__ == "__main__":
    # محاسبه مسیر مطلق پوشه dataset در پوشه بالاتر (MLP/dataset)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.abspath(os.path.join(current_dir, "..", "dataset"))
    
    train_incremental_architecture(dataset_path)