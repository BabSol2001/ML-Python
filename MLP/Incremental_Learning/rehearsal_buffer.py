import os
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from typing import Optional

class ShiftedDataset(Dataset):
    """کلاس کمکی برای شیفت دادن اندیس‌های کلاس داده‌های جدید به ۱۰-۳"""
    def __init__(self, dataset, offset: int):
        self.dataset = dataset
        self.offset = offset

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return img, label + self.offset

    def __len__(self):
        return len(self.dataset)

def train_with_replay(new_data_dir: str, replay_data_dir: Optional[str] = None, epochs: int = 3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # بررسی وجود پوشه داده‌های جدید
    if not os.path.exists(new_data_dir):
        print(f"❌ پوشه داده‌های جدید پیدا نشد: {os.path.abspath(new_data_dir)}")
        print("💡 لطفاً از وجود پوشه dataset در مسیر اصلی مطمئن شوید.")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # ۱. بارگذاری داده‌های جدید
    raw_new_dataset = datasets.ImageFolder(new_data_dir, transform=transform)
    num_old_classes = 1000
    num_new_classes = len(raw_new_dataset.classes)
    total_classes = num_old_classes + num_new_classes # ۱۰۰۳ کلاس
    
    # شیفت دادن لیبل‌های جدید به ۱۰۰۰، ۱۰۰۱ و ۱۰۰۲
    new_dataset = ShiftedDataset(raw_new_dataset, offset=num_old_classes)
    
    # ۲. ترکیب با داده‌های بافر قدیمی (در صورت وجود)
    final_dataset: Dataset
    if replay_data_dir is not None and os.path.exists(replay_data_dir):
        replay_dataset = datasets.ImageFolder(replay_data_dir, transform=transform)
        final_dataset = ConcatDataset([new_dataset, replay_dataset])
        print(f"🔄 {len(replay_dataset)} نمونه از حافظه قدیمی با داده‌های جدید ترکیب شدند.")
    else:
        final_dataset = new_dataset
        print("ℹ️ بافر بازپخش فعال نیست؛ آموزش فقط روی داده‌های جدید اجرا می‌شود.")
        
    dataloader = DataLoader(final_dataset, batch_size=8, shuffle=True)
    
    # ۳. آماده‌سازی مدل ۱۰-۳ کلاسه با حفظ وزن‌های قبلی
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    
    old_fc = model.fc
    new_fc = nn.Linear(num_ftrs, total_classes)
    with torch.no_grad():
        new_fc.weight[:num_old_classes] = old_fc.weight
        new_fc.bias[:num_old_classes] = old_fc.bias
    model.fc = new_fc
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    print(f"🚀 شروع آموزش Experience Replay (تعداد نمونه‌ها: {len(final_dataset)})...")
    
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
            
        total_samples = len(final_dataset)
        epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

    # ۴. ذخیره چک‌پوینت مدل
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'new_classes': raw_new_dataset.classes
    }
    checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay_model.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 مدل با موفقیت در {os.path.basename(checkpoint_path)} ذخیره شد.")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.abspath(os.path.join(current_dir, "..", "dataset"))
    
    # مسیر اختیاری برای پوشه بافر نمونه‌های قدیمی
    replay_buffer_path = os.path.abspath(os.path.join(current_dir, "..", "replay_buffer"))
    if not os.path.exists(replay_buffer_path):
        replay_buffer_path = None
        
    train_with_replay(dataset_path, replay_buffer_path)