import matplotlib
matplotlib.use('Agg')  # برای جلوگیری از خطای گرافیکی در سرور یا محیط‌های بدون UI

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# ۱. تنظیمات اولیه و انتخاب سخت‌افزار (GPU یا CPU)
# -------------------------------------------------------------
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"⚡ اجرای محاسبات روی: {device}")

# -------------------------------------------------------------
# ۲. آماده‌سازی تصویر و تبدیل‌ها (Transforms)
# -------------------------------------------------------------
# مدلهای ResNet نیاز به تصاویر با ابعاد حداقل ۲۲۴x۲۲۴ و نرمال‌سازی ImageNet دارند
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), # دیتا افزایی ساده برای یادگیری بهتر
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------------------------------------------------
# ۳. دانلود و بارگذاری داده‌های CIFAR-10 با DataLoader
# -------------------------------------------------------------
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# نام کلاس‌های CIFAR-10
classes = train_dataset.classes
print(f"🏷️ کلاس‌های موجود: {classes}")

# -------------------------------------------------------------
# ۴. ساخت و سفارشی‌سازی مدل ResNet-18
# -------------------------------------------------------------
# بارگذاری وزن‌های پیش‌آموزش‌دیده ImageNet
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# انجماد (Freeze) لایه‌های استخراج ویژگی
for param in model.parameters():
    param.requires_grad = False

# جایگزینی لایه آخر (Fully Connected) متناسب با ۱۰ کلاس CIFAR-10
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # این لایه به صورت خودکار requires_grad=True دارد

# انتقال مدل به GPU/CPU
model = model.to(device)

# -------------------------------------------------------------
# ۵. تنظیم بهینه‌ساز و تابع زیان
# -------------------------------------------------------------
criterion = nn.CrossEntropyLoss()

# فقط پارامترهای لایه خروجی جدید به بهینه‌ساز داده می‌شوند
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# -------------------------------------------------------------
# ۶. حلقه آموزش (Training Loop)
# -------------------------------------------------------------
EPOCHS = 3  # به دلیل قدرت ResNet، در همین ۳ اپک دقت فوق‌العاده‌ای به دست می‌آید
train_losses = []

print("\n🚀 شروع آموزش مدل ResNet-18 روی CIFAR-10...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # چاپ وضعیت در طول Epoch
        if (batch_idx + 1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] - Batch [{batch_idx+1}/{len(train_loader)}] - Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)

# -------------------------------------------------------------
# ۷. ارزیابی مدل روی داده‌های تست (Test Evaluation)
# -------------------------------------------------------------
model.eval()
correct = 0
total = 0

print("\n🧪 در حال ارزیابی مدل روی داده‌های تست...")
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print("\n" + "="*45)
print(f"🎯 دقت نهایی مدل ResNet-18 روی داده‌های تست: {accuracy:.2f}%")
print("="*45)

# -------------------------------------------------------------
# ۸. ذخیره مدل آموزش دیده
# -------------------------------------------------------------
torch.save(model.state_dict(), 'resnet18_cifar10.pth')
print("💾 وزن‌های مدل با موفقیت در فایل 'resnet18_cifar10.pth' ذخیره شد.")