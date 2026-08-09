import matplotlib
matplotlib.use('Agg') # جلوگیری از خطای محیط گرافیکی در VS Code

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# تنظیم Seed برای تکرارپذیری نتایج
torch.manual_seed(42)

# انتخاب پردازنده (GPU در صورت وجود، در غیر این صورت CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"⚡ اجرای محاسبات روی: {device}")

# -------------------------------------------------------------
# ۱. آماده‌سازی تبدیل‌ها (Transforms) و بارگذاری Dataset & DataLoader
# -------------------------------------------------------------
# تبدیل تصاویر به Tensor و نرمال‌سازی پیکسل‌ها بین ۰ تا ۱
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # مقادیر پیکسل‌ها را به بازه [-1, 1] می‌برد
])

# دانلود و بارگذاری داده‌های آموزشی و تست
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# ساخت DataLoader با سایز Batch مشخص (مثلاً ۶۴)
BATCH_SIZE = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"📊 تعداد نمونه‌های آموزشی: {len(train_dataset)}")
print(f"📊 تعداد نمونه‌های تست: {len(test_dataset)}")
print(f"📦 تعداد Batchهای آموزشی: {len(train_loader)}")

# -------------------------------------------------------------
# ۲. تعریف معماری شبکه عصبی MLP برای تصویر
# -------------------------------------------------------------
class FashionMLP(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=128, output_dim=10):
        super(FashionMLP, self).__init__()
        
        # لایه اول: ورودی ۷۸۴ (۲۸x۲۸) ➔ ۱۲۸ نورون
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        # لایه دوم: ۱۲۸ نورون ➔ ۱۰ کلاس خروجی (لباس، کفش، کیف و...)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Flatten: تبدیل ابعاد تصویر از [Batch, 1, 28, 28] به [Batch, 784]
        x = x.view(x.size(0), -1)
        
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# -------------------------------------------------------------
# ۳. مقداردهی اولیه مدل، تابع زیان و بهینه‌ساز
# -------------------------------------------------------------
model = FashionMLP(input_dim=28*28, hidden_dim=128, output_dim=10).to(device)

# CrossEntropyLoss خودش Softmax را به صورت داخلی اعمال می‌کند
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------------------------------------
# ۴. حلقه آموزش (Training Loop) روی Batchها
# -------------------------------------------------------------
EPOCHS = 5
train_losses = []

print("\n🚀 شروع آموزش مدل روی Fashion-MNIST...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        # انتقال داده‌ها به GPU یا CPU
        images, labels = images.to(device), labels.to(device)
        
        # ۱. صفر کردن گرادیان‌ها
        optimizer.zero_grad()
        
        # ۲. انتشار به جلو
        outputs = model(images)
        
        # ۳. محاسبه خطا
        loss = criterion(outputs, labels)
        
        # ۴. انتشار به عقب
        loss.backward()
        
        # ۵. به‌روزرسانی وزن‌ها
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {epoch_loss:.4f}")

# -------------------------------------------------------------
# ۵. ارزیابی مدل روی داده‌های تست (Test Accuracy)
# -------------------------------------------------------------
model.eval()
correct = 0
total = 0

# غیرفعال کردن محاسبه گرادیان برای ارزیابی (افزایش سرعت و کاهش مصرف حافظه)
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        
        # پیدا کردن کلاسی که بیشترین احتمال (Logit) را دارد
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print("\n" + "="*40)
print(f"🎯 دقت (Accuracy) مدل روی داده‌های تست: {accuracy:.2f}%")
print("="*40)

# -------------------------------------------------------------
# ۶. رسم نمودار افت خطا
# -------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, marker='o', color='royalblue', linewidth=2)
plt.title('Fashion-MNIST Training Loss (MLP)', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

output_file = 'fashion_mnist_mlp_loss.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ نمودار افت خطا در فایل '{output_file}' ذخیره شد.")