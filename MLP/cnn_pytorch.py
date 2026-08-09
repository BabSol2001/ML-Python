import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# انتخاب پردازنده
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ۱. آماده‌سازی داده‌ها
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# ۲. تعریف معماری CNN
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        
        # لایه کانولوشن اول: input=(1, 28, 28) -> output=(16, 28, 28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # کاهش ابعاد: output=(16, 14, 14)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # لایه کانولوشن دوم: input=(16, 14, 14) -> output=(32, 14, 14)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # کاهش ابعاد مجدد: output=(32, 7, 7)
        
        # لایه‌های متصل متراکم (Fully Connected)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # کانولوشن ۱ + ReLU + Pooling
        x = self.pool(self.relu(self.conv1(x)))
        
        # کانولوشن ۲ + ReLU + Pooling
        x = self.pool(self.relu(self.conv2(x)))
        
        # Flatten: تبدیل به بردار یک بعدی برای لایه‌های FC
        x = x.view(x.size(0), -1)
        
        # لایه‌های خطی
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ۳. ساخت مدل و تنظیمات آموزش
model = FashionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ۴. آموزش کوتاه مدل (برای نمونه: ۲ Epoch)
print("🚀 شروع آموزش شبکه CNN...")
for epoch in range(2):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"Epoch {epoch+1} | Loss: {running_loss/len(train_loader):.4f}")

# ۵. ارزیابی دقت مدل روی داده‌های تست
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\n🎯 دقت مدل CNN روی داده‌های تست: {100 * correct / total:.2f}%")