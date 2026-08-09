import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# تنظیم Seed برای تکرارپذیری نتایج
torch.manual_seed(42)

# -------------------------------------------------------------
# ۱. ساخت یک شبکه عصبی MLP با استفاده از torch.nn.Module
# -------------------------------------------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        
        # لایه اول: ورودی ➔ لایه پنهان
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # تابع فعال‌سازی غیرخطی
        self.relu = nn.ReLU()
        # لایه دوم: لایه پنهان ➔ خروجی
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # انتشار به جلو (Forward Pass)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# -------------------------------------------------------------
# ۲. ساخت داده‌های ساختگی (طبقه بندی ۲ کلاسی)
# -------------------------------------------------------------
# ۱۰۰ داده، هر داده دارای ۴ ویژگی
X = torch.randn(100, 4)
# لیبل‌ها (۰ یا ۱)
y = torch.randint(0, 2, (100,)).float().unsqueeze(1)

# -------------------------------------------------------------
# ۳. مقداردهی اولیه مدل، تابع زیان و بهینه‌ساز (Optimizer)
# -------------------------------------------------------------
model = SimpleMLP(input_dim=4, hidden_dim=8, output_dim=1)

# تابع زیان انتروپی متقاطع متقاطع دوتایی با لایه Sigmoid ادغام شده
criterion = nn.BCEWithLogitsLoss()

# بهینه‌ساز Adam با نرخ یادگیری (Learning Rate) 0.01
optimizer = optim.Adam(model.parameters(), lr=0.01)

# -------------------------------------------------------------
# ۴. حلقه آموزش (Training Loop)
# -------------------------------------------------------------
epochs = 100
loss_history = []

print("🚀 شروع آموزش شبکه عصبی در PyTorch...")

for epoch in range(epochs):
    # ۱. صفر کردن گرادیان‌های قبلی
    optimizer.zero_grad()
    
    # ۲. انتشار به جلو
    outputs = model(X)
    
    # ۳. محاسبه خطا
    loss = criterion(outputs, y)
    
    # ۴. انتشار به عقب (محاسبه گرادیان‌ها)
    loss.backward()
    
    # ۵. به‌روزرسانی وزن‌ها
    optimizer.step()
    
    # ثبت مقدار خطا
    loss_history.append(loss.item())
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# -------------------------------------------------------------
# ۵. رسم و ذخیره‌سازی نمودار کاهش خطا (Loss Curve)
# -------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), loss_history, color='crimson', linewidth=2)
plt.title('Training Loss Curve (PyTorch MLP)', fontsize=12, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

output_file = 'pytorch_mlp_loss.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ نمودار افت خطا با موفقیت در '{output_file}' ذخیره شد.")