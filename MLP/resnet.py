import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms

# ۱. انتخاب پردازنده (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------------------------
# ۲. بارگذاری مدل آماده ResNet-18 با وزن‌های ImageNet
# -------------------------------------------------------------
# در نسخه‌های جدید PyTorch از weights استفاده می‌شود
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# -------------------------------------------------------------
# ۳. انجماد (Freeze) تمام لایه‌های کانولوشنی
# -------------------------------------------------------------
for param in model.parameters():
    param.requires_grad = False  # مانع محاسبه گرادیان و تغییر وزن‌ها می‌شود

# -------------------------------------------------------------
# ۴. جایگزینی لایه خروجی (Fully Connected Layer)
# -------------------------------------------------------------
# لایه خروجی اصلی ResNet18 ۱۰۰۰ کلاس خروجی دارد (برای ImageNet)
num_ftrs = model.fc.in_features  # تعداد نورون‌های ورودی لایه آخر (۵۱۲ نورون)

# لایه fc را با یک لایه جدید متناسب با پروژه خودمان (مثلاً ۳ کلاس) تعویض می‌کنیم
# لایه جدید به صورت خودکار requires_grad=True دارد
model.fc = nn.Linear(num_ftrs, 3) 

# انتقال کل مدل به GPU
model = model.to(device)

print("✅ معماری مدل ResNet با موفقیت برای ۳ کلاس آماده شد.")