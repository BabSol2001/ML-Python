import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

def train_lwf(data_dir: str, epochs: int = 3, alpha_distill: float = 2.0, temperature: float = 2.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # بررسی وجود پوشه داده‌ها
    if not os.path.exists(data_dir):
        print(f"❌ پوشه داده‌ها پیدا نشد: {os.path.abspath(data_dir)}")
        return

    # ۱. مدل معلم (ثابت، فریز شده و ۱۰۰۰ کلاسه)
    teacher_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
        
    # ۲. مدل دانش‌آموز (قابل آموزش با مجموع ۱۰۰۳ کلاس)
    student_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = student_model.fc.in_features
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    num_old_classes = 1000
    num_new_classes = len(dataset.classes) # ۳ کلاس
    total_classes = num_old_classes + num_new_classes # ۱۰۰۳ کلاس
    
    # تغییر لایه آخر به ۱۰-۳ کلاس
    student_model.fc = nn.Linear(num_ftrs, total_classes)
    student_model = student_model.to(device)
    
    # بهینه‌ساز برای تمام پارامترهای قابل آموزش دانش‌آموز
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.0001)
    criterion_ce = nn.CrossEntropyLoss()
    
    print(f"🚀 شروع آموزش LwF واقعی (Teacher: {num_old_classes} | Student: {total_classes})...")
    student_model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # شیفت دادن لیبل‌ها به اندیس‌های ۱۰۰۰، ۱۰۰۱ و ۱۰۰۲
            shifted_labels = labels + num_old_classes
            
            # ۱. پیش‌بینی مدل معلم (فقط روی ۱۰۰۰ کلاس قدیمی)
            with torch.no_grad():
                teacher_logits = teacher_model(inputs)
                
            optimizer.zero_grad()
            student_logits = student_model(inputs)
            
            # ۲. زیان وظیفه جدید (Cross Entropy Loss برای ۳ کلاس جدید)
            loss_new = criterion_ce(student_logits, shifted_labels)
            
            # ۳. زیان تقطیر دانش (Distillation Loss با KL-Divergence)
            # جدا کردن ۱۰۰0 خروجی اول دانش‌آموز که مربوط به دانش معلم است
            student_old_logits = student_logits[:, :num_old_classes]
            
            # نرم‌سازی احتمالات با پارامتر دما (Temperature Scaling)
            p_student = F.log_softmax(student_old_logits / temperature, dim=1)
            p_teacher = F.softmax(teacher_logits / temperature, dim=1)
            
            # محاسبه واگرایی KL بین توزیع احتمال معلم و دانش‌آموز
            loss_distill = F.kl_div(p_student, p_teacher, reduction='batchmean') * (temperature ** 2)
            
            # زیان کل
            total_loss = loss_new + (alpha_distill * loss_distill)
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(dataset) if len(dataset) > 0 else 0.0
        print(f"Epoch {epoch+1}/{epochs} - Total Loss: {epoch_loss:.4f}")

    # ذخیره مدل آموزش‌دیده
    checkpoint = {
        'model_state_dict': student_model.state_dict(),
        'new_classes': dataset.classes
    }
    torch.save(checkpoint, "lwf_model.pth")
    print("💾 مدل LwF با موفقیت در lwf_model.pth ذخیره شد.")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.abspath(os.path.join(current_dir, "..", "dataset"))
    train_lwf(dataset_path)