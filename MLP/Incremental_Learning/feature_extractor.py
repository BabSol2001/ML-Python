"""
سناریو و کد روش دوم: استخراج ویژگی + طبقه‌بندی‌کننده آنلاین (Feature Extractor + SVM/KNN)
سناریو: سیستم بدون آموزش مجدد شبکه عصبی، عکس‌ها را به بردارهای ۵۱۲ تایی تبدیل کرده و یک مدل SGDClassifier (سایکیت-لرن) را به صورت آنلاین به‌روزرسانی می‌کند.

"""

import os
import torch
import joblib
import numpy as np
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.linear_model import SGDClassifier

def extract_features_and_train_sgd(data_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # بررسی وجود پوشه داده‌ها
    if not os.path.exists(data_dir):
        print(f"❌ پوشه داده‌ها پیدا نشد: {os.path.abspath(data_dir)}")
        print("💡 لطفاً از وجود پوشه dataset در مسیر مطمئن شوید.")
        return

    # ۱. شبکه ResNet18 به عنوان Feature Extractor فاقد لایه FC
    base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    feature_extractor = torch.nn.Sequential(*list(base_model.children())[:-1])
    feature_extractor.eval().to(device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    X_features = []
    y_labels = []
    
    print("🔍 در حال استخراج بردارهای ویژگی ۵۱۲ تایی توسط ResNet18...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            feats = feature_extractor(inputs) # خروجی: [batch, 512, 1, 1]
            feats = torch.flatten(feats, 1)    # خروجی: [batch, 512]
            
            X_features.append(feats.cpu().numpy())
            y_labels.append(labels.numpy())
            
    if not X_features:
        print("❌ هیچ داده‌ای برای استخراج ویژگی یافت نشد!")
        return

    X_features = np.concatenate(X_features, axis=0)
    y_labels = np.concatenate(y_labels, axis=0)
    
    # ۲. آموزش طبقه‌بندی‌کننده آنلاین (Incremental Classifier)
    clf = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3) # پشتیبانی از partial_fit
    
    # آموزش روی داده‌های جدید
    clf.partial_fit(X_features, y_labels, classes=np.unique(y_labels))
    
    # ۳. ذخیره مدل کلاسیفایر
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "incremental_svm.pkl")
    joblib.dump({'classifier': clf, 'classes': dataset.classes}, output_path)
    print(f"✅ مدل SVM آنلاین با موفقیت در {os.path.basename(output_path)} ذخیره شد.")

if __name__ == "__main__":
    # محاسبه مسیر مطلق پوشه dataset در پوشه بالاتر (MLP/dataset)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.abspath(os.path.join(current_dir, "..", "dataset"))
    
    extract_features_and_train_sgd(dataset_path)