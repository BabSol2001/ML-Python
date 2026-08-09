import matplotlib
matplotlib.use('Agg')  # جلوگیری از خطای Tcl در VS Code

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# ۱. ساخت داده نمونه غیرمتوازن با تنظیم درست پارامترهای ویژگی
X, y = make_classification(
    n_samples=500, 
    n_features=2, 
    n_informative=2,   # اضافه شد
    n_redundant=0,     # اضافه شد
    n_clusters_per_class=1, 
    weights=[0.85, 0.15], 
    random_state=42
)

# ۲. تقسیم به داده آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ۳. آموزش مدل رگرسیون لوجستیک
model = LogisticRegression()
model.fit(X_train, y_train)

# ۴. پیش‌بینی روی داده‌های تست
y_pred = model.predict(X_test)

# ۵. چاپ گزارش کامل معیارهای ارزیابی (Classification Report)
print("📊 گزارش کامل معیارهای ارزیابی:")
print("-" * 55)
print(classification_report(y_test, y_pred, target_names=['کلاس ۰ (سالم)', 'کلاس ۱ (بیمار)']))

# ۶. رسم و ذخیره ماتریس درهم‌ریختگی (Confusion Matrix)
fig, ax = plt.subplots(figsize=(7, 6))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy (0)', 'Sick (1)'])
disp.plot(cmap='Blues', ax=ax)

plt.title('Confusion Matrix Visual Representation', fontsize=12, fontweight='bold')
output_file = 'confusion_matrix_demo.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ تصویر ماتریس درهم‌ریختگی در فایل '{output_file}' ذخیره شد.")