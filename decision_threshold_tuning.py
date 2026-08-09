import matplotlib
matplotlib.use('Agg')  # جلوگیری از خطای گرافیکی در VS Code

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# ۱. ساخت داده نمونه غیرمتوازن
X, y = make_classification(
    n_samples=500, 
    n_features=2, 
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    weights=[0.85, 0.15], 
    random_state=42
)

# ۲. تقسیم به داده آموزش و تست (تعریف X_test و y_test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ۳. آموزش مدل رگرسیون لوجستیک (تعریف متغیر model)
model = LogisticRegression()
model.fit(X_train, y_train)

# ۴. محاسبه احتمال پیش‌بینی‌شده برای کلاس ۱ (بیمار)
y_probs = model.predict_proba(X_test)[:, 1]

# ۵. اعمال حد آستانه سفارشی (۰.۲۵ به جای ۰.۵)
custom_threshold = 0.25
y_pred_new = (y_probs >= custom_threshold).astype(int)

# ۶. چاپ گزارش ارزیابی جدید
print(f"=== گزارش ارزیابی با حد آستانه {custom_threshold} ===")
print(classification_report(y_test, y_pred_new, target_names=['Healthy (0)', 'Sick (1)']))

# ۷. رسم ماتریس درهم‌ریختگی جدید با Seaborn
cm_new = confusion_matrix(y_test, y_pred_new)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_new, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy (0)', 'Sick (1)'],
            yticklabels=['Healthy (0)', 'Sick (1)'])

plt.xlabel('Predicted label', fontsize=11)
plt.ylabel('True label', fontsize=11)
plt.title(f'Confusion Matrix (Threshold = {custom_threshold})', fontsize=12, fontweight='bold')
plt.tight_layout()

output_file = 'confusion_matrix_threshold_25.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ تصویر با موفقیت در '{output_file}' ذخیره شد!")