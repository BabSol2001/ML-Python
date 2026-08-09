import matplotlib
matplotlib.use('Agg')  # جلوگیری از خطای گرافیکی در VS Code

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

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

# ۲. تقسیم داده و آموزش مدل
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# ۳. محاسبه احتمالات پیش‌بینی‌شده برای کلاس ۱ (بیمار)
y_probs = model.predict_proba(X_test)[:, 1]

# ۴. محاسبه نقاط منحنی ROC و مقدار AUC
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
auc_score = roc_auc_score(y_test, y_probs)

# ۵. پیدا کردن بهترین حد آستانه با استفاده از شاخص Youden's J
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]
best_fpr = fpr[best_idx]
best_tpr = tpr[best_idx]

print(f"📊 AUC Score: {auc_score:.4f}")
print(f"🎯 بهترین حد آستانه بهینه‌شده: {best_threshold:.4f}")
print(f"✅ در این آستانه -> TPR (Recall): {best_tpr:.2f} | FPR: {best_fpr:.2f}")

# ۶. رسم منحنی ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess (AUC = 0.5)')

# علامت‌گذاری نقطه بهینه روی منحنی
plt.scatter(best_fpr, best_tpr, color='red', s=100, zorder=5, 
            label=f'Optimal Threshold = {best_threshold:.2f}')

plt.xlabel('False Positive Rate (FPR)', fontsize=11)
plt.ylabel('True Positive Rate (TPR / Recall)', fontsize=11)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=12, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

output_file = 'roc_curve_analysis.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ تصویر منحنی ROC در فایل '{output_file}' ذخیره شد.")