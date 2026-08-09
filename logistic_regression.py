import matplotlib
matplotlib.use('Agg')  # جلوگیری از خطای Tcl در VS Code

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# ۱. ساخت داده‌های نمونه برای طبقه‌بندی دوتایی (۰ و ۱)
X, y = make_classification(
    n_samples=100, n_features=2, n_redundant=0, 
    n_clusters_per_class=1, random_state=42
)

# ۲. ساخت و آموزش مدل رگرسیون لوجستیک
model = LogisticRegression()
model.fit(X, y)

# ۳. آماده‌سازی شبکه برای رسم مرز احتمال
x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 200), np.linspace(x2_min, x2_max, 200))

# پیش‌بینی احتمال کلاس ۱ برای تمام نقاط
Z = model.predict_proba(np.c_[xx1.ravel(), xx2.ravel()])[:, 1].reshape(xx1.shape)

# ۴. رسم نمودار مرز تصمیم و احتمالات
plt.figure(figsize=(9, 6))

# رسم طیف رنگی احتمالات (از ۰ تا ۱)
contour = plt.contourf(xx1, xx2, Z, alpha=0.8, cmap='coolwarm')
plt.colorbar(contour, label='Probability of Class 1')

# رسم خط مرز تصمیم (جایی که احتمال دقیقاً ۰.۵ است)
plt.contour(xx1, xx2, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')

# رسم نقاط واقعی (کلاس ۰ و ۱)
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k', linewidth=1)

plt.xlabel('Feature 1', fontsize=11)
plt.ylabel('Feature 2', fontsize=11)
plt.title('Logistic Regression Decision Boundary & Probabilities', fontsize=12, fontweight='bold')

output_file = 'logistic_regression_boundary.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ خروجی طبقه‌بندی ذخیره شد! تصویر '{output_file}' را بررسی کنید.")