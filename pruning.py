import matplotlib
matplotlib.use('Agg')  # جلوگیری از خطای Tcl

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor

# ۱. ساخت داده نمونه با اندکی نویز
X, y = make_regression(n_samples=100, n_features=2, noise=20, random_state=42)

# ۲. ساخت دو مدل: یکی بیش‌برازش‌شده (بدون محدودیت) و یکی هرس‌شده (با محدودیت عمق)
tree_overfit = DecisionTreeRegressor(max_depth=None, random_state=42)  # رشد بی‌رویه
tree_pruned = DecisionTreeRegressor(max_depth=3, random_state=42)      # هرس‌شده

tree_overfit.fit(X, y)
tree_pruned.fit(X, y)

# ۳. آماده‌سازی شبکه برای رسم مرزها
x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 200), np.linspace(x2_min, x2_max, 200))

# ۴. پیش‌بینی برای هر دو مدل
Z_overfit = tree_overfit.predict(np.c_[xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)
Z_pruned = tree_pruned.predict(np.c_[xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)

# ۵. رسم دو نمودار کنار هم جهت مقایسه
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# نمودار ۱: بیش‌برازش (Overfit)
axes[0].contourf(xx1, xx2, Z_overfit, alpha=0.7, cmap='viridis')
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
axes[0].set_title('Overfitted Tree (max_depth=None)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')

# نمودار ۲: هرس‌شده (Pruned)
axes[1].contourf(xx1, xx2, Z_pruned, alpha=0.7, cmap='viridis')
axes[1].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
axes[1].set_title('Pruned Tree (max_depth=3)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Feature 1')

plt.tight_layout()
output_file = 'tree_overfitting_vs_pruned.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ مقایسه با موفقیت ذخیره شد! تصویر '{output_file}' را باز کنید.")