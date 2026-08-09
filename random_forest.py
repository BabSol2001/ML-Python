import matplotlib
matplotlib.use('Agg') # جلوگیری از خطای Tcl

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# ۱. ساخت داده نمونه با نویز
X, y = make_regression(n_samples=100, n_features=2, noise=20, random_state=42)

# ۲. آموزش یک درخت تصمیم منفرد vs یک جنگل تصادفی
single_tree = DecisionTreeRegressor(max_depth=5, random_state=42)
random_forest = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

single_tree.fit(X, y)
random_forest.fit(X, y)

# ۳. شبکه شبکه‌ای برای رسم مرزها
x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 200), np.linspace(x2_min, x2_max, 200))

Z_tree = single_tree.predict(np.c_[xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)
Z_forest = random_forest.predict(np.c_[xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)

# ۴. رسم و مقایسه
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# نمودار ۱: تک درخت
axes[0].contourf(xx1, xx2, Z_tree, alpha=0.7, cmap='viridis')
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
axes[0].set_title('Single Decision Tree (Sharp Boundaries)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')

# نمودار ۲: جنگل تصادفی
axes[1].contourf(xx1, xx2, Z_forest, alpha=0.7, cmap='viridis')
axes[1].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
axes[1].set_title('Random Forest - 100 Trees (Smooth Boundaries)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Feature 1')

plt.tight_layout()
output_file = 'random_forest_vs_single_tree.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ خروجی ذخیره شد! تصویر '{output_file}' را بررسی کنید.")