import matplotlib
matplotlib.use('Agg')  # جلوگیری از خطای Tcl در VS Code

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor

# ۱. ساخت داده نمونه با ۲ ویژگی (برای اینکه بتوانیم روی صفحه ۲ بعدی رسم کنیم)
X, y = make_regression(n_samples=100, n_features=2, noise=15, random_state=42)

# ۲. ساخت و آموزش مدل درخت تصمیم
model = DecisionTreeRegressor(max_depth=3, random_state=42)
model.fit(X, y)

# ۳. ایجاد شبکه شبکه‌ای (Meshgrid) برای پوشش تمام صفحه نمودار
x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx1, xx2 = np.meshgrid(
    np.linspace(x1_min, x1_max, 200),
    np.linspace(x2_min, x2_max, 200)
)

# ۴. پیش‌بینی مقدار y برای تمام نقاط این صفحه شبکه‌ای
Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)

# ۵. رسم نمودار مرزهای تصمیم و داده‌ها
plt.figure(figsize=(10, 7))

# رسم مناطق رنگی (مرزهای تصمیم درختی)
contour = plt.contourf(xx1, xx2, Z, alpha=0.7, cmap='viridis')
plt.colorbar(contour, label='Predicted Value (y)')

# رسم نقاط واقعی داده‌ها
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', linewidth=1)

# تنظیمات ظاهری
plt.xlabel('Feature 1', fontsize=11)
plt.ylabel('Feature 2', fontsize=11)
plt.title('Decision Tree Boundaries (Max Depth = 3)', fontsize=13, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.3)

# ذخیره عکس خروجی
output_name = 'decision_tree_boundaries.png'
plt.savefig(output_name, dpi=300, bbox_inches='tight')
print(f"✅ مرزهای تصمیم با موفقیت رسم شد! عکس آن را در فایل '{output_name}' ببینید.")