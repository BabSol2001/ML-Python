import matplotlib
matplotlib.use('Agg')  # جلوگیری از خطای Tcl

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor

# ۱. ساخت داده نمونه
X, y = make_regression(n_samples=100, n_features=2, noise=15, random_state=42)

# ۲. ساخت و آموزش مدل Gradient Boosting
# n_estimators: تعداد درخت‌های متوالی
# learning_rate: نرخ ضریب تاثیر هر درخت جدید
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbr.fit(X, y)

# ۳. آماده‌سازی شبکه برای رسم مرزهای تصمیم
x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 200), np.linspace(x2_min, x2_max, 200))

Z = gbr.predict(np.c_[xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)

# ۴. رسم نمودار
plt.figure(figsize=(9, 6))
contour = plt.contourf(xx1, xx2, Z, alpha=0.7, cmap='viridis')
plt.colorbar(contour, label='Predicted Value (y)')

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.xlabel('Feature 1', fontsize=11)
plt.ylabel('Feature 2', fontsize=11)
plt.title('Gradient Boosting Regressor Boundaries (100 Sequential Trees)', fontsize=12, fontweight='bold')

output_file = 'gradient_boosting_boundaries.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ خروجی گرادیان بوستینگ ذخیره شد! تصویر '{output_file}' را بررسی کنید.")