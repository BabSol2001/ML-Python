import matplotlib
matplotlib.use('Agg')  # جلوگیری از خطای گرافیکی Tcl در VS Code

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ۱. ساخت داده نمونه
X, y = make_regression(n_samples=200, n_features=2, noise=25, random_state=42)

# ۲. ساخت و آموزش دو مدل
# XGBoost (رشد سطح به سطح / Level-wise)
xgb_model = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, verbosity=0)
xgb_model.fit(X, y)

# LightGBM (رشد برگ به برگ / Leaf-wise)
lgb_model = LGBMRegressor(n_estimators=50, max_depth=3, num_leaves=8, learning_rate=0.1, random_state=42, verbosity=-1)
lgb_model.fit(X, y)

# ۳. آماده‌سازی شبکه برای رسم مرزهای تصمیم
x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 250), np.linspace(x2_min, x2_max, 250))

Z_xgb = xgb_model.predict(np.c_[xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)
Z_lgb = lgb_model.predict(np.c_[xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)

# ۴. رسم و مقایسه در دو نمودار کنار هم
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# نمودار ۱: XGBoost
c1 = axes[0].contourf(xx1, xx2, Z_xgb, alpha=0.75, cmap='viridis')
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', alpha=0.8)
axes[0].set_title('XGBoost (Level-wise Growth)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
fig.colorbar(c1, ax=axes[0], label='Predicted Value (y)')

# نمودار ۲: LightGBM
c2 = axes[1].contourf(xx1, xx2, Z_lgb, alpha=0.75, cmap='viridis')
axes[1].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', alpha=0.8)
axes[1].set_title('LightGBM (Leaf-wise Growth)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Feature 1')
fig.colorbar(c2, ax=axes[1], label='Predicted Value (y)')

plt.tight_layout()
output_file = 'xgb_vs_lgbm_boundaries.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ تصویر با موفقیت ذخیره شد! فایل '{output_file}' را در فولدر پروژه باز کنید.")