import matplotlib
matplotlib.use('Agg')  # جلوگیری از خطای Tcl در VS Code

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# تابع کمکی برای رسم مرزهای تصمیم‌گیری
def plot_decision_boundary(ax, model, X, y, title, color_map='viridis'):
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 200), np.linspace(x2_min, x2_max, 200))
    
    Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)
    
    contour = ax.contourf(xx1, xx2, Z, alpha=0.75, cmap=color_map)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=color_map, edgecolor='k', alpha=0.8)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    return contour

# ساخت شکل کلی ۴ تایی
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# -------------------------------------------------------------
# سناریو ۱: نیاز به شفافیت و تفسیرپذیری خطی بالا -> Lasso
# -------------------------------------------------------------
X1, y1 = make_regression(n_samples=100, n_features=2, noise=15, random_state=42)
lasso = Lasso(alpha=1.0)
lasso.fit(X1, y1)
plot_decision_boundary(
    axes[0, 0], lasso, X1, y1, 
    "1. Interpretability First -> Lasso\n(Linear & Smooth Boundaries)"
)

# -------------------------------------------------------------
# سناریو ۲: داده کم و نیاز به مدل قوی بدون تنظیمات زیاد -> Random Forest
# -------------------------------------------------------------
X2, y2 = make_regression(n_samples=100, n_features=2, noise=20, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X2, y2)
plot_decision_boundary(
    axes[0, 1], rf, X2, y2, 
    "2. Small Data & No Tuning Needed -> Random Forest\n(Robust & Ensemble Average)"
)

# -------------------------------------------------------------
# سناریو ۳: بیشترین دقت ممکن روی داده متوسط -> XGBoost
# -------------------------------------------------------------
X3, y3 = make_regression(n_samples=300, n_features=2, noise=15, random_state=42)
xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, verbosity=0)
xgb.fit(X3, y3)
plot_decision_boundary(
    axes[1, 0], xgb, X3, y3, 
    "3. Maximum Accuracy on Tabular Data -> XGBoost\n(Level-wise Precise Fitting)"
)

# -------------------------------------------------------------
# سناریو ۴: داده‌های حجیم و نیاز به سرعت بالا -> LightGBM
# -------------------------------------------------------------
X4, y4 = make_regression(n_samples=1000, n_features=2, noise=10, random_state=42)
lgbm = LGBMRegressor(n_estimators=100, num_leaves=15, learning_rate=0.1, random_state=42, verbosity=-1)
lgbm.fit(X4, y4)
plot_decision_boundary(
    axes[1, 1], lgbm, X4, y4, 
    "4. Big Data & High Speed -> LightGBM\n(Leaf-wise Fast Splitting)"
)

# تنظیمات نهایی و ذخیره
plt.tight_layout()
output_file = 'decision_guide_models.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ نمودار چهارگانه سناریوها با موفقیت ذخیره شد! تصویر '{output_file}' را در پوشه پروژه ببینید.")