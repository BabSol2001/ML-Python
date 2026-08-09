import matplotlib
matplotlib.use('Agg')  # جلوگیری از خطای گرافیکی Tcl

import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor, plot_tree

# ۱. ساخت داده نمونه با ۲ ویژگی برای سادگی
X, y = make_regression(n_samples=50, n_features=2, noise=10, random_state=42)

# ۲. ساخت و آموزش مدل درخت تصمیم
# نکته: max_depth=3 عمق درخت را کنترل می‌کند تا بیش از حد پیچیده نشود
model = DecisionTreeRegressor(max_depth=3, random_state=42)
model.fit(X, y)

# ۳. تنظیم اندازه و رسم تصویر درخت
plt.figure(figsize=(14, 8))
plot_tree(
    model, 
    feature_names=['Feature 1', 'Feature 2'], 
    filled=True,      # رنگی کردن گره‌ها بر اساس مقدار پیش‌بینی
    rounded=True,     # گرد کردن گوشه‌های کادرها
    fontsize=9
)

plt.title("Decision Tree Structure (Max Depth = 3)", fontsize=14, fontweight='bold')

# ۴. ذخیره عکس خروجی در فولدر پروژه
output_file = 'decision_tree_structure.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ درخت تصمیم با موفقیت رسم شد! تصویر آن در فایل '{output_file}' ذخیره گردید.")