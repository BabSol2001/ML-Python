import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path
from sklearn.datasets import make_regression
import mplcursors  # کتابخانه ایجاد Tooltip و کلیک تعاملی

# ۱. ساخت داده نمونه
X, y = make_regression(n_samples=100, n_features=20, n_informative=5, noise=10, random_state=42)

# ۲. محاسبه مسیر ضرایب
alphas_lasso, coefs_lasso, _ = lasso_path(X, y)

# ۳. تنظیم کادر نمودار (کمی عریض‌تر برای جاگیری راهنما)
plt.figure(figsize=(12, 6))

# ۴. رسم هر خط همراه با لیبل دقیق برای Legend
for i in range(coefs_lasso.shape[0]):
    # مقدار اولیه ضریب در سمت راست نمودار (کمترین لاندا = بالاترین ضریب)
    initial_coef = coefs_lasso[i][-1]
    
    # لیبل شامل شماره ویژگی و مقدار ضریب
    label_text = f'Feature {i+1} (w = {initial_coef:.1f})'
    
    plt.plot(np.log10(alphas_lasso), coefs_lasso[i], label=label_text, linewidth=1.8)

# ۵. تنظیمات محورها و عنوان
plt.xlabel('log10(Lambda / Alpha)', fontsize=11)
plt.ylabel('Coefficients', fontsize=11)
plt.title('Lasso Path Plot with Coefficients Legend', fontsize=13, fontweight='bold')
plt.grid(True)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.gca().invert_xaxis()

# =====================================================================
# ۶. اضافه کردن Legend کامل و مرتب در خارج از کادر راست
# =====================================================================
plt.legend(
    bbox_to_anchor=(1.02, 1), # انتقال به بیرون از کادر سمت راست
    loc='upper left',          # چسباندن از بالا سمت چپ
    borderaxespad=0,           # بدون فاصله اضافی
    ncols=2,                   # مرتب‌سازی در ۲ ستون مجزا
    fontsize='small'           # فونت خوانا و مرتب
)

# تنظیم فاصله حاشیه‌ها جهت جلوگیری از افتادن Legend به بیرون صفحه
plt.tight_layout()

# =====================================================================
#  تولتیپ هوشمند و سازگار با حالت زوم (Zoom-friendly Tooltip)
# =====================================================================
cursor = mplcursors.cursor(multiple=True) # multiple=True اجازه می‌دهد چند تولتیپ همزمان در زوم داشته باشی

@cursor.connect("add")
def on_add(sel):
    # تنظیم متن تولتیپ
    sel.annotation.set_text(
        f"{sel.artist.get_label()}\nCoeff: {sel.target[1]:.2f}\nlog10(λ): {sel.target[0]:.2f}"
    )
    # تنظیم کادر گرافیکی شفاف‌تر برای ماندگاری در زوم
    sel.annotation.get_bbox_patch().set(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8)

# نمایش پنجره پویا
plt.show()