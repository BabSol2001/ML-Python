import matplotlib
matplotlib.use('Agg')  # جلوگیری از خطای محیط گرافیکی در VS Code

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ۱. ساخت داده‌های ساختگی با ۱۰ ویژگی (ابعاد بالا)
X_raw, _ = make_classification(
    n_samples=500, 
    n_features=10, 
    n_informative=7, 
    random_state=42
)

# ۲. گام بسیار حیاتی: مقیاس‌بندی داده‌ها (Standardization)
# PCA شدیداً به مقیاس متغیرها حساس است!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# ۳. اجرای PCA کامل (بدون کاهش بعد، برای بررسی میزان واریانس تمام ۱۰ مؤلفه)
pca_full = PCA()
pca_full.fit(X_scaled)

# محاسبه واریانس تجمعی (Cumulative Explained Variance)
explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# ۴. اجرا و اعمال PCA برای کاهش ابعاد به ۲ بعد (جهت تجسم تصویری)
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

# -------------------------------------------------------------
# ۵. رسم و ذخیره نمودارها
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# نمودار ۱: میزان واریانس تجمعی (چند بعد کافی است؟)
axes[0].plot(range(1, 11), cumulative_variance, marker='o', color='purple', linestyle='-')
axes[0].axhline(y=0.85, color='red', linestyle='--', label='85% Variance Threshold')
axes[0].set_title('Cumulative Explained Variance by PCA Components', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Number of Principal Components')
axes[0].set_ylabel('Cumulative Explained Variance')
axes[0].set_xticks(range(1, 11))
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# نمودار ۲: نمایش داده‌ها روی ۲ مؤلفه اصلی اول (PC1 و PC2)
axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c='teal', alpha=0.6, edgecolors='k')
axes[1].set_title(f'Data Projected to 2D (Retains {cumulative_variance[1]*100:.1f}% Info)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Principal Component 1 (PC1)')
axes[1].set_ylabel('Principal Component 2 (PC2)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
output_file = 'pca_analysis_demo.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ تحلیل PCA با موفقیت در فایل '{output_file}' ذخیره شد.")