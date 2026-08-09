import matplotlib
matplotlib.use('Agg')  # جلوگیری از خطای محیط گرافیکی در VS Code

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------
# ۱. ساخت داده‌های پیچیده (دو هلال + داده‌های پرت/نویز)
# -------------------------------------------------------------
X_moons, y_moons = make_moons(n_samples=500, noise=0.08, random_state=42)

# اضافه کردن ۲۰ نقطه نویز تصادفی پرت
np.random.seed(42)
outliers = np.random.uniform(low=-1.5, high=2.5, size=(20, 2))
X = np.vstack([X_moons, outliers])

# ۲. گام بسیار حیاتی: مقیاس‌بندی داده‌ها
X_scaled = StandardScaler().fit_transform(X)

# -------------------------------------------------------------
# ۳. اجرای K-Means (با K=2)
# -------------------------------------------------------------
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)

# -------------------------------------------------------------
# ۴. اجرای DBSCAN
# -------------------------------------------------------------
# eps: شعاع همسایگی | min_samples: حداقل تعداد همسایه
dbscan = DBSCAN(eps=0.25, min_samples=5)
y_dbscan = dbscan.fit_predict(X_scaled)

# -------------------------------------------------------------
# ۵. رسم و ذخیره نمودار مقایسه‌ای
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# نمودار ۱: خروجی K-Means
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, cmap='coolwarm', alpha=0.7, edgecolors='k')
axes[0].set_title('K-Means Failure (Splits Moons & Forces Outliers)', fontsize=11, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# نمودار ۲: خروجی DBSCAN
# نقاط با برچسب 1- نویز هستند و با ضربدر مشکی رسم می‌شوند
is_noise = (y_dbscan == -1)
axes[1].scatter(X_scaled[~is_noise, 0], X_scaled[~is_noise, 1], c=y_dbscan[~is_noise], cmap='viridis', alpha=0.8, edgecolors='k')
axes[1].scatter(X_scaled[is_noise, 0], X_scaled[is_noise, 1], color='black', marker='X', s=70, label='Noise (Outliers)')

axes[1].set_title(f'DBSCAN Success (Detected Moons & {sum(is_noise)} Outliers)', fontsize=11, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
output_file = 'dbscan_vs_kmeans.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ نمودار مقایسه DBSCAN و K-Means در فایل '{output_file}' ذخیره شد.")