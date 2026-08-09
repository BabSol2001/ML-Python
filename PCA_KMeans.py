import matplotlib
matplotlib.use('Agg')  # جلوگیری از خطای محیط گرافیکی در VS Code

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# -------------------------------------------------------------
# ۱. ساخت داده‌های ساختگی با ۱۵ ویژگی (ابعاد بالا)
# -------------------------------------------------------------
X_raw, _ = make_classification(
    n_samples=800, 
    n_features=15, 
    n_informative=10, 
    n_redundant=3,
    n_clusters_per_class=2,
    random_state=42
)

# -------------------------------------------------------------
# ۲. استانداردسازی داده‌ها (گام حیاتی پیش از PCA و K-Means)
# -------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# -------------------------------------------------------------
# ۳. کاهش ابعاد به ۲ بعد با استفاده از PCA
# -------------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_
total_variance = np.sum(explained_variance) * 100

print(f"📊 واریانس حفظ شده توسط دو مؤلفه اصلی: {total_variance:.2f}%")

# -------------------------------------------------------------
# ۴. خوشه‌بندی داده‌های فشرده‌شده با K-Means
# -------------------------------------------------------------
# فرض می‌کنیم می‌خواهیم داده‌ها را به ۴ خوشه تقسیم کنیم
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_pca)
centroids = kmeans.cluster_centers_

# -------------------------------------------------------------
# ۵. رسم و ذخیره نمودار ۲ بعدی خوشه‌ها
# -------------------------------------------------------------
plt.figure(figsize=(10, 7))

# رسم نقاط داده‌ها با رنگ‌های متفاوت برای هر خوشه
scatter = plt.scatter(
    X_pca[:, 0], 
    X_pca[:, 1], 
    c=cluster_labels, 
    cmap='tab10', 
    alpha=0.7, 
    edgecolors='k', 
    s=50
)

# رسم مراکز خوشه‌ها (Centroids) با علامت X بزرگ قرمز
plt.scatter(
    centroids[:, 0], 
    centroids[:, 1], 
    s=300, 
    c='red', 
    marker='X', 
    edgecolors='black', 
    linewidths=2,
    label='Centroids (PCA Space)'
)

plt.title(
    f'PCA + K-Means Clustering (15 Features Reduced to 2D)\nRetained Variance: {total_variance:.1f}%', 
    fontsize=12, 
    fontweight='bold'
)
plt.xlabel(f'Principal Component 1 (PC1) - {explained_variance[0]*100:.1f}% Var', fontsize=11)
plt.ylabel(f'Principal Component 2 (PC2) - {explained_variance[1]*100:.1f}% Var', fontsize=11)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

output_file = 'pca_kmeans_combined.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ نمودار ترکیب PCA و K-Means در فایل '{output_file}' ذخیره شد.")