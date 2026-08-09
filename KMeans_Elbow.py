import matplotlib
matplotlib.use('Agg')  # جلوگیری از خطای محیط گرافیکی در VS Code

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# ۱. ساخت داده‌های ساختگی ۲ بعدی بدون لیبل
X, _ = make_blobs(n_samples=600, centers=4, cluster_std=0.8, random_state=42)

# ۲. اجرای روش Elbow برای پیدا کردن بهترین K (از ۱ تا ۱۰)
inertia_values = []
k_range = range(1, 10)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

# ۳. آموزش مدل نهایی با K=4 (مقدار بهینه)
optimal_k = 4
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
y_clusters = kmeans_final.fit_predict(X)
centroids = kmeans_final.cluster_centers_

# -------------------------------------------------------------
# ۴. رسم و ذخیره نمودارها
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# نمودار ۱: روش Elbow
axes[0].plot(k_range, inertia_values, marker='o', color='purple', linestyle='--')
axes[0].axvline(x=optimal_k, color='red', linestyle=':', label=f'Optimal K = {optimal_k}')
axes[0].set_title('Elbow Method For Optimal K', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia (Within-Cluster Sum of Squares)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# نمودار ۲: نتیجه خوشه‌بندی K-Means
scatter = axes[1].scatter(X[:, 0], X[:, 1], c=y_clusters, cmap='viridis', alpha=0.6, edgecolors='k')
# رسم مراکز خوشه‌ها (ضربدرهای قرمز)
axes[1].scatter(centroids[:, 0], centroids[:, 1], s=250, c='red', marker='X', label='Centroids', edgecolor='black')
axes[1].set_title(f'K-Means Clustering Result (K = {optimal_k})', fontsize=12, fontweight='bold')
axes[1].legend()

plt.tight_layout()
output_file = 'kmeans_clustering_demo.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ نمودار خوشه‌بندی با موفقیت در فایل '{output_file}' ذخیره شد.")