import matplotlib
matplotlib.use('Agg')  # جلوگیری از خطای محیط گرافیکی در VS Code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# -------------------------------------------------------------
# ۱. ساخت یک DataFrame خام واقعی (شامل ویژگی‌های عددی، متنی، NaN و نویز)
# -------------------------------------------------------------
np.random.seed(42)
n_samples = 600

# ساخت داده‌های پایه عددی با ۳ خوشه طبیعی
X_blobs, _ = make_blobs(n_samples=n_samples, n_features=6, centers=3, cluster_std=1.2, random_state=42)

# ساخت DataFrame و اضافه کردن ویژگی‌های متنی، NaN و داده‌های پرت
raw_data = pd.DataFrame(X_blobs, columns=[f'Num_Feature_{i+1}' for i in range(6)])

# اضافه کردن ویژگی‌های کاتگوریکال
raw_data['Category_A'] = np.random.choice(['Type_X', 'Type_Y', 'Type_Z'], size=n_samples)
raw_data['Category_B'] = np.random.choice(['Low', 'Medium', 'High'], size=n_samples)

# تزریق مقادیر مفقود (NaN) به صورت تصادفی (حدود ۵٪ داده‌ها)
for col in raw_data.columns:
    nan_mask = np.random.rand(n_samples) < 0.05
    raw_data.loc[nan_mask, col] = np.nan

# تزریق ۱۵ داده پرت/نویز شدید (Outliers)
outliers = np.random.uniform(low=-15, high=15, size=(15, 6))
outliers_df = pd.DataFrame(outliers, columns=[f'Num_Feature_{i+1}' for i in range(6)])
outliers_df['Category_A'] = 'Type_X'
outliers_df['Category_B'] = 'High'

# ترکیب داده‌های اصلی با نویزها
raw_data = pd.concat([raw_data, outliers_df], ignore_index=True)

print(f"📊 ابعاد مجموعه‌داده خام اولیه: {raw_data.shape}")
print(f"🔹 تعداد کل مقادیر مفقود (NaN): {raw_data.isna().sum().sum()}")

# -------------------------------------------------------------
# ۲. تعریف پایپ‌لاین پیش‌پردازش داده‌ها (ColumnTransformer)
# -------------------------------------------------------------
num_cols = [f'Num_Feature_{i+1}' for i in range(6)]
cat_cols = ['Category_A', 'Category_B']

# پایپ‌لاین عددی: پر کردن NaNها با میانه + مقیاس‌بندی با StandardScaler
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# پایپ‌لاین کاتگوریکال: پر کردن NaNها با مد + One-Hot Encoding
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
])

# ترکیب پیش‌پردازش‌گرها
preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# -------------------------------------------------------------
# ۳. ساخت پایپ‌لاین فشرده‌سازی با PCA
# -------------------------------------------------------------
# داده‌ها ابتدا پیش‌پردازش شده و سپس با PCA به ۲ بعد فشرده می‌شوند
full_prep_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=2, random_state=42))
])

print("\n🚀 در حال اجرای پیش‌پردازش و فشرده‌سازی با PCA...")
X_pca = full_prep_pipeline.fit_transform(raw_data)

# استخراج میزان واریانس حفظ شده
pca_step = full_prep_pipeline.named_steps['pca']
explained_variance = np.sum(pca_step.explained_variance_ratio_) * 100
print(f"✅ واریانس حفظ‌شده توسط ۲ مؤلفه اصلی PCA: {explained_variance:.2f}%")

# -------------------------------------------------------------
# ۴. خوشه‌بندی و جداسازی نویز با DBSCAN
# -------------------------------------------------------------
print("⚡ در حال جداسازی نویزها و خوشه‌بندی با DBSCAN...")
dbscan = DBSCAN(eps=0.45, min_samples=6)
cluster_labels = dbscan.fit_predict(X_pca)

# شمارش خوشه‌ها و نویزها
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)

print("\n" + "="*50)
print(f"🏆 تعداد خوشه‌های کشف‌شده توسط DBSCAN: {n_clusters}")
print(f"🚨 تعداد داده‌های پرت/نویز کشف‌شده (Outliers): {n_noise}")
print("="*50)

# -------------------------------------------------------------
# ۵. رسم و ذخیره‌سازی نمودار نهایی پروژه
# -------------------------------------------------------------
plt.figure(figsize=(10, 7))

# تفکیک داده‌های عادی از نویزها
is_noise = (cluster_labels == -1)

# رسم نقاط عادی (خوشه‌ها)
scatter = plt.scatter(
    X_pca[~is_noise, 0], 
    X_pca[~is_noise, 1], 
    c=cluster_labels[~is_noise], 
    cmap='tab10', 
    alpha=0.8, 
    edgecolors='k', 
    s=50,
    label='Clustered Data'
)

# رسم داده‌های نویز (Outliers) با علامت ضربدر مشکی
plt.scatter(
    X_pca[is_noise, 0], 
    X_pca[is_noise, 1], 
    color='black', 
    marker='X', 
    s=90, 
    linewidths=1.5,
    label=f'Noise / Outliers (N={n_noise})'
)

plt.title(
    f'Full End-to-End Unsupervised Pipeline\n(Preprocessing ➔ PCA 2D [{explained_variance:.1f}% Var] ➔ DBSCAN)', 
    fontsize=12, 
    fontweight='bold'
)
plt.xlabel('Principal Component 1 (PC1)', fontsize=11)
plt.ylabel('Principal Component 2 (PC2)', fontsize=11)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

output_file = 'full_unsupervised_pipeline_result.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ نمودار خروجی پروژه در فایل '{output_file}' ذخیره شد.")