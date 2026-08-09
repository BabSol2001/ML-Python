import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from collections import Counter

# ۱. ساخت داده‌های شدیداً نامتوازن (۹۵٪ کلاس ۰ و ۵٪ کلاس ۱)
X, y = make_classification(
    n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, weights=[0.95, 0.05], random_state=42
)

print(f"تعداد داده‌ها قبل از SMOTE: {Counter(y)}")
# خروجی: Counter({0: 943, 1: 57})

# ۲. اعمال تکنیک SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"تعداد داده‌ها بعد از SMOTE: {Counter(y_resampled)}")
# خروجی: Counter({0: 943, 1: 943})

# ۳. رسم نمودار مقایسه‌ای قبل و بعد از SMOTE
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# قبل از SMOTE
axes[0].scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Majority (0)', alpha=0.6, c='blue')
axes[0].scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Minority (1)', alpha=0.9, c='red')
axes[0].set_title(f'Before SMOTE {Counter(y)}', fontsize=12, fontweight='bold')
axes[0].legend()

# بعد از SMOTE
axes[1].scatter(X_resampled[y_resampled == 0][:, 0], X_resampled[y_resampled == 0][:, 1], label='Majority (0)', alpha=0.4, c='blue')
axes[1].scatter(X_resampled[y_resampled == 1][:, 0], X_resampled[y_resampled == 1][:, 1], label='Synthetic Minority (1)', alpha=0.7, c='red')
axes[1].set_title(f'After SMOTE {Counter(y_resampled)}', fontsize=12, fontweight='bold')
axes[1].legend()

plt.tight_layout()
output_file = 'smote_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ تصویر مقایسه SMOTE در '{output_file}' ذخیره شد.")