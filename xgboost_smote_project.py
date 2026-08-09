import matplotlib
matplotlib.use('Agg')  # جلوگیری از خطای گرافیکی در VS Code

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# -------------------------------------------------------------
# ۱. ساخت داده‌های شدیداً نامتوازن (۹۷٪ عادی، ۳٪ هدف/اقلیت)
# -------------------------------------------------------------
print("🔄 در حال ساخت مجموعه‌داده نامتوازن...")
X, y = make_classification(
    n_samples=2000, 
    n_features=5, 
    n_informative=4,
    n_redundant=1,
    n_clusters_per_class=1,
    weights=[0.97, 0.03],  # ۳ درصد کلاس مثبت (اقلیت)
    random_state=42
)

# -------------------------------------------------------------
# ۲. تقسیم داده‌ها به Train و Test (قبل از هرگونه نمونه‌برداری)
# -------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"📊 توزیع کل داده‌ها: {Counter(y)}")
print(f"🔹 توزیع داده‌های آموزش: {Counter(y_train)}")
print(f"🔹 توزیع داده‌های تست (دست‌نخورده): {Counter(y_test)}\n")

# -------------------------------------------------------------
# ۳. سناریو ۱: آموزش XGBoost بدون SMOTE
# -------------------------------------------------------------
print("⚙️ [سناریو ۱] آموزش مدل XGBoost بدون SMOTE...")
model_raw = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, eval_metric='logloss')
model_raw.fit(X_train, y_train)

# پیش‌بینی روی داده تست
y_pred_raw = model_raw.predict(X_test)
y_prob_raw = model_raw.predict_proba(X_test)[:, 1]

# -------------------------------------------------------------
# ۴. سناریو ۲: اعمال SMOTE (فقط روی Train) و آموزش XGBoost
# -------------------------------------------------------------
print("⚙️ [سناریو ۲] اعمال SMOTE روی داده‌های آموزش و آموزش مدل...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"   --> توزیع جدید آموزش پس از SMOTE: {Counter(y_train_resampled)}")

model_smote = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, eval_metric='logloss')
model_smote.fit(X_train_resampled, y_train_resampled)

# پیش‌بینی روی همان داده تست اولیه
y_pred_smote = model_smote.predict(X_test)
y_prob_smote = model_smote.predict_proba(X_test)[:, 1]

# -------------------------------------------------------------
# ۵. چاپ و مقایسه نتایج ترمینال
# -------------------------------------------------------------
print("\n" + "="*60)
print("📌 نتایج سناریو ۱: XGBoost بدون SMOTE")
print("="*60)
print(classification_report(y_test, y_pred_raw, target_names=['Normal (0)', 'Target (1)']))
print(f"AUC Score: {roc_auc_score(y_test, y_prob_raw):.4f}")

print("\n" + "="*60)
print("📌 نتایج سناریو ۲: XGBoost همراه با SMOTE")
print("="*60)
print(classification_report(y_test, y_pred_smote, target_names=['Normal (0)', 'Target (1)']))
print(f"AUC Score: {roc_auc_score(y_test, y_prob_smote):.4f}")

# -------------------------------------------------------------
# ۶. رسم ماتریس‌های درهم‌ریختگی مقایسه‌ای
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ماتریس سناریو ۱
cm_raw = confusion_matrix(y_test, y_pred_raw)
sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Normal (0)', 'Target (1)'],
            yticklabels=['Normal (0)', 'Target (1)'])
axes[0].set_title('Scenario 1: XGBoost WITHOUT SMOTE', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

# ماتریس سناریو ۲
cm_smote = confusion_matrix(y_test, y_pred_smote)
sns.heatmap(cm_smote, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['Normal (0)', 'Target (1)'],
            yticklabels=['Normal (0)', 'Target (1)'])
axes[1].set_title('Scenario 2: XGBoost WITH SMOTE', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')

plt.tight_layout()
output_file = 'xgboost_smote_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ نمودار مقایسه‌ای دو سناریو در فایل '{output_file}' ذخیره شد.")