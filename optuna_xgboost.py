import matplotlib
matplotlib.use('Agg')  # جلوگیری از خطای محیط گرافیکی در VS Code

import optuna
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score

# ۱. خاموش کردن لوگ‌های غیرضروری Optuna برای خروجی تمیزتر
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ۲. ساخت داده‌های نمونه برای مسئله طبقه‌بندی
X, y = make_classification(
    n_samples=1500,
    n_features=10,
    n_informative=8,
    n_redundant=2,
    weights=[0.85, 0.15],  # داده‌های نامتوازن
    random_state=42
)

# ۳. تعریف تابع هدف (Objective Function) برای Optuna
def objective(trial):
    # الف) تعریف فضای جستجوی هایپرپارامترها (Hyperparameter Space)
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0), # برای داده نامتوازن
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    
    # ب) تنظیم Cross-Validation با ۵ Fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # ج) ساخت مدل با پارامترهای پیشنهادی این مرحله
    model = XGBClassifier(**params)
    
    # د) ارزیابی مدل روی ۵ Fold با معیار F1-Score
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
    
    # ه) بازگرداندن میانگین امتیاز ۵ Fold
    return scores.mean()

# ۴. ساخت Study و اجرای بهینه‌سازی
print("🚀 در حال اجرای جستجوی هوشمند Optuna...")
study = optuna.create_study(direction='maximize')  # می‌خواهیم F1-Score حداکثر شود
study.optimize(objective, n_trials=30)             # انجام ۳۰ آزمایش هوشمند

# ۵. چاپ نتایج
print("\n" + "="*50)
print(f"🏆 بهترین F1-Score میانگین در CV: {study.best_value:.4f}")
print("🎯 بهترین هایپرپارامترهای کشف‌شده:")
for key, value in study.best_params.items():
    print(f"   • {key}: {value}")
print("="*50)

# -------------------------------------------------------------
# ۶. آموزش مدل نهایی با بهترین تنظیمات و رسم روند بهینه‌سازی
# -------------------------------------------------------------
best_model = XGBClassifier(**study.best_params, random_state=42, eval_metric='logloss')
best_model.fit(X, y)

# رسم تاریخچه آزمایش‌ها (Optimization History)
fig, ax = plt.subplots(figsize=(8, 5))
trial_scores = [t.value for t in study.trials if t.value is not None]
ax.plot(trial_scores, marker='o', color='purple', linestyle='-', linewidth=2)
ax.axhline(study.best_value, color='red', linestyle='--', label=f'Best Score ({study.best_value:.4f})')

ax.set_title('Optuna Optimization History (F1-Score vs Trials)', fontsize=12, fontweight='bold')
ax.set_xlabel('Trial Number', fontsize=11)
ax.set_ylabel('Mean F1-Score (5-Fold CV)', fontsize=11)
ax.legend()
ax.grid(True, alpha=0.3)

output_file = 'optuna_optimization_history.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ نمودار روند بهینه‌سازی در فایل '{output_file}' ذخیره شد.")