import matplotlib
matplotlib.use('Agg')  # جلوگیری از خطای محیط گرافیکی در VS Code

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# -------------------------------------------------------------
# ۱. ساخت یک DataFrame خام واقعی (شامل NaN، ستون متنی و ناهمگونی)
# -------------------------------------------------------------
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'Age': np.random.choice([25, 30, 45, 50, np.nan], size=n_samples),       # عددی (دارای NaN)
    'Income': np.random.choice([30000, 60000, 90000, np.nan], size=n_samples), # عددی (دارای NaN)
    'Education': np.random.choice(['HighSchool', 'Bachelor', 'Master', np.nan], size=n_samples), # کاتگوریکال (دارای NaN)
    'City': np.random.choice(['Tehran', 'Shiraz', 'Isfahan'], size=n_samples) # کاتگوریکال
})

# برچسب هدف نامتوازن (۹۲٪ صفر و ۸٪ یک)
y = np.random.choice([0, 1], size=n_samples, p=[0.92, 0.08])

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42, stratify=y)

# -------------------------------------------------------------
# ۲. تعریف تفکیک‌کننده پیش‌پردازش ستون‌ها (ColumnTransformer)
# -------------------------------------------------------------
# ستون‌های عددی و کاتگوریکال را جدا می‌کنیم
numeric_features = ['Age', 'Income']
categorical_features = ['Education', 'City']

# پایپ‌لاین اختصاصی برای ستون‌های عددی (پر کردن جای خالی + مقیاس‌بندی)
num_transformer = ImbPipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# پایپ‌لاین اختصاصی برای ستون‌های کاتگوریکال (پر کردن جای خالی + One-Hot Encoding)
cat_transformer = ImbPipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# ترکیب پیش‌پردازش‌گرها با ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numeric_features),
    ('cat', cat_transformer, categorical_features)
])

# -------------------------------------------------------------
# ۳. ساخت پایپ‌لاین اصلی (پیش‌پردازش -> SMOTE -> مدل XGBoost)
# -------------------------------------------------------------
full_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(n_estimators=100, max_depth=4, random_state=42, eval_metric='logloss'))
])

# -------------------------------------------------------------
# ۴. آموزش و ارزیابی (تنها با یک خط fit و یک خط predict!)
# -------------------------------------------------------------
print("🚀 در حال آموزش پایپ‌لاین یکپارچه...")
full_pipeline.fit(X_train, y_train)

# پیش‌بینی روی داده‌های تست
y_pred = full_pipeline.predict(X_test)
y_probs = full_pipeline.predict_proba(X_test)[:, 1]

print("\n" + "="*50)
print("📊 گزارش ارزیابی مدل روی داده‌های تست:")
print("="*50)
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Target (1)']))
print(f"🎯 AUC Score: {roc_auc_score(y_test, y_probs):.4f}")