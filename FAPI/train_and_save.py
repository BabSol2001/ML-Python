import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline

# ۱. ساخت داده‌های ساختگی نمونه
np.random.seed(42)
n_samples = 500

df = pd.DataFrame({
    'age': np.random.choice([25, 35, 45, 50, np.nan], size=n_samples),
    'income': np.random.choice([40000, 70000, 100000, np.nan], size=n_samples),
    'education': np.random.choice(['HighSchool', 'Bachelor', 'Master'], size=n_samples),
    'city': np.random.choice(['Tehran', 'Shiraz', 'Isfahan'], size=n_samples)
})
y = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])

# ۲. تعریف ColumnTransformer برای پیش‌پردازش
num_cols = ['age', 'income']
cat_cols = ['education', 'city']

num_pipeline = ImbPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = ImbPipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# ۳. پایپ‌لاین کامل (پیش‌پردازش + مدل)
full_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(n_estimators=50, random_state=42, eval_metric='logloss'))
])

# ۴. آموزش روی کل داده‌ها
full_pipeline.fit(df, y)

# ۵. ذخیره پایپ‌لاین کامل روی هارد دیسک
model_filename = 'pipeline_model.joblib'
joblib.dump(full_pipeline, model_filename)

print(f"✅ پایپ‌لاین کامل با موفقیت در فایل '{model_filename}' ذخیره شد.")