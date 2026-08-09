import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

# ۱. ساخت یک DataFrame خام واقعی نمونه
np.random.seed(42)
data = pd.DataFrame({
    'Age': [25, np.nan, 45, 35, 50, 23],                  # عددی (دارای NaN)
    'Income': [50000, 60000, np.nan, 80000, 120000, 45000], # عددی (دارای NaN)
    'Education': ['HighSchool', 'Bachelor', 'Master', np.nan, 'Doctorate', 'Bachelor'], # دارای ترتیب
    'City': ['Tehran', 'Shiraz', 'Tehran', 'Isfahan', np.nan, 'Shiraz']                 # بدون ترتیب
})

y = np.array([0, 1, 0, 1, 1, 0])

# ۲. مشخص کردن ستون‌ها بر اساس نوع آن‌ها
num_cols = ['Age', 'Income']
ord_cols = ['Education']
nom_cols = ['City']

# ۳. تعریف پایپ‌لاین‌های اختصاصی برای هر نوع ویژگی
# الف) پایپ‌لاین عددی: پر کردن با میانه + مقیاس‌بندی با StandardScaler
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# ب) پایپ‌لاین کاتگوریکال ترتیبی: پر کردن با مد + OrdinalEncoder با ترتیب مشخص
education_order = [['HighSchool', 'Bachelor', 'Master', 'Doctorate']]
ord_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(categories=education_order))
])

# ج) پایپ‌لاین کاتگوریکال اسمی: پر کردن با مد + OneHotEncoder
nom_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
])

# ۴. ترکیب همه‌ی پایپ‌لاین‌ها در یک ColumnTransformer یکپارچه
preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_cols),
    ('ord', ord_pipeline, ord_cols),
    ('nom', nom_pipeline, nom_cols)
])

# ۵. تقسیم داده‌ها و اعمال پیش‌پردازش
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=42)

# فت کردن فقط روی Train و تبدیل هر دو
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("✅ داده‌های خام اولیه:")
print(X_train)
print("\n🚀 ماتریس پردازش‌شده و آماده برای مدل (بدون NaN، عددشده و مقیاس‌شده):")
print(np.round(X_train_processed, 2))